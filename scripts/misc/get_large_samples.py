#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
from monai import transforms
from monai.utils import set_determinism
from torch.cuda.amp import GradScaler, autocast

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.evaluation import create_fake_volume_dataloader
from src.util import load_wand_credentials, Stopwatch, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, load_model_from_run, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K, get_dataloader

from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import synthseg_classes, central_areas_close_to_ventricles_indices, ventricle_indices, white_matter_indices, cortex_indices, CSF_indices, background_indices, decode_one_hot
from src.evaluation import MaskDiffusionModelEvaluator



import torch.multiprocessing


from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")
WANDB_AUTOENCODER_RUNID = os.environ.get("WANDB_AUTOENCODER_RUNID")
WANDB_DIFFUSIONMODEL_RUNID = os.environ.get("WANDB_DIFFUSIONMODEL_RUNID")


project, entity = load_wand_credentials()

experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))

run_config = experiment_config["run"]
run_config["oversample_large_ventricles"] = False
LOGGER.info(f"Oversample large ventricles: {run_config['oversample_large_ventricles']}")

# for reproducibility purposes set a seed
set_determinism(42)

image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")


LOGGER.info("Loading dataset...")

CLASSIFIER_FREE_GUIDANCE = False

_, val_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["mask", "image"]),
        transforms.EnsureChannelFirstd(keys=["mask", "image"]),
        transforms.EnsureTyped(keys=["mask", "image"]),
        transforms.Orientationd(keys=["mask", "image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["mask"], pixdim=run_config["input_image_downsampling_factors"], mode=("nearest")),
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["mask", "image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
        transforms.Lambdad(keys=["volume"], func = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)),
                # Use extra conditioning variable to signify conditioned or non-conditioned case
        transforms.Lambdad(keys=["volume"], 
                           func = lambda x: (torch.tensor([x, 1.], dtype=torch.float32) 
                                             if CLASSIFIER_FREE_GUIDANCE 
                                             else torch.tensor([x], dtype=torch.float32)).unsqueeze(0)),
    ]
)

dataset_size = run_config["dataset_size"]

train_ds = SyntheticLDM100K(
    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
    section="training",  # validation
    size=dataset_size,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    val_frac=0.0,
    test_frac=0.0,
    seed=0,
    transform=val_transforms,
    sorted_by_volume=False,
    filter_function=lambda d: d["volume"] >= 0.9,
)

train_loader = get_dataloader(train_ds, run_config)

LOGGER.info(f"Valid length: {len(train_ds)} in {len(train_loader)} batches")
LOGGER.info(f'Mask shape {train_ds[0]["mask"].shape}')
batch_size = iter(train_loader).__next__()["image"].shape[0]

from src.util import visualize_3d_image_slice_wise

for j, batch in enumerate(train_loader):
    for i in range(batch_size):
        img = batch["image"][i, 0].detach().cpu()
        mask = batch["mask"][i, 0].int().detach().cpu()
        condition = batch["volume"][i, 0]
        visualize_3d_image_slice_wise(img, OUTPUT_DIRECTORY + f"/{j * batch_size + i:03d}_img.png", description_prefix="", log_to_wandb=False, conditioning_information=condition)
        visualize_3d_image_slice_wise(mask, OUTPUT_DIRECTORY + f"/{j * batch_size + i:03d}_mask.png", description_prefix="", log_to_wandb=False, conditioning_information=condition,
                                      is_image_mask=True)



