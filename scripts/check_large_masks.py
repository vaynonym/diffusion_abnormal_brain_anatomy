#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import numpy as np
import matplotlib.pyplot as plt

from src.util import load_wand_credentials, Stopwatch, read_config
from src.model_util import load_model_from_run_with_matching_config, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import synthseg_classes, ventricle_indices, white_matter_indices, cortex_indices, CSF_indices, background_indices, decode_one_hot



import torch.multiprocessing
from torch.utils.data import TensorDataset


from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")

experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
run_config = experiment_config["run"]
auto_encoder_config = experiment_config["auto_encoder"]
patch_discrim_config = experiment_config["patch_discrim"]
auto_encoder_training_config = experiment_config["autoencoder_training"]
diffusion_model_unet_config = experiment_config["diffusion_model_unet"]
diffusion_model_training_config = experiment_config["diffusion_model_unet_training"]

check_dimensions(run_config, auto_encoder_config, diffusion_model_unet_config)

project, entity = load_wand_credentials()

import wandb
wandb_run = wandb.init(project=project, entity=entity, name=WANDB_RUN_NAME, 
           config={"run_config": run_config,
                   "auto_encoder_config": auto_encoder_config, 
                   "patch_discrim_config": patch_discrim_config,
                   "auto_encoder_training_config": auto_encoder_training_config,
                   "diffusion_model_unet_config": diffusion_model_unet_config,
                   "diffusion_model_training_config": diffusion_model_training_config,
                   })

# for reproducibility purposes set a seed
set_determinism(42)



down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))

train_transforms = transforms.Compose(
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
    ]
)

LOGGER.info("Loading Data")

dataset_size = run_config["dataset_size"]

validation_ds = SyntheticLDM100K(
    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
    section="validation",  # validation
    size=dataset_size,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    seed=0,
    transform=train_transforms,
)


valid_loader = DataLoader(validation_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)


LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Mask shape {validation_ds[0]["mask"].shape}')

from src.util import log_image_with_mask

for batch in iter(valid_loader):
    for batch_index in range(2):
        volume = batch["volume"][batch_index, 0].detach().cpu()
        if volume > 0.95:
            original_image = batch["image"][batch_index, 0].detach().cpu().numpy()
            ground_truth_mask = batch["mask"][batch_index, 0].detach().cpu().numpy()

            
            log_image_with_mask(image=original_image, 
                                original_mask=ground_truth_mask, 
                                reconstruction_image=None,
                                reconstruction_mask=None,
                                description_prefix=f"large_volumes/sample_images",
                                conditioning_information=volume)
                