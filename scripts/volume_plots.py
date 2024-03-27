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
from generative.networks.schedulers import DDPMScheduler

import numpy as np

from src.util import load_wand_credentials, log_image_to_wandb, Stopwatch, read_config, visualize_reconstructions
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor


import torch.multiprocessing

from src.torch_setup import device

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

# for reproducibility purposes set a seed
set_determinism(42)



down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))

#encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
#LOGGER.info(f"Encoding shape: {encoding_shape}")
#
##autoencoder = AutoencoderKL(
#    **auto_encoder_config
#)
#autoencoder.to(device)
#
#
## Try to load identically trained autoencoder if it already exists. Else train a new one.
#if not load_model_from_run_with_matching_config([run_config, auto_encoder_config, auto_encoder_training_config],
#                                            ["run_config", "auto_encoder_config", "auto_encoder_training_config"],
#                                            project=project, entity=entity, 
#                                            model=autoencoder, artifact_name=autoencoder.__class__.__name__,
#                                            ):
#    LOGGER.error("This script expects existing autoencoder")
#    quit()
#else:
#    LOGGER.info("Loaded existing autoencoder")
#
#
#unet = DiffusionModelUNet(
#    **diffusion_model_unet_config
#)
#unet.to(device)
#
#if not load_model_from_run_with_matching_config([ run_config, auto_encoder_config, auto_encoder_training_config, diffusion_model_training_config, diffusion_model_unet_config],
#                                            ["run_config", "auto_encoder_config", "auto_encoder_training_config", "diffusion_model_training_config", "diffusion_model_unet_config"],
#                                            project=project, entity=entity, 
#                                            model=unet, artifact_name=unet.__class__.__name__,
#                                            ):
#    LOGGER.error("This script expects existing Diffusion Model")
#    quit()
#else:
#    LOGGER.info("Loaded existing Diffusion Model")

LOGGER.info("Loading dataset...")


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


#scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)


#scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data= next(iter(valid_loader))["mask"].int().to(device))
#LOGGER.info(f"Scaling factor set to {scale_factor}")
# -

# We define the inferer using the scale factor:

#inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

#autoencoder.eval()

from collections import defaultdict
from src.synthseg_masks import ventricle_indices, white_matter_indices
import numpy as np


stopwatch = Stopwatch("Calculating volumes over validationset took: ").start()
bucket_size = 0.1
buckets = torch.tensor(np.arange(0, 1.01, bucket_size)).to(device)
results_ventricles = {[] for i in range(len(buckets))}
results_white_matter = {[] for i in range(len(buckets))}
for i, batch in enumerate(valid_loader):
    mask = batch["mask"].int().to(device)
    conditioning_value = batch["volume"].to(device)
    ventricle_voxel_count = torch.isin(mask, ventricle_indices).sum(dim=(1,2,3,4))
    white_matter_voxel_count = torch.isin(mask, white_matter_indices).sum(dim=(1,2,3,4))
    
    buckets = torch.bucketize(conditioning_value, buckets)

    for batch_i in range(mask.shape[0]):
        results_ventricles[buckets[batch_i].item()].append(ventricle_voxel_count[batch_i])
        results_white_matter[buckets[batch_i].item()].append(white_matter_voxel_count[batch_i])

results_ventricles = {k:torch.stack(v).mean().item() for k, v in results_ventricles}
results_white_matter = {k:torch.stack(v).mean().item() for k, v in results_white_matter}

stopwatch.stop()


stopwatch = Stopwatch("drawing plots took: ")

import matplotlib.pyplot as plt

plt.bar(x=np.arange(0, 1.0, bucket_size) + 0.5, width=1, height=results_ventricles.values())
plt.savefig(f"{OUTPUT_DIRECTORY}/ground_truth_ventricle_volume.png")
plt.clf()

plt.bar(x=np.arange(0, 1.0, bucket_size) + 0.5, width=1, height=results_white_matter.values())
plt.savefig(f"{OUTPUT_DIRECTORY}/ground_truth_white_matter_volume.png")
plt.clf()

stopwatch.stop()

LOGGER.info("DONE!")


    

