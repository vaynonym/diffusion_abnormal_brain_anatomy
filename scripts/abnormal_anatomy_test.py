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

import wandb
import numpy as np

from src.util import load_wand_credentials, log_image_to_wandb, Stopwatch, read_config, visualize_reconstructions
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, check_dimensions
from src.training import train_autoencoder
from src.logging_util import LOGGER
from src.datasets import SyntheticLDM100K
from src.evaluation import evaluate_diffusion_model

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
wandb_run = wandb.init(project=project, entity=entity, name=WANDB_RUN_NAME, 
           config={"run_config": run_config,
                   "auto_encoder_config": auto_encoder_config, 
                   "patch_discrim_config": patch_discrim_config,
                   "auto_encoder_training_config": auto_encoder_training_config,
                   "diffusion_model_unet_config": diffusion_model_unet_config,
                   "diffusion_model_training_config": diffusion_model_training_config,
                   })

autoencoder = AutoencoderKL(
    **auto_encoder_config
)

# for reproducibility purposes set a seed
set_determinism(42)



down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")

autoencoder = AutoencoderKL(
    **auto_encoder_config
)
autoencoder.to(device)


# Try to load identically trained autoencoder if it already exists. Else train a new one.
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config],
                                            ["auto_encoder_config", "auto_encoder_training_config"],
                                            project=project, entity=entity, 
                                            model=autoencoder, artifact_name=AutoencoderKL.__name__,
                                            ):
    LOGGER.error("This script expects existing autoencoder")
    quit()
else:
    LOGGER.info("Loaded existing autoencoder")


unet = DiffusionModelUNet(
    **diffusion_model_unet_config
)
unet.to(device)

if not load_model_from_run_with_matching_config([diffusion_model_training_config, diffusion_model_unet_config],
                                            ["diffusion_model_training_config", "diffusion_model_unet_config"],
                                            project=project, entity=entity, 
                                            model=unet, artifact_name=DiffusionModelUNet.__name__,
                                            ):
    LOGGER.error("This script expects existing Diffusion Model")
    quit()
else:
    LOGGER.info("Loaded existing Diffusion Model")


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)

# REMOVING SCALING FACTOR FOR SIMPLICITY, MIGHT AFFECT PERFORMANCE

scale_factor = 1.0
LOGGER.info(f"Scaling factor set to {scale_factor}")
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

autoencoder.eval()
import random
def generate_abnormal_sample_images(autoencoder, unet, scheduler, prefix_string):
    autoencoder.eval()
    unet.eval()
    with torch.no_grad():
        for offset in [0.1, 0.2, 0.3, 0.4, 0.5, 1.0, 2.0, -1.1, -1.2, -1.3, -1.4]:
            conditioning = (torch.tensor([[[1]]] * run_config["batch_size"]) + offset).to(device)
            latent_noise = torch.randn(encoding_shape).to(device)

            scheduler.set_timesteps(num_inference_steps=1000)
            synthetic_images = inferer.sample(input_noise=latent_noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler, conditioning=conditioning)
            synthetic_images = torch.clamp(synthetic_images, 0., 1.)

            # ### Visualise synthetic data
            for sample_index in range(run_config["batch_size"]):
                img = synthetic_images[sample_index, 0].detach().cpu().numpy()  # images
                log_image_to_wandb(img, None, prefix_string, WANDB_LOG_IMAGES, conditioning[sample_index, 0].detach().cpu())

with Stopwatch("Diffusion sampling took: ").start():
    generate_abnormal_sample_images(autoencoder=autoencoder, unet=unet, scheduler=scheduler, prefix_string="Abnormal samples:")

# ## Clean-up data

wandb.finish()
