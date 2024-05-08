#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.utils import  set_determinism

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import wandb
import numpy as np

from src.util import load_wand_credentials, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, check_dimensions
from src.evaluation import MaskDiffusionModelEvaluator
from src.custom_autoencoders import EmbeddingWrapper
from src.logging_util import LOGGER

import torch.multiprocessing

from src.torch_setup import device

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")
WANDB_AUTOENCODER_RUNID = os.environ.get("WANDB_AUTOENCODER_RUNID")
WANDB_DIFFUSIONMODEL_RUNID = os.environ.get("WANDB_DIFFUSIONMODEL_RUNID")

project, entity = load_wand_credentials()


experiment_config = read_config_from_wandb_run(entity, project, WANDB_RUN_NAME)
run_config = experiment_config["run"]
auto_encoder_config = experiment_config["auto_encoder"]
patch_discrim_config = experiment_config["patch_discrim"]
auto_encoder_training_config = experiment_config["autoencoder_training"]
diffusion_model_unet_config = experiment_config["diffusion_model_unet"]
diffusion_model_training_config = experiment_config["diffusion_model_unet_training"]

check_dimensions(run_config, auto_encoder_config, diffusion_model_unet_config)

diffusion_model_training_config["classifier_free_guidance"] = 4.0

WANDB_RUN_NAME += " " + str(diffusion_model_training_config["classifier_free_guidance"] )

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

CLASSIFIER_FREE_GUIDANCE = diffusion_model_training_config["classifier_free_guidance"] if "classifier_free_guidance" in diffusion_model_training_config else None
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")

from src.synthseg_masks import synthseg_classes

def create_embedding_autoencoder(*args, **kwargs):
    base_autoencoder = AutoencoderKL(*args, **kwargs)
    autoencoder = EmbeddingWrapper(base_autoencoder=base_autoencoder, vocabulary_size=max(synthseg_classes) + 1, embedding_dim=64)
    return autoencoder

from src.model_util import load_model_from_run

autoencoder = load_model_from_run(run_id=WANDB_AUTOENCODER_RUNID, project=project, entity=entity,
                                  #model_class=AutoencoderKL,
                                  model_class=EmbeddingWrapper,  
                                  #create_model_from_config=None,
                                  create_model_from_config=create_embedding_autoencoder
                                  )


unet = load_model_from_run(run_id=WANDB_DIFFUSIONMODEL_RUNID, project=project, entity=entity,
                                      model_class=DiffusionModelUNet, 
                                      create_model_from_config=None
                                     )


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
scheduler.set_timesteps(1000)
evaluation_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
evaluation_scheduler.set_timesteps(50)

# REMOVING SCALING FACTOR FOR SIMPLICITY, MIGHT AFFECT PERFORMANCE

scale_factor = 1.0
LOGGER.info(f"Scaling factor set to {scale_factor}")
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

dataset = torch.arange(0.85, 1.5, 0.1).unsqueeze(0).repeat(3, 1).flatten().unsqueeze(1).unsqueeze(2)

from torch.utils.data import TensorDataset

fake_volume_dataset = TensorDataset(dataset)

def collate(data):
    volume = torch.stack([x[0] for x in data])

    return {"volume": volume,
            "image": torch.zeros(image_shape), 
            "mask": torch.zeros(image_shape)}

def collate_guidance(data):
    volume = torch.stack([x[0] for x in data])
    volume = torch.concat([volume, torch.ones_like(volume)], dim=2)
    return {"volume": volume,
            "image": torch.zeros(image_shape), 
            "mask": torch.zeros(image_shape)}

fake_volume_dataloader = DataLoader(fake_volume_dataset,
                                    batch_size=run_config["batch_size"],
                                    shuffle=False, num_workers=2,
                                    drop_last=True,
                                    persistent_workers=True,
                                    collate_fn=collate_guidance if CLASSIFIER_FREE_GUIDANCE else collate
                                    )

autoencoder.eval()
unet.eval()

MaskDiffusionModelEvaluator(
                 diffusion_model=unet,
                 autoencoder=autoencoder,
                 latent_shape=encoding_shape,
                 inferer=inferer,
                 val_loader=fake_volume_dataloader,
                 train_loader=fake_volume_dataloader,
                 wandb_prefix="abnormal_mask/DDIM",
                 evaluation_scheduler=evaluation_scheduler,
                 guidance=CLASSIFIER_FREE_GUIDANCE
                 ).log_samples(500, True)

MaskDiffusionModelEvaluator(
                 diffusion_model=unet,
                 autoencoder=autoencoder,
                 latent_shape=encoding_shape,
                 inferer=inferer,
                 val_loader=fake_volume_dataloader,
                 train_loader=fake_volume_dataloader,
                 wandb_prefix="abnormal_mask/DDPM",
                 evaluation_scheduler=evaluation_scheduler,
                 guidance=CLASSIFIER_FREE_GUIDANCE
                 ).log_samples(500, False)


# ## Clean-up data

wandb.finish()
