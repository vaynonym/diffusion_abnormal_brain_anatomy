#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
from monai.data import DataLoader
from monai.utils import  set_determinism

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import SPADEAutoencoderKL, SPADEDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import wandb
import numpy as np

from src.util import load_wand_credentials, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, check_dimensions
from src.evaluation import MaskDiffusionModelEvaluator
from src.custom_autoencoders import EmbeddingWrapper
from src.directory_management import DATA_DIRECTORY
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

from src.model_util import load_model_from_run
from src.datasets import AbnormalSyntheticMaskDataset

autoencoder = load_model_from_run(run_id=WANDB_AUTOENCODER_RUNID, project=project, entity=entity,
                                  #model_class=AutoencoderKL,
                                  model_class=SPADEAutoencoderKL,  
                                  create_model_from_config=None,
                                  #create_model_from_config=create_embedding_autoencoder
                                  )


diffusion_model = load_model_from_run(run_id=WANDB_DIFFUSIONMODEL_RUNID, project=project, entity=entity,
                                      model_class=SPADEDiffusionModelUNet, 
                                      create_model_from_config=None
                                     )


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
evaluation_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
evaluation_scheduler.set_timesteps(50)


scale_factor = 1.0
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

from monai import transforms

def peek(mask):
    print(f"Actual size: {mask.shape}, expected: {run_config['input_image_crop_roi']}")
    return mask



train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["mask"]),
        transforms.EnsureChannelFirstd(keys=["mask"]),
        transforms.EnsureTyped(keys=["mask"]),
        #transforms.Orientationd(keys=["mask", "image"], axcodes="IPL"), # axcodes="RAS"
        #transforms.Spacingd(keys=["mask"], pixdim=run_config["input_image_downsampling_factors"], mode=("nearest")),
        transforms.CenterSpatialCropd(keys=["mask"], roi_size=run_config["input_image_crop_roi"]),
        transforms.Lambdad(keys=["mask"], func=peek),
        transforms.Lambdad(keys=["volume"], func = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)),
    ]
)


dataset = AbnormalSyntheticMaskDataset(
    dataset_path=os.path.join(DATA_DIRECTORY, "abnormal_masks"),
    section="training",
    val_frac=0.0,
    test_frac=0.0,
    size=20,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=2,
    seed=0,
    transform=train_transforms,
)

train_loader = DataLoader(dataset, batch_size=run_config["batch_size"], shuffle=True, num_workers=2, persistent_workers=True, drop_last=False)
print(len(train_loader))

LOGGER.info(f"mask shape: {iter(train_loader).__next__()['mask'].shape}")

from src.evaluation import SpadeDiffusionModelEvaluator

evaluator = SpadeDiffusionModelEvaluator(
                 diffusion_model=diffusion_model,
                 autoencoder=autoencoder,
                 latent_shape=encoding_shape,
                 inferer=inferer,
                 val_loader=train_loader,
                 train_loader=train_loader,
                 wandb_prefix="diffusion/evaluation",
                 evaluation_scheduler=evaluation_scheduler,
                 guidance=CLASSIFIER_FREE_GUIDANCE
                 )
#evaluator.log_samples(16, False)

batch_size = run_config["batch_size"]

from src.util import visualize_3d_image_slice_wise
from src.directory_management import OUTPUT_DIRECTORY

for j, batch in enumerate(train_loader):
    images = evaluator.image_preprocessing(evaluator.get_synthetic_output(batch, False))
    for i in range(batch_size):
        img = images[i, 0].detach().cpu()
        mask = batch["mask"][i, 0].int().detach().cpu()
        condition = batch["volume"][i, 0]
        visualize_3d_image_slice_wise(img, OUTPUT_DIRECTORY + f"/{j * batch_size + i:03d}_img.png", description_prefix="", log_to_wandb=False, conditioning_information=condition)
        visualize_3d_image_slice_wise(mask, OUTPUT_DIRECTORY + f"/{j * batch_size + i:03d}_mask.png", description_prefix="", log_to_wandb=False, conditioning_information=condition,
                                      is_image_mask=True)


