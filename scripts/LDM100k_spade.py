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
from generative.networks.nets import SPADEAutoencoderKL, SPADEDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

import wandb
import numpy as np

from src.util import load_wand_credentials, log_image_to_wandb, Stopwatch, read_config, visualize_reconstructions, log_image_with_mask
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, check_dimensions
from src.training import train_diffusion_model
from src.logging_util import LOGGER
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor, generate_and_log_sample_images
from src.directory_management import DATA_DIRECTORY
from src.trainer import SpadeAutoencoderTrainer
from src.evaluation import SpadeAutoencoderEvaluator
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import synthseg_classes

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

# for reproducibility purposes set a seed
set_determinism(42)

def peek_shape(x):
    LOGGER.info(x.shape)
    return x

def peek_max(x):
    LOGGER.info(f"max {x.max()}")
    return x

def peek_min(x):
    LOGGER.info(f"min {x.min()}")
    return x

def peek(x):
    LOGGER.info(x)
    return x


train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["mask", "image"]),
        transforms.EnsureChannelFirstd(keys=["mask", "image"]),
        transforms.EnsureTyped(keys=["mask", "image"]),
        transforms.Orientationd(keys=["mask", "image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["mask"], pixdim=run_config["input_image_downsampling_factors"], mode=("nearest")),
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["mask", "image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=1, upper=99, b_min=0, b_max=1, clip=True),
        transforms.Lambdad(keys=["volume"], func = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)),
    ]
)


LOGGER.info("Loading dataset...")
dataset_preparation_stopwatch = Stopwatch("Done! Loading the Dataset took: ").start()

dataset_size = run_config["dataset_size"]

train_ds = SyntheticLDM100K(
    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
    section="training",
    size=dataset_size,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    seed=0,
    transform=train_transforms,
)

validation_ds = SyntheticLDM100K(
    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
    section="validation",  # validation
    size=dataset_size,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    seed=0,
    transform=train_transforms,
)


train_loader = DataLoader(train_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)
valid_loader = DataLoader(validation_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=4, drop_last=True, persistent_workers=True)

dataset_preparation_stopwatch.stop().display()

LOGGER.info(f"Train length: {len(train_ds)} in {len(train_loader)} batches")
LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Mask shape {train_ds[0]["mask"].shape}')

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")

# ### Visualise examples from the training set
iterator = enumerate(train_loader)
sample_data = None # reused later
for i in range(3):
    _, sample_data = next(iterator)
    sample_index = 0
    mask = sample_data["mask"][sample_index, 0].detach().cpu().numpy()
    image = sample_data["image"][sample_index, 0].detach().cpu().numpy()
    log_image_with_mask(image, mask, None, None, "trainingdata", sample_data["volume"][sample_index])


LOGGER.info(f"Batch image shape: {sample_data['mask'].shape}")

autoencoder = SPADEAutoencoderKL(**auto_encoder_config).to(device)

if not load_model_from_run_with_matching_config([run_config, auto_encoder_config, auto_encoder_training_config, patch_discrim_config],
                                            ["run_config", "auto_encoder_config", "auto_encoder_training_config", "patch_discrim_config"],
                                            project=project, entity=entity,
                                            model=autoencoder, artifact_name=autoencoder.__class__.__name__,
                                            ):
    LOGGER.info("Training new autoencoder...")
    trainer = SpadeAutoencoderTrainer(autoencoder=autoencoder, 
                                 train_loader=train_loader, val_loader=valid_loader, 
                                 patch_discrim_config=patch_discrim_config, auto_encoder_training_config=auto_encoder_training_config,
                                 WANDB_LOG_IMAGES=WANDB_LOG_IMAGES,
                                 evaluation_intervall=run_config["evaluation_intervall"],
                                 starting_epoch=0
                                 )
    
    with Stopwatch("Training took: "):
        autoencoder = trainer.train()

    # clean up fully
    del trainer

    save_model_as_artifact(wandb_run, autoencoder, type(autoencoder).__name__, auto_encoder_config)
else:
    LOGGER.info("Loaded existing autoencoder")

with Stopwatch("Evaluating Autoencoder took: "):
    evaluator = SpadeAutoencoderEvaluator(autoencoder=autoencoder, val_loader=valid_loader, wandb_prefix="autoencoder/evaluation")
    evaluator.visualize_batches(5)
    evaluator.evaluate()
    del evaluator

quit()