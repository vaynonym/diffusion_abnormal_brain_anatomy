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

from src.util import load_wand_credentials, log_image_to_wandb, Stopwatch, device, read_config, visualize_reconstructions
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, check_dimensions
from src.training import train_autoencoder, train_diffusion_model
from src.logging_util import LOGGER
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor, generate_and_log_sample_images
from src.directory_management import DATA_DIRECTORY

import torch.multiprocessing

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

def peek(x):
    LOGGER.info(x)
    return x


train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        #transforms.Lambdad(keys="image", func=peek_shape),
        #transforms.Lambdad(keys="volume", func=peek),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0.5, upper=99.5, b_min=0, b_max=1),
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
valid_loader = DataLoader(validation_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=6, drop_last=True, persistent_workers=True)

dataset_preparation_stopwatch.stop().display()

LOGGER.info(f"Train length: {len(train_ds)} in {len(train_loader)} batches")
LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Image shape {train_ds[0]["image"].shape}')

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
    img = sample_data["image"][sample_index, 0].detach().cpu().numpy()
    log_image_to_wandb(img, None, "trainingdata", WANDB_LOG_IMAGES, conditioning_information=sample_data["volume"][sample_index].unsqueeze(0))

LOGGER.info(f"Batch image shape: {sample_data['image'].shape}")

autoencoder = AutoencoderKL(**auto_encoder_config).to(device)

# Try to load identically trained autoencoder if it already exists. Else train a new one.
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config],
                                            ["auto_encoder_config", "auto_encoder_training_config"],
                                            project=project, entity=entity, 
                                            model=autoencoder, artifact_name=AutoencoderKL.__name__,
                                            ):
    LOGGER.info("Training new autoencoder...")
    autoencoder = train_autoencoder(autoencoder, train_loader, valid_loader,
                                                    patch_discrim_config, auto_encoder_training_config, run_config["evaluation_intervall"],
                                                    WANDB_LOG_IMAGES)

    save_model_as_artifact(wandb_run, autoencoder, type(autoencoder).__name__, auto_encoder_config)
else:
    LOGGER.info("Loaded existing autoencoder")

visualize_reconstructions(train_loader, autoencoder, 10)


unet = DiffusionModelUNet(**diffusion_model_unet_config).to(device)
scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)

# We define the inferer using the scale factor:
scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=sample_data["image"].to(device))
inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

optimizer_diff = torch.optim.Adam(params=unet.parameters(), 
                                  lr=diffusion_model_training_config["learning_rate"])
scaler = GradScaler()

LOGGER.info("Training new diffusion model...")
with Stopwatch("Diffusion training took:"):
    train_diffusion_model(autoencoder=autoencoder,
                          unet=unet, 
                          optimizer_diff=optimizer_diff, 
                          scaler=scaler, 
                          inferer=inferer, 
                          train_loader=train_loader,
                          valid_loader=valid_loader,
                          encoding_shape=encoding_shape,
                          n_epochs=diffusion_model_training_config["n_epochs"],
                          evaluation_intervall=run_config["evaluation_intervall"])

LOGGER.info("Saving diffusion model as artifact")
save_model_as_artifact(wandb_run, unet, type(unet).__name__, diffusion_model_unet_config)

LOGGER.info("Sampling from trained diffusion model...")
for i in range(5):
    generate_and_log_sample_images( autoencoder=autoencoder, unet=unet, 
                                    scheduler=scheduler,
                                    encoding_shape=encoding_shape, 
                                    inferer=inferer,
                                    prefix_string="diffusion/synthetic images")

wandb.finish()
