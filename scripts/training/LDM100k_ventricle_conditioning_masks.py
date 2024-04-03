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

import wandb
import numpy as np

from src.util import load_wand_credentials, Stopwatch, read_config, log_image_with_mask
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, check_dimensions
from src.training import train_diffusion_model
from src.logging_util import LOGGER
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor
from src.directory_management import DATA_DIRECTORY
from src.trainer import MaskAutoencoderTrainer, MaskEmbeddingAutoencoderTrainer, MaskDiffusionModelTrainer, MaskEmbeddingDiffusionModelTrainer
from src.evaluation import MaskAutoencoderEvaluator, MaskEmbeddingAutoencoderEvaluator, MaskDiffusionModelEvaluator
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

CLASSIFIER_FREE_GUIDANCE = "classifier_free_guidance" in diffusion_model_training_config and diffusion_model_training_config["classifier_free_guidance"]

import random

base_transforms = [
        transforms.LoadImaged(keys=["mask", "image"]),
        transforms.EnsureChannelFirstd(keys=["mask", "image"]),
        transforms.EnsureTyped(keys=["mask", "image"]),
        transforms.Orientationd(keys=["mask", "image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["mask"], pixdim=run_config["input_image_downsampling_factors"], mode=("nearest")),
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["mask", "image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),

        # Use extra conditioning variable to signify conditioned or non-conditioned case
        transforms.Lambdad(keys=["volume"], 
                           func = lambda x: (torch.tensor([x, 1.], dtype=torch.float32) if CLASSIFIER_FREE_GUIDANCE else torch.tensor([x], dtype=torch.float32)).unsqueeze(0)),
    ]


train_transforms = transforms.Compose([
        *base_transforms,
        # use random conditioning value if unconditioned case
        transforms.RandLambdad(keys=["volume"], prob=0.2 if CLASSIFIER_FREE_GUIDANCE else 0, func=lambda x: torch.tensor([random.random(), 0.]).unsqueeze(0)),
        ])

valid_transforms = transforms.Compose(base_transforms)


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
    transform=valid_transforms,
)


train_loader = DataLoader(train_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
valid_loader = DataLoader(validation_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)


dataset_preparation_stopwatch.stop().display()

LOGGER.info(f"Train length: {len(train_ds)} in {len(train_loader)} batches")
LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Mask shape {train_ds[0]["mask"].shape}')

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")


# ### Visualise examples from the training set
train_iter = iter(train_loader)
for i in range(1):
    sample_data = next(train_iter)
    sample_index = 0
    mask = sample_data["mask"][sample_index, 0].detach().cpu().numpy()
    image = sample_data["image"][sample_index, 0].detach().cpu().numpy()

    LOGGER.info(f"Volume.shape: {sample_data['volume'].shape}")

    log_image_with_mask(image, mask, None, None, "trainingdata", sample_data["volume"][sample_index, 0])


    LOGGER.info(f"Batch image shape: {sample_data['mask'].shape}")

#autoencoder = AutoencoderKL(**auto_encoder_config).to(device)
base_autoencoder = AutoencoderKL(**auto_encoder_config).to(device)
autoencoder = EmbeddingWrapper(base_autoencoder=base_autoencoder, vocabulary_size=max(synthseg_classes) + 1, embedding_dim=64)

# Try to load identically trained autoencoder if it already exists. Else train a new one.
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config, patch_discrim_config],
                                            ["auto_encoder_config", "auto_encoder_training_config", "patch_discrim_config"],
                                            project=project, entity=entity,
                                            model=autoencoder, artifact_name=autoencoder.__class__.__name__,
                                            ):
    LOGGER.info("Training new autoencoder...")
    #trainer = MaskAutoencoderTrainer(
    trainer = MaskEmbeddingAutoencoderTrainer(
                                 autoencoder=autoencoder, 
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
    #evaluator = MaskAutoencoderEvaluator(autoencoder=autoencoder, val_loader=valid_loader, wandb_prefix="autoencoder/evaluation")
    evaluator = MaskEmbeddingAutoencoderEvaluator(autoencoder=autoencoder, val_loader=valid_loader, wandb_prefix="autoencoder/evaluation")
    evaluator.visualize_batches(5)
    evaluator.evaluate()
    del evaluator

diffusion_model = DiffusionModelUNet(**diffusion_model_unet_config).to(device)
training_scheduler = DDPMScheduler(num_train_timesteps=1000,
                          schedule="scaled_linear_beta",
                          beta_start=0.0015, 
                          beta_end=0.0205, 
                          clip_sample=False)

evaluation_scheduler = DDIMScheduler(
                          num_train_timesteps=1000,
                          schedule="scaled_linear_beta",
                          beta_start=0.0015, 
                          beta_end=0.0205, 
                          clip_sample=False)
evaluation_scheduler.set_timesteps(num_inference_steps=25)

# We define the inferer using the scale factor:
from src.synthseg_masks import encode_one_hot
#scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data= encode_one_hot(sample_data["mask"].to(device)))
scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=train_iter.__next__()["mask"].to(device))
inferer = LatentDiffusionInferer(training_scheduler, scale_factor=scale_factor)


# update batch-size for diffusion model
train_loader = DataLoader(train_ds, batch_size=diffusion_model_training_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
valid_loader = DataLoader(validation_ds, batch_size=diffusion_model_training_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

encoding_shape = list(encoding_shape)
encoding_shape[0] = diffusion_model_training_config["batch_size"]
encoding_shape = tuple(encoding_shape)

# this should perhaps include also the autoencoder configs
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config, run_config, diffusion_model_unet_config, diffusion_model_training_config],
                                                ["auto_encoder_config", "auto_encoder_training_config", "run_config", "diffusion_model_unet_config", "diffusion_model_training_config"],
                                                project=project, entity=entity,
                                                model=diffusion_model, 
                                                artifact_name=diffusion_model.__class__.__name__,
                                            ):
    LOGGER.info("Training new Diffusion Model...")
    optimizer_diff = torch.optim.Adam(params=diffusion_model.parameters(), 
                                      lr=diffusion_model_training_config["learning_rate"])
    grad_scaler = GradScaler()
    #trainer = MaskDiffusionModelTrainer(
    trainer = MaskEmbeddingDiffusionModelTrainer(
        train_loader=train_loader, valid_loader=valid_loader, 
        autoencoder=autoencoder,
        unet=diffusion_model,
        optimizer_diff=optimizer_diff,
        scaler=grad_scaler,
        inferer=inferer,
        encoding_shape=encoding_shape,
        diffusion_model_training_config=diffusion_model_training_config,
        evaluation_intervall=run_config["evaluation_intervall"],
        evaluation_scheduler=evaluation_scheduler,
        starting_epoch=0,
        guidance=CLASSIFIER_FREE_GUIDANCE
    )
    
    with Stopwatch("Training took: "):
        diffusion_model = trainer.train()

    # clean up fully
    del trainer
    del optimizer_diff
    del grad_scaler

    save_model_as_artifact(wandb_run, diffusion_model, type(diffusion_model).__name__, auto_encoder_config)
else:
    LOGGER.info("Loaded existing diffusion model")

evaluator = MaskDiffusionModelEvaluator(
                 diffusion_model=diffusion_model,
                 autoencoder=autoencoder,
                 latent_shape=encoding_shape,
                 inferer=inferer,
                 val_loader=valid_loader,
                 train_loader=train_loader,
                 wandb_prefix="diffusion/evaluation",
                 evaluation_scheduler=training_scheduler,
                 guidance=CLASSIFIER_FREE_GUIDANCE
                 )

LOGGER.info("Generating final samples")
evaluator.log_samples(5, True)
evaluator.log_samples(5, False)

wandb.finish()