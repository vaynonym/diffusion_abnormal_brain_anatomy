#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing
import wandb
from src.torch_setup import device
from torch.cuda.amp import GradScaler

from monai import transforms
from monai.data import DataLoader
from monai.utils import first, set_determinism
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import SPADEAutoencoderKL, SPADEDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler


from src.util import load_wand_credentials, Stopwatch, read_config, log_image_with_mask
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, load_model_from_run, check_dimensions
from src.logging_util import LOGGER
from src.datasets import get_dataloader
from src.diffusion import get_scale_factor
from src.trainer import SpadeAutoencoderTrainer, SpadeDiffusionModelTrainer
from src.evaluation import SpadeAutoencoderEvaluator, SpadeDiffusionModelEvaluator
from src.transforms import get_crop_around_mask_center


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

import numpy as np
from src.datasets import get_default_transforms, load_dataset_from_config            

CLASSIFIER_FREE_GUIDANCE = diffusion_model_training_config["classifier_free_guidance"]
assert not CLASSIFIER_FREE_GUIDANCE, "Spade model should have guidance turned off"

train_transforms, val_transforms = get_default_transforms(run_config, guidance=CLASSIFIER_FREE_GUIDANCE)


LOGGER.info("Loading dataset...")
dataset_preparation_stopwatch = Stopwatch("Done! Loading the Dataset took: ").start()

dataset_size = run_config["dataset_size"]

train_ds = load_dataset_from_config(run_config, "training", train_transforms)
validation_ds = load_dataset_from_config(run_config, "validation", train_transforms)

from src.dataset_analysis import log_bucket_counts
import numpy as np

np.set_printoptions(suppress=True)

log_bucket_counts(train_ds)


train_loader = get_dataloader(train_ds, run_config)
valid_loader = get_dataloader(validation_ds, run_config)


dataset_preparation_stopwatch.stop().display()

LOGGER.info(f"Train length: {len(train_ds)} in {len(train_loader)} batches")
LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Mask shape {train_ds[0]["mask"].shape}')

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")

# ### Visualise examples from the training set
for i, batch in enumerate(train_loader):
    sample_index = 0
    mask = batch["mask"][sample_index, 0].detach().cpu().numpy()
    image = batch["image"][sample_index, 0].detach().cpu().numpy()

    log_image_with_mask(image, mask, None, None, "trainingdata", batch["volume"][sample_index, 0])
    if i == 0:
        LOGGER.info(f"Batch image shape: {batch['mask'].shape}")
    
    if i == 10:
        break


WANDB_AUTOENCODER_RUNID = os.environ.get("WANDB_AUTOENCODER_RUNID")

autoencoder = SPADEAutoencoderKL(**auto_encoder_config).to(device)


if "pretrained" in auto_encoder_training_config and auto_encoder_training_config["pretrained"]:
    LOGGER.info("Attempting to load pretrained autoencoder model")
    (a_run_id, a_entity, a_project) = auto_encoder_training_config["pretrained"]
    autoencoder = load_model_from_run(run_id=a_run_id, project=a_project, entity=a_entity,
                                  model_class=SPADEAutoencoderKL,
                                  #model_class=EmbeddingWrapper,  
                                  create_model_from_config=None,
                                  #create_model_from_config=create_embedding_autoencoder
                                  )
    
    LOGGER.info("Finetuning pretrained autoencoder...")
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

wandb.finish()
quit()

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
scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=iter(train_loader).__next__()["image"].to(device), is_mask=False)
inferer = LatentDiffusionInferer(training_scheduler, scale_factor=scale_factor)


diffusion_model = SPADEDiffusionModelUNet(**diffusion_model_unet_config).to(device)

# update batch-size for diffusion model
train_loader = DataLoader(train_ds, batch_size=diffusion_model_training_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
valid_loader = DataLoader(validation_ds, batch_size=diffusion_model_training_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)
encoding_shape = list(encoding_shape)
encoding_shape[0] = diffusion_model_training_config["batch_size"]
encoding_shape = tuple(encoding_shape)
LOGGER.info(f"Encoding shape with updated batch-size for diffusion training: {encoding_shape}")

if "pretrained" in diffusion_model_training_config and diffusion_model_training_config["pretrained"]:
    LOGGER.info("Attempting to use pretrained weights...")
    (d_run_id, d_entity, d_project) = diffusion_model_training_config["pretrained"]
    diffusion_model = load_model_from_run(run_id=d_run_id, project=d_project, entity=d_entity,
                                  model_class=SPADEDiffusionModelUNet,
                                  #model_class=EmbeddingWrapper,  
                                  create_model_from_config=None,
                                  #create_model_from_config=create_embedding_autoencoder
                                  )
    
    LOGGER.info("Finetuning pretrained diffusion Model...")
    optimizer_diff = torch.optim.Adam(params=diffusion_model.parameters(), 
                                    lr=diffusion_model_training_config["learning_rate"])
    grad_scaler = GradScaler()
    trainer = SpadeDiffusionModelTrainer(
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
    )
    
    with Stopwatch("Training took: "):
        diffusion_model = trainer.train()

    # clean up fully
    del trainer
    del optimizer_diff
    del grad_scaler

    save_model_as_artifact(wandb_run, diffusion_model, type(diffusion_model).__name__, diffusion_model_unet_config)
else:
    if not load_model_from_run_with_matching_config([run_config, auto_encoder_config, auto_encoder_training_config, diffusion_model_unet_config, diffusion_model_training_config],
                                                    ["run_config", "auto_encoder_config", "auto_encoder_training_config", "diffusion_model_unet_config", "diffusion_model_training_config"],
                                                    project=project, entity=entity,
                                                    model=diffusion_model, 
                                                    artifact_name=diffusion_model.__class__.__name__,
                                                ):
        LOGGER.info("Training new Diffusion Model...")
        optimizer_diff = torch.optim.Adam(params=diffusion_model.parameters(), 
                                        lr=diffusion_model_training_config["learning_rate"])
        grad_scaler = GradScaler()
        trainer = SpadeDiffusionModelTrainer(
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
        )
        
        with Stopwatch("Training took: "):
            diffusion_model = trainer.train()

        # clean up fully
        del trainer
        del optimizer_diff
        del grad_scaler

        save_model_as_artifact(wandb_run, diffusion_model, type(diffusion_model).__name__, diffusion_model_unet_config)
    else:
        LOGGER.info("Loaded existing diffusion model")

evaluator = SpadeDiffusionModelEvaluator(
                 diffusion_model=diffusion_model,
                 autoencoder=autoencoder,
                 latent_shape=encoding_shape,
                 inferer=inferer,
                 val_loader=valid_loader,
                 train_loader=train_loader,
                 wandb_prefix="diffusion/evaluation",
                 evaluation_scheduler=evaluation_scheduler,
                 )

evaluator.log_samples(5, True)
evaluator.log_samples(5, False)

wandb.finish()

LOGGER.info("Finished Run!")