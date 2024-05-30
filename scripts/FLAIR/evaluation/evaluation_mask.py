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
from generative.networks.nets import DiffusionModelUNet, SPADEAutoencoderKL, SPADEDiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from monai.data.meta_tensor import MetaTensor


from src.custom_autoencoders import create_embedding_autoencoder
from src.util import load_wand_credentials, Stopwatch, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, load_model_from_run, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K, get_dataloader

from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import decode_one_hot_to_consecutive_indices, decode_one_hot
from src.evaluation import MaskDiffusionModelEvaluator, SpadeDiffusionModelEvaluator
from monai.data.meta_tensor import MetaTensor




import torch.multiprocessing
from torch.utils.data import TensorDataset


from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")


WANDB_MASK_AUTOENCODER_MODEL_RUN_ID = "9cxxbdhj"
WANDB_MASK_DIFFUSION_MODEL_RUN_ID = "3dikjm6s"
MASK_Project = "thesis_testruns"


_, entity = load_wand_credentials()

#experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
mask_experiment_config = read_config_from_wandb_run(entity, MASK_Project, WANDB_MASK_DIFFUSION_MODEL_RUN_ID)

import wandb

wandb_run = wandb.init(project=MASK_Project, entity=entity, name=f"final_evaluation_{WANDB_MASK_DIFFUSION_MODEL_RUN_ID}", 
           config=mask_experiment_config)

run_config = mask_experiment_config["run"]
run_config["batch_size"] = 4
run_config["oversample_large_ventricles"] = False

LOGGER.info(f"Oversample large ventricles: {run_config['oversample_large_ventricles']}")

mask_run_config = mask_experiment_config["run"]
mask_auto_encoder_config = mask_experiment_config["auto_encoder"]
mask_diffusion_model_unet_config = mask_experiment_config["diffusion_model_unet"]

check_dimensions(run_config, mask_auto_encoder_config, mask_diffusion_model_unet_config)

CLASSIFIER_FREE_GUIDANCE = 4.0
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")

def get_encoding_shape(A_config, R_config):
    down_sampling_factor = (2 ** (len(A_config["num_channels"]) -1) )
    dim_xyz = tuple(map(lambda x: x // down_sampling_factor, R_config["input_image_crop_roi"]))
    encoding_shape = (run_config["batch_size"], A_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
    return encoding_shape

mask_encoding_shape = get_encoding_shape(mask_auto_encoder_config, mask_run_config)
LOGGER.info(f"Mask encoding shape: {mask_encoding_shape}")


# for reproducibility purposes set a seed
set_determinism(36)

LOGGER.info("Loading models...")

LOGGER.info("Loading Mask Generator")

mask_autoencoder = load_model_from_run(run_id=WANDB_MASK_AUTOENCODER_MODEL_RUN_ID, project=MASK_Project, entity=entity,
                                  #model_class=AutoencoderKL,
                                  model_class=EmbeddingWrapper,  
                                  #create_model_from_config=None,
                                  create_model_from_config=create_embedding_autoencoder
                                  ).eval()

mask_diffusion_model = load_model_from_run(run_id=WANDB_MASK_DIFFUSION_MODEL_RUN_ID, project=MASK_Project, entity=entity,
                                      model_class=DiffusionModelUNet, 
                                      create_model_from_config=None
                                     ).eval()

################# Mask Model ################

from src.datasets import load_dataset_from_config, get_default_transforms

_, val_transforms = get_default_transforms(run_config, CLASSIFIER_FREE_GUIDANCE)
validation_ds = load_dataset_from_config(run_config, section="validation", transforms=val_transforms)

val_DL = get_dataloader(validation_ds, run_config)
mask_scale_factor = get_scale_factor(autoencoder=mask_autoencoder, sample_data=next(iter(val_DL))["mask"].to(device), is_mask=True)

mask_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
mask_scheduler.set_timesteps(50)

LOGGER.info(f"Mask model using scheduler {mask_scheduler.__class__.__name__} with timesteps {mask_scheduler.num_inference_steps}")

mask_inferer = LatentDiffusionInferer(mask_scheduler, scale_factor=mask_scale_factor)

from src.evaluation import MaskEmbeddingAutoencoderEvaluator

evaluator = MaskEmbeddingAutoencoderEvaluator(autoencoder=mask_autoencoder, val_loader=val_DL, wandb_prefix="autoencoder/final_evaluation")
evaluator.visualize_batches(5)
evaluator.evaluate()

mask_evaluator = MaskDiffusionModelEvaluator(
                diffusion_model=mask_diffusion_model,
                autoencoder=mask_autoencoder,
                latent_shape=mask_encoding_shape,
                inferer=mask_inferer,
                val_loader=val_DL,
                train_loader=None,
                wandb_prefix="diffusion/final_evaluation",
                evaluation_scheduler=mask_scheduler,
                guidance=CLASSIFIER_FREE_GUIDANCE
                )
mask_evaluator.log_samples(5)
mask_evaluator.evaluate()










