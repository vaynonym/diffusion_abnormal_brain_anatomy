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
from src.evaluation import MaskDiffusionModelEvaluator, SpadeDiffusionModelEvaluator, SpadeAutoencoderEvaluator
from monai.data.meta_tensor import MetaTensor




import torch.multiprocessing
from torch.utils.data import TensorDataset


from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")


#WANDB_MASK_AUTOENCODER_MODEL_RUN_ID = "9cxxbdhj"
#WANDB_MASK_DIFFUSION_MODEL_RUN_ID = "3dikjm6s"
#WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID =  "xrsbiz3d"
#WANDB_SPADE_DIFFUSION_MODEL_RUN_ID = "xrsbiz3d"

# final flair model
WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID =  "lg81hq9f"
WANDB_SPADE_DIFFUSION_MODEL_RUN_ID = "lg81hq9f"
SPADE_Project = "thesis"

# final MRI model
#WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID =  "uj6bvcmy"
#WANDB_SPADE_DIFFUSION_MODEL_RUN_ID = "2cuvdhsn"
#SPADE_Project = "thesis_testruns"

_, entity = load_wand_credentials()

#experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
spade_experiment_config = read_config_from_wandb_run(entity, SPADE_Project, WANDB_SPADE_DIFFUSION_MODEL_RUN_ID)

run_config = spade_experiment_config["run"]
run_config["batch_size"] = 2
run_config["oversample_large_ventricles"] = False
spade_auto_encoder_config = spade_experiment_config["auto_encoder"]
spade_diffusion_model_unet_config = spade_experiment_config["diffusion_model_unet"]

import wandb

wandb_run = wandb.init(project=SPADE_Project, entity=entity, name=f"final_evaluation_{WANDB_SPADE_DIFFUSION_MODEL_RUN_ID}", 
           config=spade_experiment_config)

CLASSIFIER_FREE_GUIDANCE = 4.0
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")

def get_encoding_shape(A_config, R_config):
    down_sampling_factor = (2 ** (len(A_config["num_channels"]) -1) )
    dim_xyz = tuple(map(lambda x: x // down_sampling_factor, R_config["input_image_crop_roi"]))
    encoding_shape = (run_config["batch_size"], A_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
    return encoding_shape

spade_encoding_shape = get_encoding_shape(spade_auto_encoder_config, run_config)
#spade_encoding_shape = (mask_encoding_shape[0], spade_encoding_shape[1], mask_encoding_shape[2], mask_encoding_shape[3], mask_encoding_shape[4])
LOGGER.info(f"Spade encoding shape: {spade_encoding_shape}")


# for reproducibility purposes set a seed
set_determinism(42)

################## Spade Image Generation Model #################

LOGGER.info("Loading Image Generator")

spade_autoencoder = load_model_from_run(run_id=WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID, project=SPADE_Project, entity=entity,
                                  #model_class=AutoencoderKL,
                                  model_class=SPADEAutoencoderKL,  
                                  create_model_from_config=None,
                                  #create_model_from_config=create_embedding_autoencoder
                                  ).eval()

spade_diffusion_model = load_model_from_run(run_id=WANDB_SPADE_DIFFUSION_MODEL_RUN_ID, project=SPADE_Project, entity=entity,
                                      model_class=SPADEDiffusionModelUNet, 
                                      create_model_from_config=None
                                     ).eval()

from src.datasets import get_default_transforms, load_dataset_from_config
_, val_transforms = get_default_transforms(run_config, None)
validation_ds = load_dataset_from_config(run_config, section="validation", transforms=val_transforms)
LOGGER.info("Dataset: " + run_config["dataset"])
LOGGER.info(f"Validation dataset length: {len(validation_ds)}")
LOGGER.info(f"Dataset length: {run_config['dataset_size']}")

sample_meta_tensor: MetaTensor = validation_ds[0]["image"][0]

run_config["batch_size"] = 4
spade_encoding_shape = list(spade_encoding_shape)
spade_encoding_shape[0] = run_config["batch_size"] 
spade_encoding_shape = tuple(spade_encoding_shape)

validation_DL = get_dataloader(validation_ds, run_config)

scale_factor = get_scale_factor(autoencoder=spade_autoencoder, sample_data=next(iter(validation_DL))["image"].to(device), is_mask=False)


spade_scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
eval_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
eval_scheduler.set_timesteps(50)

LOGGER.info(f"Image generation uses scheduler {spade_scheduler.__class__.__name__} with timesteps {spade_scheduler.num_inference_steps}")

spade_inferer =LatentDiffusionInferer (spade_scheduler, scale_factor=scale_factor)

evaluator = SpadeAutoencoderEvaluator(autoencoder=spade_autoencoder, val_loader=validation_DL, wandb_prefix="autoencoder/final_evaluation")
evaluator.visualize_batches(5)
evaluator.evaluate()


spade_evaluator = SpadeDiffusionModelEvaluator(
    diffusion_model=spade_diffusion_model,
    autoencoder=spade_autoencoder,
    latent_shape=spade_encoding_shape,
    inferer=spade_inferer,
    val_loader=validation_DL,
    train_loader=None,
    wandb_prefix="diffusion/final_evaluation",
    evaluation_scheduler=eval_scheduler,
)

spade_evaluator.log_samples(3)
spade_evaluator.evaluate()



