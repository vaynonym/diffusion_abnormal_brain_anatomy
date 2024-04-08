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

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.util import load_wand_credentials, Stopwatch, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, load_model_from_run, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K, get_dataloader

from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import synthseg_classes, central_areas_close_to_ventricles_indices, ventricle_indices, white_matter_indices, cortex_indices, CSF_indices, background_indices, decode_one_hot
from src.evaluation import MaskDiffusionModelEvaluator



import torch.multiprocessing
from torch.utils.data import TensorDataset


from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")
WANDB_AUTOENCODER_RUNID = os.environ.get("WANDB_AUTOENCODER_RUNID")
WANDB_DIFFUSIONMODEL_RUNID = os.environ.get("WANDB_DIFFUSIONMODEL_RUNID")


project, entity = load_wand_credentials()

#experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
experiment_config = read_config_from_wandb_run(entity, project, WANDB_RUN_NAME)

run_config = experiment_config["run"]
LOGGER.info(f"Oversample large ventricles: {run_config['oversample_large_ventricles']}")

auto_encoder_config = experiment_config["auto_encoder"]
patch_discrim_config = experiment_config["patch_discrim"]
auto_encoder_training_config = experiment_config["autoencoder_training"]
diffusion_model_unet_config = experiment_config["diffusion_model_unet"]
diffusion_model_training_config = experiment_config["diffusion_model_unet_training"]

check_dimensions(run_config, auto_encoder_config, diffusion_model_unet_config)

diffusion_model_training_config["classifier_free_guidance"] = 4.0
CLASSIFIER_FREE_GUIDANCE = diffusion_model_training_config["classifier_free_guidance"] if "classifier_free_guidance" in diffusion_model_training_config else None
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")


# for reproducibility purposes set a seed
set_determinism(42)

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))

LOGGER.info("Loading models...")

def create_embedding_autoencoder(*args, **kwargs):
    base_autoencoder = AutoencoderKL(*args, **kwargs)
    autoencoder = EmbeddingWrapper(base_autoencoder=base_autoencoder, vocabulary_size=max(synthseg_classes) + 1, embedding_dim=64)
    return autoencoder

autoencoder = load_model_from_run(run_id=WANDB_AUTOENCODER_RUNID, project=project, entity=entity,
                                  #model_class=AutoencoderKL,
                                  model_class=EmbeddingWrapper,  
                                  #create_model_from_config=None,
                                  create_model_from_config=create_embedding_autoencoder
                                  )

autoencoder.eval()

diffusion_model = load_model_from_run(run_id=WANDB_DIFFUSIONMODEL_RUNID, project=project, entity=entity,
                                      model_class=DiffusionModelUNet, 
                                      create_model_from_config=None
                                     )

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#scheduler.set_timesteps(50)

LOGGER.info(f"Using scheduler {scheduler.__class__.__name__} with timesteps {scheduler.num_inference_steps}")


# scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=next(iter(valid_loader))["mask"].to(device))
scale_factor = 1.0

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# include abnormal values
bucket_size = 0.1
buckets = torch.tensor(np.arange(0, 1.6, bucket_size)).to(device)
# ensure that every normal value is within 0 to 1 buckets, essentially making [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0], (1.0, 1.1], ...
buckets[0] = -0.001 

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])

from src.evaluation import create_fake_volume_dataloader

volume_dataloader = create_fake_volume_dataloader(0.85, 1.6, 0.1, 1, classifier_free_guidance=CLASSIFIER_FREE_GUIDANCE, batch_size=run_config["batch_size"])

evaluator = MaskDiffusionModelEvaluator(
                diffusion_model=diffusion_model,
                autoencoder=autoencoder,
                latent_shape=encoding_shape,
                inferer=inferer,
                val_loader=None,
                train_loader=None,
                wandb_prefix="",
                evaluation_scheduler=scheduler,
                guidance=CLASSIFIER_FREE_GUIDANCE
                )

from monai import transforms
from src.datasets import AbnormalSyntheticMaskDataset

base_path = os.path.join(OUTPUT_DIRECTORY, "abnormal_masks")
if not os.path.exists(base_path):
    os.makedirs(base_path)

postfix = "mask"
save_image = transforms.SaveImage(os.path.join(base_path, "data"), output_ext=".nii.gz", separate_folder=False, output_postfix=postfix, output_dtype=np.int8, print_log=False,
                                  
                                   )

def save_to_nii(img, meta_dict): return save_image(img, meta_dict)

import csv

with open(os.path.join(base_path, AbnormalSyntheticMaskDataset.OVERVIEW_FILENAME), mode='w', newline='') as overview_file:
    writer = csv.writer(overview_file, delimiter='\t', lineterminator='\n')
    batch_size = encoding_shape[0]

    for i, batch in tqdm(enumerate(volume_dataloader), total=len(volume_dataloader), desc="Synthesis"):
        mask = decode_one_hot(evaluator.get_synthetic_output(batch, False)).to(torch.int8)
        for batch_i in range(batch_size):

            volume = batch["volume"][batch_i, :, 0].detach().cpu().item()
            sub_path = f"{i * batch_size + batch_i}"

            meta_dict = {"input_image_name": sub_path}

            save_image(mask[batch_i], meta_dict)

            writer.writerow([f"{sub_path}_mask", f"{volume:.2f}"])






