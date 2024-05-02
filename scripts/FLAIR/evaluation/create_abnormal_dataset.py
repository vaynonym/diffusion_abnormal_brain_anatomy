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
from generative.networks.nets import DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


from src.custom_autoencoders import create_embedding_autoencoder
from src.util import load_wand_credentials, Stopwatch, read_config, read_config_from_wandb_run
from src.model_util import load_model_from_run_with_matching_config, load_model_from_run, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K, get_dataloader

from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper
from src.synthseg_masks import decode_one_hot_to_consecutive_indices, decode_one_hot
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

diffusion_model_training_config["classifier_free_guidance"] = 3.0
CLASSIFIER_FREE_GUIDANCE = diffusion_model_training_config["classifier_free_guidance"] if "classifier_free_guidance" in diffusion_model_training_config else None
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")


# for reproducibility purposes set a seed
set_determinism(42)

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))

LOGGER.info("Loading models...")

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

run_config["oversample_large_ventricles"]=False
dummy_dataset_loading_config = {
    **run_config,
    "dataset": "LDM100K", # overwrite to use LDM100k for to get mask model
    "dataset_size": 8,
    }

from src.datasets import load_dataset_from_config, get_default_transforms

_, val_transforms = get_default_transforms(dummy_dataset_loading_config, CLASSIFIER_FREE_GUIDANCE)
validation_ds = load_dataset_from_config(dummy_dataset_loading_config, section="training", transforms=val_transforms)

scale_factor_DL = get_dataloader(validation_ds, run_config)
scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=next(iter(scale_factor_DL))["mask"].to(device), is_mask=True)

del scale_factor_DL
del validation_ds

scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#scheduler.set_timesteps(50)

LOGGER.info(f"Using scheduler {scheduler.__class__.__name__} with timesteps {scheduler.num_inference_steps}")

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)


down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])

from src.evaluation import create_fake_volume_dataloader

volume_dataloader = create_fake_volume_dataloader(0.85, 1.3, 0.1, 1, 
                                                  classifier_free_guidance=CLASSIFIER_FREE_GUIDANCE,
                                                  batch_size=run_config["batch_size"])

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
extension = ".nii.gz"
save_image = transforms.SaveImage(os.path.join(base_path, "data"), output_ext=".nii.gz", 
                                  separate_folder=False, output_postfix=postfix, 
                                  output_dtype=np.int8, print_log=False,
                                  savepath_in_metadict=True)

import csv
import wandb

#project, entity = load_wand_credentials()
#wandb_run = wandb.init(project=project, entity=entity, name=WANDB_RUN_NAME, 
#           config={"run_config": run_config,
#                   "auto_encoder_config": auto_encoder_config, 
#                   "patch_discrim_config": patch_discrim_config,
#                   "auto_encoder_training_config": auto_encoder_training_config,
#                   "diffusion_model_unet_config": diffusion_model_unet_config,
#                   "diffusion_model_training_config": diffusion_model_training_config,
#                   })

os.makedirs(os.path.join(base_path, "synthetic", "nnUNet_raw", "Dataset001_BrainFlair", "labelsTr"), exist_ok=True)

with open(os.path.join(base_path, AbnormalSyntheticMaskDataset.OVERVIEW_FILENAME), mode='w', newline='') as overview_file:
    writer = csv.writer(overview_file, delimiter='\t', lineterminator='\n')
    batch_size = encoding_shape[0]

    for i, batch in tqdm(enumerate(volume_dataloader), total=len(volume_dataloader), desc="Synthesis"):
        mask = decode_one_hot_to_consecutive_indices(evaluator.get_synthetic_output(batch, False)).to(torch.int8)
        #image = torch.zeros_like(mask).detach().cpu().numpy()
        #to_log = mask.detach().cpu().numpy()

        for batch_i in range(batch_size):
            volume = batch["volume"][batch_i, :, 0].detach().cpu().item()

            #log_image_with_mask(image[batch_i, 0], to_log[batch_i, 0], image[batch_i, 0], to_log[batch_i, 0], "diffusion/evaluation", batch["volume"][batch_i, :, 0].detach().cpu())


            sub_path = f"{i * batch_size + batch_i:03d}"

            meta_dict = {"input_image_name": sub_path}
            from monai.data import MetaTensor
            to_save : MetaTensor = mask[batch_i]

            to_save.meta["input_image_name"] = sub_path

            r = save_image(mask[batch_i], meta_dict)

            saved_path = r.meta["saved_to"]
            
            file_name = f"{sub_path}_{postfix}{extension}"
            
            NN_UNet_filename = f"FLAIR_{i * batch_size + batch_i:04d}{extension}"
            NN_UNetPathname = os.path.join(base_path, "synthetic", "nnUNet_raw", "Dataset001_BrainFlair", "labelsTr", NN_UNet_filename)
            r.meta["saved_to"] = NN_UNetPathname
            os.rename(saved_path,
                      NN_UNetPathname
                      )

            writer.writerow([NN_UNetPathname, f"{volume:.2f}"])






