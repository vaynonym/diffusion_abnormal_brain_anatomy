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
#WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID =  "xrsbiz3d"
#WANDB_SPADE_DIFFUSION_MODEL_RUN_ID = "xrsbiz3d"
WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID =  "lg81hq9f"
WANDB_SPADE_DIFFUSION_MODEL_RUN_ID = "lg81hq9f"
SPADE_Project = "thesis"
MASK_Project = "thesis_testruns"


_, entity = load_wand_credentials()

#experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
mask_experiment_config = read_config_from_wandb_run(entity, MASK_Project, WANDB_MASK_AUTOENCODER_MODEL_RUN_ID)
spade_experiment_config = read_config_from_wandb_run(entity, SPADE_Project, WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID)

run_config = mask_experiment_config["run"]
run_config["batch_size"] = 2
run_config["oversample_large_ventricles"] = False

LOGGER.info(f"Oversample large ventricles: {run_config['oversample_large_ventricles']}")

mask_run_config = mask_experiment_config["run"]
mask_auto_encoder_config = mask_experiment_config["auto_encoder"]
mask_diffusion_model_unet_config = mask_experiment_config["diffusion_model_unet"]

spade_run_config = spade_experiment_config["run"]
spade_auto_encoder_config = spade_experiment_config["auto_encoder"]
spade_diffusion_model_unet_config = spade_experiment_config["diffusion_model_unet"]

check_dimensions(run_config, mask_auto_encoder_config, mask_diffusion_model_unet_config)

CLASSIFIER_FREE_GUIDANCE = 1.0
LOGGER.info(f"Using classifier free guidance: {CLASSIFIER_FREE_GUIDANCE}")

def get_encoding_shape(A_config, R_config):
    down_sampling_factor = (2 ** (len(A_config["num_channels"]) -1) )
    dim_xyz = tuple(map(lambda x: x // down_sampling_factor, R_config["input_image_crop_roi"]))
    encoding_shape = (run_config["batch_size"], A_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
    return encoding_shape

mask_encoding_shape = get_encoding_shape(mask_auto_encoder_config, mask_run_config)
LOGGER.info(f"Mask encoding shape: {mask_encoding_shape}")

spade_encoding_shape = get_encoding_shape(spade_auto_encoder_config, spade_run_config)
#spade_encoding_shape = (mask_encoding_shape[0], spade_encoding_shape[1], mask_encoding_shape[2], mask_encoding_shape[3], mask_encoding_shape[4])
LOGGER.info(f"Spade encoding shape: {spade_encoding_shape}")


# for reproducibility purposes set a seed
set_determinism(40)

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

dummy_LDM100k_dataset_loading_config = {
    **run_config,
    "dataset": "LDM100K", # overwrite to use LDM100k for to get mask model
    "dataset_size": 8,
    }

from src.datasets import load_dataset_from_config, get_default_transforms

_, val_transforms = get_default_transforms(dummy_LDM100k_dataset_loading_config, CLASSIFIER_FREE_GUIDANCE)
validation_ds = load_dataset_from_config(dummy_LDM100k_dataset_loading_config, section="training", transforms=val_transforms)

scale_factor_DL = get_dataloader(validation_ds, run_config)
mask_scale_factor = get_scale_factor(autoencoder=mask_autoencoder, sample_data=next(iter(scale_factor_DL))["mask"].to(device), is_mask=True)

del scale_factor_DL
del validation_ds

mask_scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#mask_scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#mask_scheduler.set_timesteps(50)

LOGGER.info(f"Mask model using scheduler {mask_scheduler.__class__.__name__} with timesteps {mask_scheduler.num_inference_steps}")

mask_inferer = LatentDiffusionInferer(mask_scheduler, scale_factor=mask_scale_factor)

mask_evaluator = MaskDiffusionModelEvaluator(
                diffusion_model=mask_diffusion_model,
                autoencoder=mask_autoencoder,
                latent_shape=mask_encoding_shape,
                inferer=mask_inferer,
                val_loader=None,
                train_loader=None,
                wandb_prefix="",
                evaluation_scheduler=mask_scheduler,
                guidance=CLASSIFIER_FREE_GUIDANCE
                )

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

dummy_FLAIR_dataset_loading_config = {
    **run_config,
    "dataset": "RH_FLAIR", # overwrite to use LDM100k for to get mask model
    "dataset_size": 8,
    }

_, val_transforms = get_default_transforms(dummy_LDM100k_dataset_loading_config, None)
validation_ds = load_dataset_from_config(dummy_LDM100k_dataset_loading_config, section="training", transforms=val_transforms)

sample_meta_tensor: MetaTensor = validation_ds[0]["image"][0]
default_dict = sample_meta_tensor.meta
default_transforms = sample_meta_tensor.applied_operations
default_affine = sample_meta_tensor.affine


scale_factor_DL = get_dataloader(validation_ds, run_config)
scale_factor = get_scale_factor(autoencoder=spade_autoencoder, sample_data=next(iter(scale_factor_DL))["image"].to(device), is_mask=False)

del sample_meta_tensor
del scale_factor_DL
del validation_ds

spade_scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
#scheduler.set_timesteps(50)

LOGGER.info(f"Image generation uses scheduler {spade_scheduler.__class__.__name__} with timesteps {spade_scheduler.num_inference_steps}")

spade_inferer = LatentDiffusionInferer(spade_scheduler, scale_factor=scale_factor)

spade_evaluator = SpadeDiffusionModelEvaluator(
    diffusion_model=spade_diffusion_model,
    autoencoder=spade_autoencoder,
    latent_shape=spade_encoding_shape,
    inferer=spade_inferer,
    val_loader=None,
    train_loader=None,
    wandb_prefix="",
    evaluation_scheduler=spade_scheduler,
)

from src.evaluation import create_fake_volume_dataloader

volume_dataloader = create_fake_volume_dataloader(0.05, 1.0, 0.1, 20, 
                                                  classifier_free_guidance=CLASSIFIER_FREE_GUIDANCE,
                                                  batch_size=run_config["batch_size"])



from monai import transforms
base_path = os.path.join(OUTPUT_DIRECTORY, "Dataset779_1_AugmentedBrainFlair")
if not os.path.exists(base_path):
    os.makedirs(base_path)

EXTENSION = ".nii.gz"
CHANNEL_IDENTIFIER = 0
nnunet_image_postfix = f"_{CHANNEL_IDENTIFIER:04d}{EXTENSION}"
nnunet_mask_postfix = f"{EXTENSION}"

image_tmp_save_path = os.path.join(base_path, "tmp_images")
os.makedirs(image_tmp_save_path, exist_ok=True)
save_image = transforms.SaveImage(image_tmp_save_path, output_ext=EXTENSION, 
                                  separate_folder=False, output_postfix="image", 
                                  output_dtype=np.float32, print_log=False, savepath_in_metadict=True)

mask_tmp_save_path = os.path.join(base_path, "tmp_masks")
os.makedirs(mask_tmp_save_path, exist_ok=True)
save_mask = transforms.SaveImage(mask_tmp_save_path, output_ext=EXTENSION, 
                                  separate_folder=False, output_postfix="mask", 
                                  output_dtype=np.int8, print_log=False, savepath_in_metadict=True)

import csv
import wandb

os.makedirs(os.path.join(base_path, "labelsTr"), exist_ok=True)
os.makedirs(os.path.join(base_path, "imagesTr"), exist_ok=True)

from monai.utils.enums import MetaKeys, SpaceKeys
from src.synthseg_masks import encode_contiguous_labels_one_hot, encode_one_hot
from torch import Tensor
from src.util import visualize_3d_image_slice_wise

# TODO: figure out MetaDict... may need to look at example MetaDict from saved image

from monai import transforms

CASE_IDENTIFIER_OFFSET = 1412

for batch_num, vol_batch in tqdm(enumerate(volume_dataloader), total=len(volume_dataloader)):

    random_factor = torch.rand_like(vol_batch["volume"][:, :, 0:1]) / 10 - 0.05
    vol_batch["volume"][:, :, 0:1] = vol_batch["volume"][:, :, 0:1] + random_factor

    #print(vol_batch["volume"])

    mask_batch = decode_one_hot(mask_evaluator.get_synthetic_output(vol_batch, False))
    # flip to align axis from IPL to IPR
    for batch_i in range(run_config["batch_size"]):
        mask_batch[batch_i] = transforms.Flip(spatial_axis=1)(mask_batch[batch_i])

    spade_input = {
        "mask": mask_batch,
    }
 
    image_batch = torch.clip(spade_evaluator.get_synthetic_output(spade_input, False), min=0., max=1.)

    with torch.no_grad():
        one_hot = encode_one_hot(mask_batch.int().to(device))
        nnunet_encoded_mask = decode_one_hot_to_consecutive_indices(one_hot).cpu().detach().int()


    for batch_i in range(run_config["batch_size"]):

        case_identifier = batch_num * run_config["batch_size"] + batch_i
        sub_path = f"{case_identifier}"


        image = MetaTensor(image_batch[batch_i].cpu().detach(), default_affine, default_dict, default_transforms)
        mask = MetaTensor(nnunet_encoded_mask[batch_i], default_affine, default_dict, default_transforms)


        # manually overwrite affine mismatch, which is very small but causes warnings in NNUNet
        #raw_mask.affine = image.affine

        assert (mask.affine == image.affine).all()


        mask_tmp_file_name = save_mask(mask, None).meta["saved_to"]
        NN_UNet_mask_filename = f"FLAIR_{CASE_IDENTIFIER_OFFSET + case_identifier:03d}{nnunet_mask_postfix}"
        NN_UNet_mask_Pathname = os.path.join(base_path, "labelsTr", NN_UNet_mask_filename)
        os.rename( mask_tmp_file_name, NN_UNet_mask_Pathname)
        
        
        image_tmp_file_name = save_image(image, None).meta["saved_to"]
        NN_UNet_image_filename = f"FLAIR_{CASE_IDENTIFIER_OFFSET + case_identifier:03d}" + f"{nnunet_image_postfix}"
        NN_UNet_image_Pathname = os.path.join(base_path, "imagesTr", NN_UNet_image_filename)
        os.rename(image_tmp_file_name, NN_UNet_image_Pathname)

        if case_identifier < 3:
            visualize_3d_image_slice_wise(image[0].detach().cpu(), OUTPUT_DIRECTORY + f"/{CASE_IDENTIFIER_OFFSET + case_identifier:03d}_img.png", description_prefix="",
                                           log_to_wandb=False, conditioning_information=vol_batch["volume"][batch_i][0][0])
            visualize_3d_image_slice_wise(mask[0].detach().cpu().int(), OUTPUT_DIRECTORY + f"/{CASE_IDENTIFIER_OFFSET + case_identifier:03d}_mask.png", description_prefix="",
                                           log_to_wandb=False, conditioning_information=vol_batch["volume"][batch_i][0][0],
                                        is_image_mask=True)
        
        #if case_identifier >= :
        #    break






