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

# for reproducibility purposes set a seed
set_determinism(42)

import numpy as np
from src.datasets import get_default_transforms, load_dataset_from_config            

train_transforms, val_transforms = get_default_transforms(run_config, guidance=None)


LOGGER.info("Loading dataset...")
dataset_preparation_stopwatch = Stopwatch("Done! Loading the Dataset took: ").start()

train_ds = load_dataset_from_config(run_config, "training", train_transforms)
print("Loaded DS")

from src.dataset_analysis import log_bucket_counts
import numpy as np

np.set_printoptions(suppress=True)

#log_bucket_counts(train_ds)

print("Creating DL")
train_loader = get_dataloader(train_ds, run_config)
print("Created DL")

dataset_preparation_stopwatch.stop().display()

LOGGER.info(f"Train length: {len(train_ds)} in {len(train_loader)} batches")
LOGGER.info(f'Mask shape {train_ds[0]["mask"].shape}')


from src.directory_management import OUTPUT_DIRECTORY

base_path = os.path.join(OUTPUT_DIRECTORY, "Dataset777_RealBrainFlair")
mask_path = os.path.join(base_path, "labelsTr")
os.makedirs(mask_path, exist_ok=True)
image_path = os.path.join(base_path, "imagesTr")
os.makedirs(image_path, exist_ok=True)

CHANNEL_IDENTIFIER = 0
EXTENSION = ".nii.gz"
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



from src.util import visualize_3d_image_slice_wise
from src.synthseg_masks import encode_one_hot, decode_one_hot_to_consecutive_indices

from tqdm import tqdm

from monai.data.meta_tensor import MetaTensor

for case_identifier, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
    image : MetaTensor = batch["image"][0]
    raw_mask : MetaTensor = batch["mask"][0]


    with torch.no_grad():
        one_hot = encode_one_hot(batch["mask"].int().to(device))
        mask = decode_one_hot_to_consecutive_indices(
                    one_hot
                )[0].cpu().detach().int()
        raw_mask[:, :, :, :] = mask

    sub_path = f"{case_identifier}"


    # manually overwrite affine mismatch, which is very small but causes warnings in NNUNet
    raw_mask.affine = image.affine

    assert (raw_mask.affine == image.affine).all()


    mask_tmp_file_name = save_mask(raw_mask, None).meta["saved_to"]
    NN_UNet_mask_filename = f"FLAIR_{case_identifier:03d}{nnunet_mask_postfix}"
    NN_UNet_mask_Pathname = os.path.join(base_path, "labelsTr", NN_UNet_mask_filename)
    os.rename( mask_tmp_file_name, NN_UNet_mask_Pathname)
    
    
    image_tmp_file_name = save_image(image, None).meta["saved_to"]
    NN_UNet_image_filename = f"FLAIR_{case_identifier:03d}" + f"{nnunet_image_postfix}"
    NN_UNet_image_Pathname = os.path.join(base_path, "imagesTr", NN_UNet_image_filename)
    os.rename(image_tmp_file_name, NN_UNet_image_Pathname)

    if case_identifier < 3:
        visualize_3d_image_slice_wise(image[0], OUTPUT_DIRECTORY + f"/{case_identifier:03d}_img.png", description_prefix="", log_to_wandb=False, conditioning_information=batch["volume"][0])
        visualize_3d_image_slice_wise(mask[0], OUTPUT_DIRECTORY + f"/{case_identifier:03d}_mask.png", description_prefix="", log_to_wandb=False, conditioning_information=batch["volume"][0],
                                      is_image_mask=True)
    
    if case_identifier == 999:
        break


    