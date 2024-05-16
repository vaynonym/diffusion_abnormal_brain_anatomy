#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing
import wandb
from src.torch_setup import device

from monai import transforms
from monai.utils import set_determinism


from src.util import Stopwatch, read_config
from src.logging_util import LOGGER
from src.datasets import get_dataloader


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

from src.datasets import get_crop_around_mask_center

target_spacing = run_config["target_spacing"]
crop_size = run_config["input_image_crop_roi"]

base_transforms = transforms.Compose([
        transforms.LoadImaged(keys=["image", "mask"]),
        transforms.EnsureChannelFirstd(keys=["image", "mask"]),
        transforms.EnsureTyped(keys=["image", "mask"]),
        transforms.Lambdad(keys=["image"], func=lambda x: x[0, :, :, :].unsqueeze(0)), # select first channel if multiple channels occur

        transforms.Orientationd(keys=["image", "mask"], axcodes="IPR"), # IAR # axcodes="RAS"
        transforms.Spacingd(keys=["mask"], pixdim=target_spacing, mode=("nearest")),
        transforms.Spacingd(keys=["image"], pixdim=target_spacing, mode=("bilinear")),
        transforms.Lambda(func=get_crop_around_mask_center(crop_size)),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
    ])


LOGGER.info("Loading dataset...")
dataset_preparation_stopwatch = Stopwatch("Done! Loading the Dataset took: ").start()

from src.datasets import RHFlairTestDataset

ds = RHFlairTestDataset(dataset_path="/depict/users/tim/private/final_testset", transform=base_transforms, size=50, cache_rate=1.0, num_workers=4)
print(f"Loaded DS {ds.__class__.__name__}")

import numpy as np

np.set_printoptions(suppress=True)

#log_bucket_counts(train_ds)

print("Creating DL")
train_loader = get_dataloader(ds, run_config)
print("Created DL")

dataset_preparation_stopwatch.stop().display()

LOGGER.info(f"Train length: {len(ds)} in {len(train_loader)} batches")
LOGGER.info(f'Mask shape {ds[0]["mask"].shape}')


from src.directory_management import OUTPUT_DIRECTORY

base_path = os.path.join(OUTPUT_DIRECTORY, "Task809_TestRealBrainFlair")
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
from pprint import pprint


for case_identifier, batch in tqdm(enumerate(train_loader), total=len(train_loader)):
    image : MetaTensor = batch["image"][0]
    raw_mask : MetaTensor = batch["mask"][0]

    LOGGER.info(raw_mask.unique())

    mask = raw_mask.detach().int()


    #with torch.no_grad():
    #    one_hot = encode_one_hot(batch["mask"].int().to(device))
    #    mask = decode_one_hot_to_consecutive_indices(
    #                one_hot
    #            )[0].cpu().detach().int()
    #    raw_mask[:, :, :, :] = mask

    sub_path = f"{case_identifier}"

    sub_path = batch["case_identifier"][0]


    # manually overwrite affine mismatch, which is very small but causes warnings in NNUNet
    raw_mask.affine = image.affine

    assert (raw_mask.affine == image.affine).all()


    mask_tmp_file_name = save_mask(raw_mask, None).meta["saved_to"]
    NN_UNet_mask_filename = f"FLAIR-{sub_path}{nnunet_mask_postfix}"
    NN_UNet_mask_Pathname = os.path.join(base_path, "labelsTr", NN_UNet_mask_filename)
    os.rename( mask_tmp_file_name, NN_UNet_mask_Pathname)
    
    
    image_tmp_file_name = save_image(image, None).meta["saved_to"]
    NN_UNet_image_filename = f"FLAIR-{sub_path}" + f"{nnunet_image_postfix}"
    NN_UNet_image_Pathname = os.path.join(base_path, "imagesTr", NN_UNet_image_filename)
    os.rename(image_tmp_file_name, NN_UNet_image_Pathname)


    visualize_3d_image_slice_wise(image[0], base_path + f"/{sub_path}_img.png", description_prefix="", log_to_wandb=False, conditioning_information=None)
    visualize_3d_image_slice_wise(mask[0], base_path + f"/{sub_path}_mask.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
                                    is_image_mask=True)
    


    