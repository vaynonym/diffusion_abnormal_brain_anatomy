#!/usr/bin/env python3

# adapted from monai-generative tutorial (https://github.com/Project-MONAI/GenerativeModels)

# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     formats: ipynb,py
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.1
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# +
# Copyright (c) MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -

# # 3D Latent Diffusion Model
# In this tutorial, we will walk through the process of using the MONAI Generative Models package to generate synthetic data using Latent Diffusion Models (LDM)  [1, 2]. Specifically, we will focus on training an LDM to create synthetic brain images from the Brats dataset.
#
# [1] - Rombach et al. "High-Resolution Image Synthesis with Latent Diffusion Models" https://arxiv.org/abs/2112.10752
#
# [2] - Pinaya et al. "Brain imaging generation with latent diffusion models" https://arxiv.org/abs/2209.07162

# ### Set up imports

# +
import os
import shutil
import tempfile

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from monai import transforms
from monai.apps import DecathlonDataset
from monai.config import print_config
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from monai.data.dataset import CacheDataset

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

import wandb
import numpy as np
from src.util import load_wand_credentials, visualize_3d_image_slice_wise, Stopwatch, device
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config
from src.training import train_autoencoder
from src.logging_util import LOGGER
import random


# Specify a MONAI_DATA_DIRECTORY variable, where the data will be downloaded. If not specified a temporary directory will be used.
data_directory = os.environ.get("DATA_DIRECTORY")
base_directory = os.environ.get("BASE_DIRECTORY")
output_directory = os.environ.get("OUTPUT_DIRECTORY")
model_directory = os.environ.get("MODEL_DIRECTORY")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")

run_config={"batch_size": 2,
            "input_image_downsampling_factors": (2.4, 2.4, 2.2),
            "input_image_crop_roi": (96, 96, 64),
            "gen_image_intervall": 0.2,
            }

auto_encoder_config= {
    "spatial_dims":3,
    "in_channels":1,
    "out_channels":1,
    "latent_channels":3,
    "num_channels":(32, 64, 64),
    "num_res_blocks":1,
    "norm_num_groups":16,
    "attention_levels":(False, False, True),
}

patch_discrim_config= {
    "spatial_dims":3, 
    "num_layers_d":3, 
    "num_channels":32, 
    "in_channels":1, 
    "out_channels":1
}

auto_encoder_training_config= {
    "n_epochs": 100,
    "autoencoder_warm_up_n_epochs": 5,
}

diffusion_model_unet_config = {
    "spatial_dims":3,
    "in_channels":3,
    "out_channels":3,
    "num_res_blocks":1,
    "num_channels":(32, 64, 64),
    "attention_levels":(False, True, True),
    "num_head_channels":(0, 64, 64),
    "with_conditioning":True,
    "cross_attention_dim":1,
}

diffusion_model_training_config = {
    "n_epochs" : 150,
}

# Check that the image dimensions match downsampling dimensions
for image_dim in run_config["input_image_crop_roi"]:
    down_sample_factor_autoencoder = len(auto_encoder_config["num_channels"]) - 1
    down_sample_factor_diffusion_model = len(diffusion_model_unet_config["num_channels"]) - 1
    assert (image_dim % (down_sample_factor_autoencoder * down_sample_factor_diffusion_model)) == 0,\
           f"image dim {image_dim} must be evenly divisible by autoencoder downsampling factor {down_sample_factor_autoencoder} * diffusion dowmsampling factor {down_sample_factor_diffusion_model}"

LOGGER.info("Image dimensions match downsampling factors! Good to go!")

project, entity = load_wand_credentials()
wandb_run = wandb.init(project=project, entity=entity, name=WANDB_RUN_NAME, 
           config={"run_config": run_config,
                   "auto_encoder_config": auto_encoder_config, 
                   "patch_discrim_config": patch_discrim_config,
                   "auto_encoder_training_config": auto_encoder_training_config,
                   "diffusion_model_unet_config": diffusion_model_unet_config,
                   "diffusion_model_training_config": diffusion_model_training_config,
                   })

autoencoder = AutoencoderKL(
    **auto_encoder_config
)

# for reproducibility purposes set a seed
set_determinism(42)

# ### Setup a data directory and download dataset

# ### Prepare data loader for the training set
# Here we will download the Brats dataset using MONAI's `DecathlonDataset` class, and we prepare the data loader for the training set.


#	 "0": "FLAIR", 
#	 "1": "T1w", 
#	 "2": "t1gd",
#	 "3": "T2w"
channels = [0, 1]
for channel in channels:
    assert channel in [0, 1, 2, 3], "Choose a valid channel"
LOGGER.info(f"Using channels {channels}")

def select_random_channel_and_set_class(input_dict):
    selected_channel = random.choice(channels)
    input_dict["image"] = input_dict["image"][selected_channel, :, :, :]
    input_dict["class"] = selected_channel
    return input_dict

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambda(select_random_channel_and_set_class),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="RAS"),
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys="image", lower=0, upper=99.5, b_min=0, b_max=1),
        transforms.RandLambdad(keys=["class"], prob=0.15, func=lambda x: -1 * torch.ones_like(x)),
        transforms.Lambdad(
            keys=["class"], func=lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        ),
    ]
)

LOGGER.info("Loading dataset...")
dataset_preparation_stopwatch = Stopwatch("Done! Loading the Dataset took: ").start()

train_ds = DecathlonDataset(
    root_dir=data_directory,
    task="Task01_BrainTumour",
    section="training",  # validation
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    download=not os.path.exists(os.path.join(data_directory, "Task01_BrainTumour")),  # Set download to True if the dataset hasnt been downloaded yet
    seed=0,
    progress=False,
    transform=train_transforms,
)

train_loader = DataLoader(train_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=4, persistent_workers=True)
print("train_loader length:", len(train_loader))

dataset_preparation_stopwatch.stop().display()
LOGGER.info(f'Image shape {train_ds[0]["image"].shape}')

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (1, auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")
# -

# ### Visualise examples from the training set

# +
# Plot axial, coronal and sagittal slices of a training sample
iter_loader = iter(train_loader)
for i in range(10):
    check_data = next(iter_loader)
    sample_index = 0
    img = check_data["image"][sample_index, 0].detach().cpu().numpy()
    visualize_3d_image_slice_wise(img, None, "trainingdata sample/c=" + str(check_data["class"][sample_index, 0].item()), WANDB_LOG_IMAGES)

# -

# ## Autoencoder KL
#
# ### Define Autoencoder KL network
#
# In this section, we will define an autoencoder with KL-regularization for the LDM. The autoencoder's primary purpose is to transform input images into a latent representation that the diffusion model will subsequently learn. By doing so, we can decrease the computational resources required to train the diffusion component, making this approach suitable for learning high-resolution medical images.
#


autoencoder = AutoencoderKL(
    **auto_encoder_config
)
autoencoder.to(device)

# Trz to load identically trained autoencoder if it already exists. Else train a new one.
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config, run_config],
                                            ["auto_encoder_config", "auto_encoder_training_config", "run_config"],
                                            project=project, entity=entity, 
                                            model=autoencoder, artifact_name="autoencoderKL",
                                            models_path=model_directory,
                                            ):
    LOGGER.info("Training new autoencoder...")
    autoencoder = train_autoencoder(autoencoder, train_loader, 
                                                    patch_discrim_config, auto_encoder_training_config, run_config,
                                                    WANDB_LOG_IMAGES)

    save_model_as_artifact(wandb_run, autoencoder, type(autoencoder).__name__, auto_encoder_config, model_directory)
else:
    LOGGER.info("Loaded existing autoencoder")

for i in range(10):
    _, batch = next(enumerate(train_loader))
    images = batch["image"].to(device)
    autoencoder.eval()
    reconstruction, _, _ = autoencoder(images) 
    img = reconstruction[sample_index, 0].detach().cpu().numpy()
    visualize_3d_image_slice_wise(img, os.path.join(output_directory, "Visualize_Reconstruction.png"), "Visualize Reconstruction", WANDB_LOG_IMAGES)


# ### Visualise reconstructions



# ## Diffusion Model
#
# ### Define diffusion model and scheduler
#
# In this section, we will define the diffusion model that will learn data distribution of the latent representation of the autoencoder. Together with the diffusion model, we define a beta scheduler responsible for defining the amount of noise tahat is added across the diffusion's model Markov chain.

# +
unet = DiffusionModelUNet(
    **diffusion_model_unet_config
)
unet.to(device)


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)
# -

# ### Scaling factor
#
# As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
#
# _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
#

# +
with torch.no_grad():
    with autocast(enabled=True):
        z = autoencoder.encode_stage_2_inputs(check_data["image"].to(device))

LOGGER.info(f"Scaling factor set to {1/torch.std(z)}")
scale_factor = 1 / torch.std(z)
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

optimizer_diff = torch.optim.Adam(params=unet.parameters(), lr=1e-4)

# ### Train diffusion model

# +
n_epochs = diffusion_model_training_config["n_epochs"]
epoch_loss_list = []
autoencoder.eval()
scaler = GradScaler()

first_batch = first(train_loader)
z = autoencoder.encode_stage_2_inputs(first_batch["image"].to(device))

wandb.define_metric("diffusion/epoch")
wandb.define_metric("diffusion/*", step_metric="diffusion/epoch")



def eval_generate_sample_images(inferer: LatentDiffusionInferer, autoencoder, unet, scheduler, path, prefix_string):
    autoencoder.eval()
    unet.eval()
    noise = torch.randn( encoding_shape)
    noise = noise.to(device)
    c = torch.ones(1, 1, 1).float() * random.choice(channels + [-1])
    c = c.to(device)

    scheduler.set_timesteps(num_inference_steps=1000)
    synthetic_images = inferer.sample(
        input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler, conditioning=c
    )

    # ### Visualise synthetic data

    img = synthetic_images[sample_index, 0].detach().cpu().numpy()  # images
    visualize_3d_image_slice_wise(img, path, prefix_string + f"/c={c.item()}", WANDB_LOG_IMAGES)

diffusion_stopwatch = Stopwatch("Diffusion training took:").start()
for epoch in range(n_epochs):
    unet.train()
    epoch_loss = 0

    for step, batch in enumerate(train_loader):
        images = batch["image"].to(device)
        classes = batch["class"].to(device)
        optimizer_diff.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn_like(z).to(device)

            # Create timesteps
            timesteps = torch.randint(
                0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
            ).long()

            # Get model prediction
            noise_pred = inferer(
                inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps, condition=classes
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        scaler.scale(loss).backward()
        scaler.step(optimizer_diff)
        scaler.update()

        epoch_loss += loss.item()

    LOGGER.info(f"{epoch}: loss={epoch_loss / (step + 1):.5f}")
    wandb.log({ "diffusion/epoch": epoch,
                "diffusion/loss": epoch_loss / (step + 1),
              })
    epoch_loss_list.append(epoch_loss / (step + 1))

    if (epoch + 1) in (np.round(np.arange(0.0, 1.01, run_config["gen_image_intervall"]) * n_epochs)):
        eval_generate_sample_images(inferer, autoencoder, unet, scheduler, None, "Synthetic Images (training)")
# -

diffusion_stopwatch.stop().display()
save_model_as_artifact(wandb_run, unet, type(unet).__name__, diffusion_model_unet_config, model_directory)

# ### Plotting sampling example
#
# Finally, we generate an image with our LDM. For that, we will initialize a latent representation with just noise. Then, we will use the `unet` to perform 1000 denoising steps. In the last step, we decode the latent representation and plot the sampled image.

for i in range(10):
    eval_generate_sample_images(inferer, autoencoder, unet, scheduler, None, "Synthetic Images")

# ## Clean-up data

wandb.finish()
