#!/usr/bin/env python3
import wandb
import torch
from src.util import load_model, save_model
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
import os.path

auto_encoder_config= {
    "spatial_dims":3,
    "in_channels":1,
    "out_channels":1,
    "num_channels":(32, 64, 64),
    "latent_channels":3,
    "num_res_blocks":2,
    "norm_num_groups":16,
    "attention_levels":(False, False, True),
}

run_config={"batch_size": 1,
            "input_image_downsampling_factors": (1.78, 1.78, 1.7),
            "input_image_crop_roi": (128, 128, 96),
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
    "n_epochs" : 2,
    "autoencoder_warm_up_n_epochs" : 0,
}

diffusion_model_unet_config = {
    "spatial_dims":3,
    "in_channels":3,
    "out_channels":3,
    "num_res_blocks":1,
    "num_channels":(64, 64, 128),
    "attention_levels":(False, True, True),
    "num_head_channels":(0, 64, 64),
}

diffusion_model_training_config = {
    "n_epochs" : 5,
}



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


autoencoder = AutoencoderKL(
    **auto_encoder_config
).to(device)


test_image = torch.rand((1, 1, 
                         run_config["input_image_crop_roi"][0],
                         run_config["input_image_crop_roi"][1],
                         run_config["input_image_crop_roi"][2]
                         
                         )).to(device)

test_encoded = autoencoder.encode_stage_2_inputs(test_image.to(device))

print("Actual encoding shape", test_encoded.shape)

down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (1, auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
print("Theoretical encoding shape:", encoding_shape)

from generative.inferers import LatentDiffusionInferer
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.networks.schedulers import DDPMScheduler

unet = DiffusionModelUNet(
    **diffusion_model_unet_config
)
unet.to(device)


scheduler = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0195)

scale_factor = 1
# -

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

# ### Train diffusion model

# +
n_epochs = diffusion_model_training_config["n_epochs"]
epoch_loss_list = []
autoencoder.eval()


def eval_generate_sample_images(inferer, autoencoder, unet, scheduler, path, prefix_string):
    autoencoder.eval()
    unet.eval()
    noise = torch.randn( encoding_shape)
    noise = noise.to(device)
    scheduler.set_timesteps(num_inference_steps=1000)
    synthetic_images = inferer.sample(
        input_noise=noise, autoencoder_model=autoencoder, diffusion_model=unet, scheduler=scheduler
    )

    return synthetic_images[0, 0].detach().cpu() # images

result = eval_generate_sample_images(inferer, autoencoder, unet, scheduler, None, None)

print("Done! Produced image shape:", result.shape)