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
from generative.networks.schedulers import DDPMScheduler, DDIMScheduler, PNDMScheduler

import wandb
import numpy as np

from src.datasets import SyntheticLDM100K
from src.util import load_wand_credentials, log_image_to_wandb, Stopwatch, read_config 
from src.model_util import load_model_from_run_with_matching_config, check_dimensions
from src.logging_util import LOGGER
from src.diffusion import get_scale_factor
import os
import torch
import torch.nn.functional as F
from monai import transforms
from monai.data import DataLoader
from monai.utils import first, set_determinism
from torch.cuda.amp import GradScaler, autocast

from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

import wandb
import numpy as np

from src.util import load_wand_credentials, Stopwatch, read_config
from src.model_util import load_model_from_run_with_matching_config, check_dimensions
from src.logging_util import LOGGER
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor
from src.directory_management import DATA_DIRECTORY

import torch.multiprocessing

from src.torch_setup import device

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

train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["image"]),
        transforms.EnsureChannelFirstd(keys=["image"]),
        transforms.Lambdad(keys="image", func=lambda x: x[0, :, :, :]),
        transforms.EnsureChannelFirstd(keys=["image"], channel_dim="no_channel"),
        transforms.EnsureTyped(keys=["image"]),
        transforms.Orientationd(keys=["image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
        transforms.Lambdad(keys=["volume"], func = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)),
    ]
)

LOGGER.info("Loading small sample dataset...")

validation_ds = SyntheticLDM100K(
    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
    section="validation",  # validation
    size=25, # hard-coded to just get a few samples
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    seed=0,
    transform=train_transforms,
)


valid_loader = DataLoader(validation_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)

LOGGER.info("Finshed loading dataset")

autoencoder = AutoencoderKL(
    **auto_encoder_config
)

# for reproducibility purposes set a seed
set_determinism(42)



down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
LOGGER.info(f"Encoding shape: {encoding_shape}")

autoencoder = AutoencoderKL(
    **auto_encoder_config
)
autoencoder.to(device)


# Try to load identically trained autoencoder if it already exists. Else train a new one.
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config],
                                            ["auto_encoder_config", "auto_encoder_training_config"],
                                            project=project, entity=entity, 
                                            model=autoencoder, artifact_name=AutoencoderKL.__name__,
                                            ):
    LOGGER.error("This script expects existing autoencoder")
    quit()
else:
    LOGGER.info("Loaded existing autoencoder")


unet = DiffusionModelUNet(
    **diffusion_model_unet_config
)
unet.to(device)

if not load_model_from_run_with_matching_config([diffusion_model_training_config, diffusion_model_unet_config],
                                            ["diffusion_model_training_config", "diffusion_model_unet_config"],
                                            project=project, entity=entity, 
                                            model=unet, artifact_name=DiffusionModelUNet.__name__,
                                            ):
    LOGGER.error("This script expects existing Diffusion Model")
    quit()
else:
    LOGGER.info("Loaded existing Diffusion Model")


DDPM = DDPMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
DDIM = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
PNDM = PNDMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205)



scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data=next(iter(valid_loader))["image"])

# We define the inferer using the scale factor:

inferer = LatentDiffusionInferer(DDPM, scale_factor=scale_factor)

all_timesteps = [1, 5, 10, 25, 50, 100, 1000]
schedulers = [PNDM, DDPM, DDIM]

image_slices = [ 
    ("axial", lambda x:  x[..., x.shape[2] // 2]),
    ("coronal", lambda x: x[:, x.shape[1] // 2, ...]),
    ("sagittal", lambda x: x[x.shape[0] // 2, ...]),
]

from collections import defaultdict

def test_samplers(autoencoder, unet):
    autoencoder.eval()
    unet.eval()
    results = defaultdict(lambda: defaultdict(dict))

    with torch.no_grad():
        conditioning = (torch.tensor([[[0.5]]] * run_config["batch_size"])).to(device)
        latent_noise = torch.randn(encoding_shape).to(device)


        for scheduler in schedulers:
            scheduler_name = scheduler.__class__.__name__

            for timesteps in all_timesteps:
                # for some reason PNDM doesn't work for /some/ values under 10, so we just skip it here
                if timesteps < 6 and scheduler is PNDM:
                    for slice_name, slice_f in image_slices:
                        results[scheduler_name][timesteps][slice_name] = None
                    continue

                scheduler.set_timesteps(num_inference_steps=timesteps)
                synthetic_images = inferer.sample(input_noise=latent_noise,
                                                  autoencoder_model=autoencoder,
                                                  diffusion_model=unet,
                                                  scheduler=scheduler,
                                                  conditioning=conditioning)
                synthetic_images = torch.clamp(synthetic_images, 0., 1.)

                conditioning[0, 0].detach().cpu()
                img = synthetic_images[0, 0].detach().cpu().numpy()  # images

                for slice_name, slice_f in image_slices:
                    results[scheduler_name][timesteps][slice_name] = slice_f(img)
    image_slice_names = [x[0] for x in image_slices]
    for scheduler in schedulers:
        scheduler_name = scheduler.__class__.__name__
        data = []
        for timesteps in all_timesteps:
            columns = [str(timesteps)]
            for slice_name in image_slice_names:
                img = results[scheduler_name][timesteps][slice_name]
                columns.append(wandb.Image(img) if img is not None else None)
            data.append(columns)

        column_names = ["timesteps", *image_slice_names]

        table = wandb.Table(columns=column_names, data=data)

        wandb.log({str(scheduler_name): table})




with Stopwatch("Diffusion sampling took: ").start():
    test_samplers(autoencoder=autoencoder, unet=unet)

# ## Clean-up data

wandb.finish()
