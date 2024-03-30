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

from src.util import load_wand_credentials, log_image_to_wandb, Stopwatch, read_config, visualize_reconstructions
from src.model_util import save_model_as_artifact, load_model_from_run_with_matching_config, check_dimensions
from src.logging_util import LOGGER
from src.directory_management import DATA_DIRECTORY, OUTPUT_DIRECTORY
from src.datasets import SyntheticLDM100K
from src.diffusion import get_scale_factor
from src.custom_autoencoders import EmbeddingWrapper


import torch.multiprocessing

from src.torch_setup import device

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

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

# for reproducibility purposes set a seed
set_determinism(42)



down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
image_shape = (run_config["batch_size"], 1, run_config["input_image_crop_roi"][0], run_config["input_image_crop_roi"][1], run_config["input_image_crop_roi"][2])
LOGGER.info(f"Image shape: {image_shape}")
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))

LOGGER.info("Loading models...")

base_autoencoder = AutoencoderKL(
    **auto_encoder_config
).to(device)

from src.synthseg_masks import synthseg_classes

autoencoder = EmbeddingWrapper(base_autoencoder=base_autoencoder, vocabulary_size=max(synthseg_classes) + 1, embedding_dim=64)


# Try to load identically trained autoencoder if it already exists. Else train a new one.
if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config],
                                            ["auto_encoder_config", "auto_encoder_training_config"],
                                            project=project, entity=entity, 
                                            model=autoencoder, artifact_name=autoencoder.__class__.__name__,
                                            ):
    LOGGER.error("This script expects existing autoencoder")
    quit()
else:
    LOGGER.info("Loaded existing autoencoder")


diffusion_model = DiffusionModelUNet(
    **diffusion_model_unet_config
).to(device)

if not load_model_from_run_with_matching_config([auto_encoder_config, auto_encoder_training_config, diffusion_model_training_config, diffusion_model_unet_config],
                                            ["auto_encoder_config", "auto_encoder_training_config", "diffusion_model_training_config", "diffusion_model_unet_config"],
                                            project=project, entity=entity, 
                                            model=diffusion_model, artifact_name=diffusion_model.__class__.__name__,
                                            ):
    LOGGER.error("This script expects existing Diffusion Model")
    quit()
else:
    LOGGER.info("Loaded existing Diffusion Model")


LOGGER.info("Loading dataset...")


train_transforms = transforms.Compose(
    [
        transforms.LoadImaged(keys=["mask", "image"]),
        transforms.EnsureChannelFirstd(keys=["mask", "image"]),
        transforms.EnsureTyped(keys=["mask", "image"]),
        transforms.Orientationd(keys=["mask", "image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["mask"], pixdim=run_config["input_image_downsampling_factors"], mode=("nearest")),
        transforms.Spacingd(keys=["image"], pixdim=run_config["input_image_downsampling_factors"], mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["mask", "image"], roi_size=run_config["input_image_crop_roi"]),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
        transforms.Lambdad(keys=["volume"], func = lambda x: torch.tensor(x, dtype=torch.float32).unsqueeze(0).unsqueeze(0)),
    ]
)

dataset_size = run_config["dataset_size"]

validation_ds = SyntheticLDM100K(
    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
    section="validation",  # validation
    size=dataset_size,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    seed=0,
    transform=train_transforms,
)


valid_loader = DataLoader(validation_ds, batch_size=run_config["batch_size"], shuffle=True, num_workers=2, drop_last=True, persistent_workers=True)


LOGGER.info(f"Valid length: {len(validation_ds)} in {len(valid_loader)} batches")
LOGGER.info(f'Mask shape {validation_ds[0]["mask"].shape}')


from collections import defaultdict
from src.synthseg_masks import ventricle_indices, white_matter_indices, cortex_indices, CSF_indices, background_indices


stopwatch = Stopwatch("Calculating volumes over validationset took: ").start()
bucket_size = 0.1
buckets = torch.tensor(np.arange(0, 1.0, bucket_size)).to(device)
# ensure that every normal value is within 0 to 1 buckets, essentially making [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0], (1.0, 1.1], ...
buckets[0] = -0.001 

classes = [
    ("ventricles", ventricle_indices, {i:[] for i in range(len(buckets)+1)}),
    ("white_matter", white_matter_indices, {i:[] for i in range(len(buckets)+1)}),
    ("CSF", CSF_indices, {i:[] for i in range(len(buckets)+1)}),
    ("cortex", cortex_indices , {i:[] for i in range(len(buckets)+1)}),
    ("background", background_indices, {i:[] for i in range(len(buckets)+1)}),
]

def append_voxel_count(mask, buckets_of_batch, relevant_indices, result_voxel_counts):
    voxel_count = torch.isin(mask, relevant_indices).sum(dim=(1,2,3,4))
    for batch_i in range(mask.shape[0]):
        result_voxel_counts[buckets_of_batch[batch_i].item()].append(voxel_count[batch_i].float())
    return result_voxel_counts

for i, batch in enumerate(valid_loader):
    mask = batch["mask"].int().to(device)
    conditioning_value = batch["volume"].to(device)
    buckets_of_batch = torch.bucketize(input=conditioning_value, boundaries=buckets, right=True)

    for batch_i in range(mask.shape[0]):
        for (name, indices, voxel_counts) in classes:
            append_voxel_count(mask, buckets_of_batch, indices, voxel_counts)



for (name, _ , voxel_counts) in classes:
    maxes = [torch.stack(v).max().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()]
    LOGGER.info(f"Max {name}: {np.array(maxes)}")
    LOGGER.info(f"Real (Bin, #Samples): {[(i, len(x)) for i, x in enumerate(voxel_counts.values())]}")


results = [(name,
            {k:torch.stack(v).mean().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()}, 
            {k:(torch.stack(v).mean().item() - torch.stack(v).min().item() if len(v) > 0 else 0,
                torch.stack(v).max().item() - torch.stack(v).mean().item() if len(v) > 0 else 0,
                ) for k, v in voxel_counts.items()}, 
            ) for (name, _, voxel_counts) in classes]

stopwatch.stop().display()


stopwatch = Stopwatch("drawing plots took: ").start()

import matplotlib.pyplot as plt


print(np.arange(0, 1.0, bucket_size) + 0.05)

for name, mean_voxel_counts, min_max_voxel_counts in results:
    min_max_error = np.array(list(min_max_voxel_counts.values())).T
    plt.bar(x=np.arange(-bucket_size, 1.0, bucket_size) + bucket_size/2,
            width=bucket_size,
            height=mean_voxel_counts.values(),
            yerr=min_max_error)
    plt.savefig(f"{OUTPUT_DIRECTORY}/ground_truth_{name}_volume.png")
    plt.clf()

    LOGGER.info(f"{name}: {np.array(list(mean_voxel_counts.values()))}")

stopwatch.stop().display()

LOGGER.info("DONE!")

scheduler = DDIMScheduler(num_train_timesteps=1000, schedule="scaled_linear_beta", beta_start=0.0015, beta_end=0.0205, clip_sample=False)
scheduler.set_timesteps(num_inference_steps=25)

scale_factor = get_scale_factor(autoencoder=autoencoder, sample_data= next(iter(valid_loader))["mask"].int().to(device))

inferer = LatentDiffusionInferer(scheduler, scale_factor=scale_factor)

autoencoder.eval()

# include abnormal values
buckets = torch.tensor(np.arange(0, 2.0, bucket_size)).to(device)
# ensure that every normal value is within 0 to 1 buckets, essentially making [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0], (1.0, 1.1], ...
buckets[0] = -0.001 


classes = [
    ("ventricles", ventricle_indices, {i:[] for i in range(len(buckets) + 1)}),
    ("white_matter", white_matter_indices, {i:[] for i in range(len(buckets) + 1)}),
    ("CSF", CSF_indices, {i:[] for i in range(len(buckets) +1)}),
    ("cortex", cortex_indices , {i:[] for i in range(len(buckets)+1)}),
    ("background", background_indices, {i:[] for i in range(len(buckets)+1)}),
]

from src.synthseg_masks import decode_one_hot
down_sampling_factor = (2 ** (len(auto_encoder_config["num_channels"]) -1) )
dim_xyz = tuple(map(lambda x: x // down_sampling_factor, run_config["input_image_crop_roi"]))
encoding_shape = (run_config["batch_size"], auto_encoder_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])

LOGGER.info(f"Encoding shape: {encoding_shape}")
stopwatch = Stopwatch("Generating synthetic examples: ").start()
LOGGER.info("Generating synthetic examples...")
for i, batch in enumerate(valid_loader):
    conditioning_value = batch["volume"].to(device)
    latent_noise = torch.randn(encoding_shape).to(device)    
    mask = decode_one_hot(inferer.sample(input_noise=latent_noise,
                                autoencoder_model=autoencoder,
                                diffusion_model=diffusion_model, 
                                conditioning=conditioning_value)).int()

    buckets_of_batch = torch.bucketize(input=conditioning_value, boundaries=buckets, right=True)

    for batch_i in range(mask.shape[0]):
        for (name, indices, voxel_counts) in classes:
            append_voxel_count(mask, buckets_of_batch, indices, voxel_counts)

stopwatch.stop().display()



from torch.utils.data import TensorDataset

dataset = torch.arange(1.05, 2.0, 0.1).unsqueeze(0).repeat(10, 1).flatten().unsqueeze(1).unsqueeze(2)

fake_volume_dataset = TensorDataset(dataset)

fake_volume_dataloader = DataLoader(fake_volume_dataset,
                                    batch_size=run_config["batch_size"],
                                    shuffle=True, num_workers=2,
                                    drop_last=True,
                                    persistent_workers=True)

stopwatch = Stopwatch("Generating abnormal synthetic samples: ").start()

for batch in iter(fake_volume_dataloader):
    conditioning_value = batch[0].to(device)
    latent_noise = torch.randn(encoding_shape).to(device)    

    mask = decode_one_hot(inferer.sample(input_noise=latent_noise,
                                autoencoder_model=autoencoder,
                                diffusion_model=diffusion_model, 
                                conditioning=conditioning_value)).int()

    buckets_of_batch = torch.bucketize(input=conditioning_value, boundaries=buckets, right=True)

    for batch_i in range(mask.shape[0]):
        for (name, indices, voxel_counts) in classes:
            append_voxel_count(mask, buckets_of_batch, indices, voxel_counts)

stopwatch.stop().display()

LOGGER.info(f"Synthetic: (Bin, #Samples): {[(i, len(x)) for i, x in enumerate(voxel_counts.values())]}")

for (name, _ , voxel_counts) in classes:
    maxes = [torch.stack(v).max().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()]
    LOGGER.info(f"Max {name}: {np.array(maxes)}")
    LOGGER.info(f"Real (Bin, #Samples): {[(i, len(x)) for i, x in enumerate(voxel_counts.values())]}")

results = [(name,
            {k:torch.stack(v).mean().item() if len(v) > 0 else 0 for k, v in voxel_counts.items()}, 
            {k:(torch.stack(v).mean().item() - torch.stack(v).min().item() if len(v) > 0 else 0,
                torch.stack(v).max().item() - torch.stack(v).mean().item() if len(v) > 0 else 0,
                ) for k, v in voxel_counts.items()}, 
            ) for (name, _, voxel_counts) in classes]


for name, mean_voxel_counts, min_max_voxel_counts in results:

    min_max_error = np.array(list(min_max_voxel_counts.values())).T
    plt.bar(x=np.arange(-bucket_size, 2.0, bucket_size) + bucket_size/2,
            width=bucket_size,
            height=mean_voxel_counts.values(),
            yerr=min_max_error)

    plt.savefig(f"{OUTPUT_DIRECTORY}/synthetic_{name}_volume.png")
    plt.clf()

    LOGGER.info(f"{name}: {np.array(list(mean_voxel_counts.values()))}")


