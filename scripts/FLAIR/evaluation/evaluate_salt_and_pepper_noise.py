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

# for reproducibility purposes set a seed
set_determinism(42)

np.set_printoptions(formatter={"float": lambda x: "{0:0.2f}".format(x)})

torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")



_, entity = load_wand_credentials()

WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID =  "lg81hq9f"
SPADE_Project = "thesis"
spade_experiment_config = read_config_from_wandb_run(entity, SPADE_Project, WANDB_SPADE_AUTOENCODER_MODEL_RUN_ID)
run_config = spade_experiment_config["run"]
run_config["batch_size"] = 3
run_config["oversample_large_ventricles"] = False

from src.datasets import get_default_transforms, load_dataset_from_config, get_dataloader

_, val_transforms = get_default_transforms(run_config, None)
validation_ds = load_dataset_from_config(run_config, section="validation", transforms=val_transforms)
validation_dl = get_dataloader(validation_ds, run_config)


model_ids = {
    "first model used": "xrsbiz3d",
    "no discriminator": "n8ja1w79",
    "larger less relevant discrim": "bw0sjlju",
    "higher capacity": "h1pbmai3",
    "final model used": "lg81hq9f",
}

results = dict()

for name, wandb_id in model_ids.items():

    spade_experiment_config = read_config_from_wandb_run(entity, SPADE_Project, wandb_id)
    spade_run_config = spade_experiment_config["run"]
    spade_auto_encoder_config = spade_experiment_config["auto_encoder"]
    spade_diffusion_model_unet_config = spade_experiment_config["diffusion_model_unet"]

    def get_encoding_shape(A_config, R_config):
        down_sampling_factor = (2 ** (len(A_config["num_channels"]) -1) )
        dim_xyz = tuple(map(lambda x: x // down_sampling_factor, R_config["input_image_crop_roi"]))
        encoding_shape = (run_config["batch_size"], A_config["latent_channels"], dim_xyz[0], dim_xyz[1], dim_xyz[2])
        return encoding_shape

    spade_encoding_shape = get_encoding_shape(spade_auto_encoder_config, spade_run_config)
    #spade_encoding_shape = (mask_encoding_shape[0], spade_encoding_shape[1], mask_encoding_shape[2], mask_encoding_shape[3], mask_encoding_shape[4])
    LOGGER.info(f"Spade encoding shape: {spade_encoding_shape}")




    ################## Spade Image Generation Model #################

    LOGGER.info("Loading Image Generator")

    spade_autoencoder = load_model_from_run(run_id=wandb_id, project=SPADE_Project, entity=entity,
                                    #model_class=AutoencoderKL,
                                    model_class=SPADEAutoencoderKL,  
                                    create_model_from_config=None,
                                    #create_model_from_config=create_embedding_autoencoder
                                    ).eval().to(device)
    
    from src.synthseg_masks import white_matter_indices, encode_one_hot, brain_stem_indices

    white_matter_stds = []
    count = 0
    with torch.no_grad():
        for i, batch in tqdm(enumerate(validation_dl)):
            image : torch.Tensor = batch["image"].to(device)
            mask : torch.Tensor = batch["mask"].to(device)

            reconstruction = spade_autoencoder.reconstruct(image, encode_one_hot(mask))

            white_matter_only_mask = torch.isin(mask, white_matter_indices)
            for batch_i in range(image.shape[0]):

                reconstructed_white_matter_standard_deviation = reconstruction[batch_i].masked_select(white_matter_only_mask[batch_i]).std()
                white_matter_stds.append(reconstructed_white_matter_standard_deviation.item())
                count += 1
                if count == 200:
                    break

            if count == 200:
                break
    
    results[name] = np.mean(white_matter_stds)


white_matter_stds = []
count = 0
with torch.no_grad():
    for i, batch in tqdm(enumerate(validation_dl)):
        image : torch.Tensor = batch["image"].to(device)
        mask : torch.Tensor = batch["mask"].to(device)
        
        white_matter_only_mask = torch.isin(mask, white_matter_indices)
        for batch_i in range(image.shape[0]):

            white_matter_standard_deviation = image[batch_i].masked_select(white_matter_only_mask[batch_i]).std()
            white_matter_stds.append(reconstructed_white_matter_standard_deviation.item())
            count += 1
            if count == 200:
                break

        if count == 200:
            break

results["ground truth"] = np.mean(white_matter_stds)

import pprint

pprint.pprint(results)


















