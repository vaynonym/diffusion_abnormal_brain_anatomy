import yaml
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
import time
import yaml
from datetime import timedelta
from pathlib import Path
import os
from typing import Optional

from monai.data.meta_tensor import MetaTensor
from src.directory_management import BASE_DIRECTORY
from src.logging_util import LOGGER, all_logging_disabled
from src.torch_setup import device
from src.synthseg_masks import synthseg_class_to_string_map

def load_wand_credentials():
    with open(os.path.join(BASE_DIRECTORY, "local_config.yml")) as file:
        local_user_config = yaml.safe_load(file)
    
    return local_user_config["project"], local_user_config["entity"]

def normalize(x):
    return (x - np.min(x)) / (np.max(x) - np.min(x))

def log_image_with_mask(image: np.ndarray, 
                        original_mask: np.ndarray,
                        reconstruction_image: np.ndarray = None,
                        reconstruction_mask: np.ndarray = None,
                        description_prefix="",
                        conditioning_information:Optional[torch.Tensor] = None,
                        class_label:Optional[torch.Tensor] = None
                        ):

    if conditioning_information is not None:
        conditioning_information = str([round(c, 2) for c in conditioning_information.tolist() ])

    if class_label is not None:
        class_label = str([round(c, 2) for c in class_label.tolist() ])

    data = []
    for name, get_indices in [ 
                              ("axial", lambda x:  x[..., x.shape[2] // 2]),
                              ("coronal", lambda x: x[:, x.shape[1] // 2, ...]),
                              ("sagittal", lambda x: x[x.shape[0] // 2, ...]),
                             ]:
        input_image = wandb.Image(get_indices(image), 
                masks={ "image": {
                                            "mask_data": get_indices(original_mask),
                                            "class_labels": synthseg_class_to_string_map
                                          }
             })
        
        columns = [name, conditioning_information if class_label is None else conditioning_information + " | " + class_label, input_image]

        if reconstruction_image is not None and reconstruction_mask is not None:
            side_image = wandb.Image(
                get_indices(reconstruction_image), 
                masks={ "reconstruction": {
                                            "mask_data": get_indices(reconstruction_mask),
                                            "class_labels": synthseg_class_to_string_map
                                          }
                      },
                )
            columns.append(side_image)
        data.append(columns)
    
    column_names = ["type", "conditioning", "image"]
    if reconstruction_image is not None and reconstruction_mask is not None:
        column_names.append("reconstruction")

    table = wandb.Table(columns=column_names, data=data)

    wandb.log({description_prefix: table})
    

# setting path to None means no graph will be created and only images logged
def log_image_to_wandb(img: np.ndarray,
                       reconstruction: np.ndarray = None,
                       description_prefix="",
                       log_to_wandb=False,
                       conditioning_information=None,
                       preprocess_image=None):
    if conditioning_information:
        conditioning_information = round(conditioning_information.item(), 2)
    if preprocess_image:
        img = preprocess_image(img)

    axial = img[..., img.shape[2] // 2]
    coronal = img[:, img.shape[1] // 2, ...]
    sagittal = img[img.shape[0] // 2, ...]
    multiplier = 2 if reconstruction is not None else 1
    fig, axs = plt.subplots(nrows=1, ncols=3 * multiplier, figsize=(12 *multiplier, 4), constrained_layout=False)

    fig.suptitle(f"N. Ventricle Volume: {conditioning_information}", fontweight="bold")

    for ax in axs:
        ax.axis("off")
    ax = axs[0 * multiplier]
    ax.imshow(axial, cmap="gray")
    ax = axs[1 * multiplier]
    ax.imshow(coronal, cmap="gray")
    ax = axs[2 * multiplier]
    ax.imshow(sagittal, cmap="gray")
    if reconstruction is not None:
        if preprocess_image:
            reconstruction = preprocess_image(reconstruction)
        reconstruction = np.clip(reconstruction, 0., 1.)
        axial_r = reconstruction[..., reconstruction.shape[2] // 2]
        coronal_r = reconstruction[:, reconstruction.shape[1] // 2, ...]
        sagittal_r = reconstruction[reconstruction.shape[0] // 2, ...]
        ax = axs[0 * multiplier + 1]
        ax.imshow(axial_r, cmap="gray")
        ax = axs[1 * multiplier + 1]
        ax.imshow(coronal_r, cmap="gray")
        ax = axs[2 * multiplier + 1]
        ax.imshow(sagittal_r, cmap="gray")

    plt.subplots_adjust(wspace=0.2, hspace=0)

    if log_to_wandb:
        # expects images in 0-1 range if floats
        # suppress warning "Images sizes do not match. This will causes images to be display incorrectly in the UI" as it's not actually a problem
        with all_logging_disabled():
            wandb.log({description_prefix: plt}) 
    
    plt.cla()
    plt.clf()
    plt.close("all")

def visualize_reconstructions(train_loader, autoencoder, num_examples):
    # ### Visualise reconstructions
    autoencoder.eval()
    with torch.no_grad():
        for i, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            reconstructions, _, _ = autoencoder(images) 
            r_img = reconstructions[0, 0].detach().cpu().numpy()
            img = images[0, 0].detach().cpu().numpy()
            log_image_to_wandb(img, r_img, "Reconstructions", True)
            
            if i+1 >= num_examples:
                break

import skimage
from src.colors import default_mask_colors

# setting path to None means no graph will be created and only images logged
def visualize_3d_image_slice_wise(img: MetaTensor, path, description_prefix="", log_to_wandb=False, conditioning_information=None, is_image_mask=False):
    axial = img[..., img.shape[2] // 2  - 20]
    coronal = img[:, img.shape[1] // 2 - 20, ...]
    sagittal = img[img.shape[0] // 2 - 20, ...]

    fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(16, 4), constrained_layout=False)
    if conditioning_information is not None:
        fig.suptitle(f"N. Ventricle Volume: {round(conditioning_information.item(), 3)}")

    for ax in axs:
        ax.axis("off")

    if is_image_mask:
        for i, m in enumerate([axial, coronal, sagittal]):
            colored_mask = default_mask_colors[m.cpu().detach().numpy()]
            axs[i].imshow(colored_mask)
    else:
        ax = axs[0]
        ax.imshow(np.clip(axial, 0., 1.), cmap="gray")
        ax = axs[1]
        ax.imshow(np.clip(coronal, 0., 1.), cmap="gray")
        ax = axs[2]
        ax.imshow(np.clip(sagittal, 0., 1.), cmap="gray")
    if path:
        plt.savefig(path)

    if log_to_wandb:
        # expects images in 0-1 range if floats
        # suppress warning "Images sizes do not match. This will causes images to be display incorrectly in the UI" as it's not actually a problem
        with all_logging_disabled():
            wandb.log({description_prefix: plt}) 

    for i in range(3):
        axs[i].clear()
    fig.clf()
    plt.cla()
    plt.clf()

class Stopwatch():
    def __init__(self, prefix_string: str):
        self.prefix_string = prefix_string
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self, exc_type, exc_value, traceback):
        self.stop().display()
    
    def start(self):
        self.start_time = time.monotonic()
        return self
    
    def stop(self):
        self.end_time = time.monotonic()
        return self
    
    def display(self, log_to_wandb=False):
        if self.start_time is None or self.end_time is None:
            LOGGER.warn("Warning: Incorrect use of Stopwatch! Need to start and stop before displaying!")
            return self
        timedelta_string = "{}".format(timedelta(seconds=self.end_time - self.start_time))
        LOGGER.info(self.prefix_string + timedelta_string)
        
        if log_to_wandb:
            wandb.log({self.prefix_string: timedelta_string})

        self.start_time = None
        self.end_time = None
        return self

# Intended to be used as a decorator abstraction for stopwatch applied to function calls
def measure_time(prefix_string: str):
    def inner(func):
        stopwatch = Stopwatch(prefix_string)
        stopwatch.start()
        func()
        stopwatch.stop().display()
    return inner

def read_config(path: str):
    return yaml.safe_load(Path(path).read_text())

def read_config_from_wandb_run(entity, project, id: str):
    api = wandb.Api()
    run = api.run(f"{entity}/{project}/{id}")
    #run.file("config.yml").download()
    relevant_configs = ["run_config",
                   "auto_encoder_config", 
                   "patch_discrim_config",
                   "auto_encoder_training_config",
                   "diffusion_model_unet_config",
                   "diffusion_model_training_config",]
    
    # map for backwards compatability:
    compatability_map = {
        "run_config" : "run",
        "auto_encoder_config" : "auto_encoder",
        "patch_discrim_config" : "patch_discrim",
        "auto_encoder_training_config" : "autoencoder_training",
        "diffusion_model_unet_config" : "diffusion_model_unet",
        "diffusion_model_training_config" : "diffusion_model_unet_training",
    }

    return {compatability_map[k]:v for k, v in run.config.items() if k in relevant_configs}