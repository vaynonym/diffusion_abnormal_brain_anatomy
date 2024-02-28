import yaml
import torch
import matplotlib.pyplot as plt
import wandb
import warnings
import numpy as np
from monai.data.meta_tensor import MetaTensor
import time
from datetime import timedelta
from src.logging_util import LOGGER, all_logging_disabled
import yaml
from pathlib import Path

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using {device}")

def load_wand_credentials():
    with open("./local_config.yml") as file:
        local_user_config = yaml.safe_load(file)
    
    return local_user_config["project"], local_user_config["entity"]

# setting path to None means no graph will be created and only images logged
def visualize_3d_image_slice_wise(img: MetaTensor, path, description_prefix="", log_to_wandb=False):
    axial = img[..., img.shape[2] // 2]
    coronal = img[:, img.shape[1] // 2, ...]
    sagittal = img[img.shape[0] // 2, ...]

    if path:
        fig, axs = plt.subplots(nrows=1, ncols=3)
        for ax in axs:
            ax.axis("off")
        ax = axs[0]
        ax.imshow(axial, cmap="gray")
        ax = axs[1]
        ax.imshow(coronal, cmap="gray")
        ax = axs[2]
        ax.imshow(sagittal, cmap="gray")
        plt.savefig(path)
        plt.cla()
        plt.clf()

    def normalize(x):
        return (x - np.min(x)) / (np.max(x) - np.min(x))

    if log_to_wandb:
        # expects images in 0-1 range if floats
        wandb_images = [ wandb.Image(normalize(img), mode="L", caption=caption) for img, caption in zip([axial, coronal, sagittal], ["Axial", "Coronal", "Sagittal"])]
        # suppress warning "Images sizes do not match. This will causes images to be display incorrectly in the UI" as it's not actually a problem
        with all_logging_disabled():
            wandb.log({description_prefix: wandb_images}) 

class Stopwatch():
    def __init__(self, prefix_string: str):
        self.prefix_string = prefix_string
        self.start_time = None
        self.end_time = None

    def __enter__(self):
        self.start()
        return self
    
    def __exit__(self):
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