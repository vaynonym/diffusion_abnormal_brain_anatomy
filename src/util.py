import yaml
import torch
import matplotlib.pyplot as plt
import wandb
import numpy as np
import PIL
from monai.data.meta_tensor import MetaTensor
import torchvision.transforms as T

def load_wand_credentials():
    with open("./local_config.yml") as file:
        local_user_config = yaml.safe_load(file)
    
    return local_user_config["project"], local_user_config["entity"]

def save_model(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer=None):

    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)

def load_model(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer=None):
    state = torch.load(path)
    model.load_state_dict(state["model"])

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])

def visualize_3d_image_slice_wise(img: MetaTensor, path, description_prefix="", log_to_wandb=False):
    axial = img[..., img.shape[2] // 2]
    coronal = img[:, img.shape[1] // 2, ...]
    sagittal = img[img.shape[0] // 2, ...]

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
        wandb.log({description_prefix: wandb_images}) 

