from src.util import Stopwatch, log_image_to_wandb
from src.logging_util import LOGGER
import torch
import wandb
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL

from monai.networks.nets.resnet import resnet50
import torch.nn as nn
from torch.utils.data import DataLoader 

import os.path
from collections import OrderedDict
import torch.nn.functional as F
import torchvision

from src.directory_management import PRETRAINED_MODEL_DIRECTORY
from src.torch_setup import device

def evaluate_autoencoder(val_loader, autoencoder, is_training=True):
    metrics = { 
        "MS-SSIM": MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4),
        "SSIM": SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4),
        "MMD": MMDMetric(),
    }
    
    accumulations = [[] for _ in metrics.keys()]


    for _, batch in enumerate(val_loader):
        image = batch["image"].to(device)

        with torch.no_grad():
            image_recon = autoencoder.reconstruct(image)

            for metric, accumulator in zip(metrics.values(), accumulations):
                res = metric(image, image_recon)
                if(len(res.size()) == 0):
                    res = res.unsqueeze(0)
                accumulator.append(res)

    with torch.no_grad():
        prefix = "autoencoder/training" if is_training else "autoencoder"
        wandb.log(
            {
                f"{prefix}/{name}": (torch.cat(accumulator, dim=0).mean().item()) for (name, accumulator) in zip(metrics.keys(), accumulations)
            }
        )
    
    return None


############### Diffusion Model Evaluation #####################

def evaluate_diffusion_model(diffusion_model: nn.Module,
                             scheduler,
                             autoencoder: AutoencoderKL,
                             latent_shape: torch.Tensor,
                             inferer: LatentDiffusionInferer,
                             val_loader: DataLoader):
    weights_path = os.path.join(PRETRAINED_MODEL_DIRECTORY, "resnet_50.pth")
    assert os.path.exists(weights_path)

    feature_extraction_model = resnet50(pretrained=False,
                                        shortcut_type="B",
                                        feed_forward=False,
                                        bias_downsample=False,
                                        n_input_channels=1,
                                        spatial_dims=3
                                        ).to(device)
    feature_extraction_model.eval()

    loaded = torch.load(weights_path)
    state_dict = loaded["state_dict"]
    state_dict_without_data_parallel = OrderedDict()
    prefix_len = len("module.")
    for k, v in state_dict.items():
        name = k[prefix_len:] # remove "module." prefix
        state_dict_without_data_parallel[name] = v

    feature_extraction_model.load_state_dict(state_dict_without_data_parallel)

    def medicalNetNormalize(img):
        return nn.functional.interpolate(
            (img - img.mean(dim=(1,2,3,4), keepdim=True)) / img.std(dim=(1,2,3,4), keepdim=True),
            size=(125, 125, 125), mode="trilinear"
            )
    
    synth_features = []
    real_features = []
    losses = []

    # get latent representations of all real data (for FID)
    with Stopwatch("Getting real features took: "):
        for i, x in enumerate(val_loader):
            real_images = x["image"].to(device)
            with torch.no_grad():
                normalized = medicalNetNormalize(real_images)
                if i < 2:
                    log_image_to_wandb(real_images[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, False)
                    log_image_to_wandb(real_images[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, False)

                    log_image_to_wandb(normalized[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, False)
                    log_image_to_wandb(normalized[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, False)

                real_eval_feats : torch.Tensor = feature_extraction_model(normalized)
                real_features.append(real_eval_feats)

    # calculate validation loss for sample
    # get latent representations of synthetic images (for FID)
    number_images_to_consider = 20
    for _, x in enumerate(val_loader):
        # Get the real images
        real_images = x["image"].to(device)
        conditioning = x["volume"].to(device)

        # Generate some synthetic images using the defined model
        latent_noise = torch.randn(latent_shape).to(device)
        scheduler.set_timesteps(num_inference_steps=1000)

        with torch.no_grad():
            # Generating images far dominates the time to extract features
            syn_images = inferer.sample(input_noise=latent_noise, autoencoder_model=autoencoder, diffusion_model=diffusion_model, scheduler=scheduler, conditioning=conditioning)
            syn_images = torch.clamp(syn_images, 0., 1.)

            losses.append(F.mse_loss(syn_images.float(), real_images.float()).detach())
            syn_images = medicalNetNormalize(syn_images)

            # Get the features for the synthetic data
            synth_eval_feats = feature_extraction_model(syn_images)
            synth_features.append(synth_eval_feats)
        
        batch_size = synth_features[0].shape[0]
        
        if len(synth_features) * batch_size >= number_images_to_consider:
            break

    synth_features = torch.vstack(synth_features)
    real_features = torch.vstack(real_features)
    mean_loss = torch.stack(losses).mean()

    fid = FIDMetric()
    fid_res = fid(synth_features, real_features)
    wandb.log({"diffusion/training/FID": fid_res.item(),
               "diffusion/training/valid_loss": mean_loss.item()})
    
    print(f"FID Score: {fid_res.item():.4f}")