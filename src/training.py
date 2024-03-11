import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.inferers import LatentDiffusionInferer

from src.util import Stopwatch, log_image_to_wandb, log_image_with_mask
from src.logging_util import LOGGER
from src.evaluation import evaluate_diffusion_model
from src.diffusion import generate_and_log_sample_images
from src.torch_setup import device

def train_diffusion_model(autoencoder: AutoencoderKL, unet: DiffusionModelUNet,
                          optimizer_diff: torch.optim.Optimizer, scaler: GradScaler,
                          inferer: LatentDiffusionInferer,
                          train_loader: DataLoader, valid_loader: DataLoader,
                          encoding_shape: torch.Size,
                          n_epochs: int, 
                          evaluation_intervall: float,
                          get_input_from_batch = lambda batch: (batch["image"].to(device), batch["volume"].to(device)),
                          postprocessing_for_logging = None,
                          ):
    
    wandb.define_metric("diffusion/training/epoch")
    wandb.define_metric("diffusion/training/*", step_metric="diffusion/training/epoch")

    for epoch in range(n_epochs):
        autoencoder.eval()
        unet.train()
        
        epoch_loss = 0

        for step, batch in enumerate(train_loader):
            network_input, conditioning = get_input_from_batch(batch)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn(encoding_shape).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (network_input.shape[0],), device=network_input.device
                ).long()

                # Get model prediction
                noise_pred = inferer(
                    inputs=network_input, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps, condition=conditioning
                )

                loss = F.mse_loss(noise_pred.float(), noise.float())

            scaler.scale(loss).backward()
            scaler.step(optimizer_diff)
            scaler.update()

            epoch_loss += loss.item()

        LOGGER.info(f"{epoch}: loss={epoch_loss / (step + 1):.5f}")
        wandb.log({ 
                    "diffusion/training/epoch": epoch,
                    "diffusion/training/loss": epoch_loss / (step + 1),
                  })
        
        # Validation
        if (epoch + 1) in (np.round(np.arange(0.0, 1.01, evaluation_intervall) * n_epochs)):

            with Stopwatch("Sampling example images took: "):
                generate_and_log_sample_images(autoencoder=autoencoder, unet=unet, 
                                               scheduler=inferer.scheduler,
                                               encoding_shape=encoding_shape, 
                                               inferer=inferer,
                                               prefix_string="diffusion/training/synthetic images")
            
            with Stopwatch("Generating metrics took: "):
                evaluate_diffusion_model(diffusion_model=unet,
                                         scheduler=inferer.scheduler,
                                         autoencoder=autoencoder,
                                         latent_shape=encoding_shape,
                                         inferer=inferer,
                                         val_loader=valid_loader,
                                         get_input_from_batch=get_input_from_batch
                                         )