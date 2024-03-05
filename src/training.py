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

from src.util import device, Stopwatch, log_image_to_wandb
from src.logging_util import LOGGER
from src.evaluation import evaluate_autoencoder, evaluate_diffusion_model
from src.diffusion import generate_and_log_sample_images



def train_autoencoder(autoencoder: AutoencoderKL, train_loader: DataLoader, val_loader: DataLoader, patch_discrim_config: dict, auto_encoder_training_config: dict, evaluation_intervall: float, WANDB_LOG_IMAGES: bool):
    train_stopwatch = Stopwatch("Autoencoder training time: ").start()

    wandb_prefix = "autoencoder/training"
    
    discriminator = PatchDiscriminator(**patch_discrim_config)
    discriminator.to(device)
    # -

    # ### Defining Losses
    #
    # We will also specify the perceptual and adversarial losses, including the involved networks, and the optimizers to use during the training process.

    # +
    l1_loss = L1Loss()
    adv_loss = PatchAdversarialLoss(criterion="least_squares")
    loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2)
    loss_perceptual.to(device)


    def KL_loss(z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]


    adv_weight = 0.01
    perceptual_weight = 0.001
    kl_weight = 1e-6
    # -

    lr=auto_encoder_training_config["learning_rate"]
    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=lr)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=lr)

    # ### Train model

    # +
    n_epochs = auto_encoder_training_config["n_epochs"]
    autoencoder_warm_up_n_epochs = auto_encoder_training_config["autoencoder_warm_up_n_epochs"]

    wandb.define_metric(f"{wandb_prefix}/epoch")
    wandb.define_metric(f"{wandb_prefix}/*", step_metric=f"{wandb_prefix}/epoch")
    for epoch in range(n_epochs):
        autoencoder.train()
        discriminator.train()
        epoch_loss = 0
        gen_epoch_loss = 0
        disc_epoch_loss = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)  # choose only one of Brats channels

            # Generator part
            optimizer_g.zero_grad(set_to_none=True)
            reconstruction, z_mu, z_sigma = autoencoder(images)
            kl_loss = KL_loss(z_mu, z_sigma)

            recons_loss = l1_loss(reconstruction.float(), images.float())
            p_loss = loss_perceptual(reconstruction.float(), images.float())
            loss_g = recons_loss + kl_weight * kl_loss + perceptual_weight * p_loss

            if epoch >= autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch >= autoencoder_warm_up_n_epochs:
                # Discriminator part
                optimizer_d.zero_grad(set_to_none=True)
                logits_fake = discriminator(reconstruction.contiguous().detach())[-1]
                loss_d_fake = adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
                logits_real = discriminator(images.contiguous().detach())[-1]
                loss_d_real = adv_loss(logits_real, target_is_real=True, for_discriminator=True)
                discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

                loss_d = adv_weight * discriminator_loss

                loss_d.backward()
                optimizer_d.step()

            epoch_loss += recons_loss.item()
            if epoch >= autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

        LOGGER.info(f"{epoch} | recons_loss={epoch_loss / (step + 1):.5f}, gen_loss={gen_epoch_loss / (step + 1):.5f},  disc_loss={disc_epoch_loss / (step + 1):.5f}")
        wandb.log({ f"{wandb_prefix}/epoch": epoch, 
                    f"{wandb_prefix}/recons_loss": epoch_loss / (step + 1),
                    f"{wandb_prefix}/gen_loss": gen_epoch_loss / (step + 1),
                    f"{wandb_prefix}/disc_loss": disc_epoch_loss / (step + 1), 
                })

        if epoch + 1 in (np.round(np.arange(0.0, 1.01, evaluation_intervall) * n_epochs)):
            images = batch["image"][0, 0].detach().cpu().numpy()
            autoencoder.eval()
            r_img = reconstruction[0, 0].detach().cpu().numpy()
            wandb.log({f"{wandb_prefix}/max_r_intensity": r_img.max(), f"{wandb_prefix}/min_r_intensity": r_img.min()})
            log_image_to_wandb(images, r_img, f"{wandb_prefix}/reconstruction", WANDB_LOG_IMAGES, conditioning_information=batch["volume"][0].detach().cpu())
            
            if val_loader is not None:
                with Stopwatch("Validation took: "):
                    evaluate_autoencoder(val_loader, autoencoder)


    # clean up data
    optimizer_g.zero_grad(set_to_none=True)
    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()

    train_stopwatch.stop().display()

    return autoencoder

def train_diffusion_model(autoencoder: AutoencoderKL, unet: DiffusionModelUNet,
                          optimizer_diff: torch.optim.Optimizer, scaler: GradScaler,
                          inferer: LatentDiffusionInferer,
                          train_loader: DataLoader, valid_loader: DataLoader,
                          encoding_shape: torch.Size,
                          n_epochs: int, 
                          evaluation_intervall: float):
    
    wandb.define_metric("diffusion/training/epoch")
    wandb.define_metric("diffusion/training/*", step_metric="diffusion/training/epoch")

    for epoch in range(n_epochs):
        autoencoder.eval()
        unet.train()
        
        epoch_loss = 0

        for step, batch in enumerate(train_loader):
            images = batch["image"].to(device)
            volume = batch["volume"].to(device)
            optimizer_diff.zero_grad(set_to_none=True)

            with autocast(enabled=True):
                # Generate random noise
                noise = torch.randn(encoding_shape).to(device)

                # Create timesteps
                timesteps = torch.randint(
                    0, inferer.scheduler.num_train_timesteps, (images.shape[0],), device=images.device
                ).long()

                # Get model prediction
                noise_pred = inferer(
                    inputs=images, autoencoder_model=autoencoder, diffusion_model=unet, noise=noise, timesteps=timesteps, condition=volume
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
                                         val_loader=valid_loader)