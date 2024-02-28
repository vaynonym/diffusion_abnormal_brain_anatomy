import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader 

from torch.nn import L1Loss
from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator

from src.util import device, Stopwatch, log_image_to_wandb
from src.logging_util import LOGGER

def train_autoencoder(autoencoder: AutoencoderKL, train_loader: DataLoader, patch_discrim_config: dict, auto_encoder_training_config: dict, run_config: dict, WANDB_LOG_IMAGES: bool):
    train_stopwatch = Stopwatch("Autoencoder training time: ").start()
    
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

    optimizer_g = torch.optim.Adam(params=autoencoder.parameters(), lr=1e-4)
    optimizer_d = torch.optim.Adam(params=discriminator.parameters(), lr=1e-4)

    # ### Train model

    # +
    n_epochs = auto_encoder_training_config["n_epochs"]
    autoencoder_warm_up_n_epochs = auto_encoder_training_config["autoencoder_warm_up_n_epochs"]

    wandb.define_metric("autoencoder/epoch")
    wandb.define_metric("autoencoder/*", step_metric="autoencoder/epoch")
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

            if epoch > autoencoder_warm_up_n_epochs:
                logits_fake = discriminator(reconstruction.contiguous().float())[-1]
                generator_loss = adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
                loss_g += adv_weight * generator_loss

            loss_g.backward()
            optimizer_g.step()

            if epoch > autoencoder_warm_up_n_epochs:
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
            if epoch > autoencoder_warm_up_n_epochs:
                gen_epoch_loss += generator_loss.item()
                disc_epoch_loss += discriminator_loss.item()

        LOGGER.info(f"{epoch} | recons_loss={epoch_loss / (step + 1):.5f}, gen_loss={gen_epoch_loss / (step + 1):.5f},  disc_loss={disc_epoch_loss / (step + 1):.5f}")
        wandb.log({ "autoencoder/epoch": epoch, 
                    "autoencoder/recons_loss": epoch_loss / (step + 1),
                    "autoencoder/gen_loss": gen_epoch_loss / (step + 1),
                    "autoencoder/disc_loss": disc_epoch_loss / (step + 1), 
                })

        if epoch + 1 in (np.round(np.arange(0.0, 1.01, run_config["gen_image_intervall"]) * n_epochs)):
            with torch.no_grad():
                images = batch["image"][0, 0].detach().cpu().numpy()
                autoencoder.eval()
                r_img = reconstruction[0, 0].detach().cpu().numpy()
                log_image_to_wandb(images, r_img, "Visualize Reconstruction", WANDB_LOG_IMAGES, conditioning_information=batch["volume"][0].detach().cpu())


    del discriminator
    del loss_perceptual
    torch.cuda.empty_cache()

    train_stopwatch.stop().display()

    return autoencoder
