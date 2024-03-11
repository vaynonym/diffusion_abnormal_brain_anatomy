import numpy as np
import wandb
import torch
from torch.utils.data import DataLoader 
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.nn import L1Loss
import torch.nn as nn

from generative.losses import PatchAdversarialLoss, PerceptualLoss
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet, PatchDiscriminator
from generative.inferers import LatentDiffusionInferer

from src.util import Stopwatch, log_image_to_wandb, log_image_with_mask
from src.model_util import save_model
from src.logging_util import LOGGER
from src.evaluation import evaluate_diffusion_model
from src.diffusion import generate_and_log_sample_images
from src.torch_setup import device
from src.synthseg_masks import decode_one_hot, encode_one_hot
from src.evaluation import AutoencoderEvaluator, MaskAutoencoderEvaluator
from src.directory_management import MODEL_DIRECTORY
from typing import Tuple
import os

class AutoencoderTrainer():

    def __init__(self,
                 autoencoder: AutoencoderKL,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 patch_discrim_config: dict,
                 auto_encoder_training_config: dict,
                 evaluation_intervall: float,
                 WANDB_LOG_IMAGES: bool,
                 starting_epoch=0,
                 ) -> None:

        self.autoencoder : AutoencoderKL = autoencoder
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.wandb_prefix = "autoencoder/training"
        self.evaluator = AutoencoderEvaluator(self.autoencoder, self.val_loader, self.wandb_prefix)


        self.discriminator =  PatchDiscriminator(**patch_discrim_config).to(device)
        self.l1_loss = L1Loss()
        self.adv_loss = PatchAdversarialLoss(criterion="least_squares")
        self.loss_perceptual = PerceptualLoss(spatial_dims=3, network_type="squeeze", is_fake_3d=True, fake_3d_ratio=0.2).to(device)
        
        self.adv_weight = 0.01
        self.perceptual_weight = 0.001
        self.kl_weight = 1e-6

        self.lr=auto_encoder_training_config["learning_rate"]
        self.optimizer_g = torch.optim.Adam(params=self.autoencoder.parameters(), lr=self.lr)
        self.optimizer_d = torch.optim.Adam(params=self.discriminator.parameters(), lr=self.lr)

        self.n_epochs = auto_encoder_training_config["n_epochs"]
        self.current_epoch = starting_epoch
        assert self.current_epoch <= self.n_epochs
        
        # continue training from checkpoint
        if self.current_epoch != 0:
            self.load_state()

        self.autoencoder_warm_up_n_epochs = auto_encoder_training_config["autoencoder_warm_up_n_epochs"]

        self.WANDB_LOG_IMAGES = WANDB_LOG_IMAGES
        self.evaluation_intervall = evaluation_intervall

    def KL_loss(self, z_mu, z_sigma):
        kl_loss = 0.5 * torch.sum(z_mu.pow(2) + z_sigma.pow(2) - torch.log(z_sigma.pow(2)) - 1, dim=[1, 2, 3, 4])
        return torch.sum(kl_loss) / kl_loss.shape[0]
    
    def get_input_from_batch(self, batch) -> Tuple[torch.Tensor, None]:
        return (batch["image"].to(device), None)    

    def train(self):
        wandb.define_metric(f"{self.wandb_prefix}/epoch")
        wandb.define_metric(f"{self.wandb_prefix}/*", step_metric=f"{self.wandb_prefix}/epoch")

        LOGGER.info(f"Start training {'with' if self.loss_perceptual is not None else 'without'} perceptual loss")
        
        train_stopwatch = Stopwatch("Autoencoder training time: ").start()

        for epoch in range(self.current_epoch, self.n_epochs):
            self.autoencoder.train()
            self.discriminator.train()
            epoch_loss = 0
            gen_epoch_loss = 0
            disc_epoch_loss = 0

            for step, batch in enumerate(self.train_loader):
                step_loss, gen_step_loss, disc_step_loss = self.do_step(batch, epoch)
                epoch_loss += step_loss
                gen_epoch_loss += gen_step_loss
                disc_epoch_loss += disc_step_loss

            self.log_epoch(step, epoch, epoch_loss, gen_epoch_loss, disc_epoch_loss)

            if epoch + 1 in (np.round(np.arange(0.0, 1.01, self.evaluation_intervall) * self.n_epochs)):
                self.evaluator.visualize_batch(batch=batch)
                
                with Stopwatch("Validation took: "):
                    self.evaluator.evaluate()

                with Stopwatch("Saving new state took: "):
                    previously_saved_epoch = self.current_epoch
                    self.current_epoch = epoch + 1
                    self.save_state()
                    self.delete_save(previously_saved_epoch)

        # clean up data
        self.clean_up()

        train_stopwatch.stop().display()

        return self.autoencoder
        
    def save_state(self):

        state = {f"model_{self.autoencoder.__class__.__name__}": self.autoencoder.state_dict(),
                 f"discriminator": self.discriminator.state_dict(),
                 f"optimizer_g": self.optimizer_g.state_dict(),
                 f"optimizer_d": self.optimizer_d.state_dict()
                 }
        torch.save(state, os.path.join(MODEL_DIRECTORY, f"{self.__class__.__name__}_E{self.current_epoch}.pth"))
    
    def delete_save(self, epoch):
        path = os.path.join(MODEL_DIRECTORY, f"{self.__class__.__name__}_E{epoch}.pth")
        if os.path.exists(path):
            os.remove(path)
    
    def load_state(self):
        path = os.path.join(MODEL_DIRECTORY, f"{self.__class__.__name__}_E{self.current_epoch}.pth")
        assert os.path.exists(path), f"Since starting epoch is not zero, expects state to exist at: {path}"

        state = torch.load(path, map_location=device)

        autoencoder_name = f"model_{self.autoencoder.__class__.__name__}"
        assert autoencoder_name in state.keys(), f"Autoencoder with class {self.autoencoder.__class__.__name__} not present in saved state"

        self.autoencoder.load_state_dict(state[autoencoder_name])
        self.discriminator.load_state_dict(state["discriminator"])
        self.optimizer_g.load_state_dict(state["optimizer_g"])
        self.optimizer_d.load_state_dict(state["optimizer_d"])
        
    def do_step(self, batch, epoch):
        (step_loss, gen_step_loss, disc_step_loss) = (0, 0, 0)
        
        images, _ = self.get_input_from_batch(batch)

        # Generator part
        self.optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = self.autoencoder(images)
        

        recons_loss = self.l1_loss(reconstruction.float(), images.float())
        loss_g = recons_loss

        loss_g += self.kl_weight * self.KL_loss(z_mu, z_sigma)

        if self.loss_perceptual is not None:
            loss_g += self.perceptual_weight * self.loss_perceptual(reconstruction.float(), images.float())

        # Adverserial loss determined by the discriminator
        if epoch >= self.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.adv_weight * generator_loss

        loss_g.backward()
        self.optimizer_g.step()

        if epoch >= self.autoencoder_warm_up_n_epochs:
            # Discriminator part
            discriminator_loss = self.discriminator_step(reconstruction, images)


        step_loss += recons_loss.item()
        if epoch >= self.autoencoder_warm_up_n_epochs:
            gen_step_loss += generator_loss.item()
            disc_step_loss += discriminator_loss.item()
        
        return step_loss, gen_step_loss, disc_step_loss
    
    def discriminator_step(self, reconstruction, images):
        self.optimizer_d.zero_grad(set_to_none=True)
        
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(images.contiguous().detach())[-1]
        loss_d_real = self.adv_loss(logits_real, target_is_real=True, for_discriminator=True)
        discriminator_loss = (loss_d_fake + loss_d_real) * 0.5

        loss_d = self.adv_weight * discriminator_loss

        loss_d.backward()
        self.optimizer_d.step()

        return discriminator_loss
    
            
    def log_epoch(self, step, epoch, recon_loss, gen_loss, disc_loss):
        LOGGER.info(f"{epoch} | recons_loss={recon_loss / (step + 1):.5f}," +
                    f"gen_loss={gen_loss / (step + 1):.5f}," +
                    f"disc_loss={disc_loss / (step + 1):.5f}")
        wandb.log({     
                    f"{self.wandb_prefix}/epoch": epoch, 
                    f"{self.wandb_prefix}/recons_loss": recon_loss / (step + 1),
                    f"{self.wandb_prefix}/gen_loss": gen_loss / (step + 1),
                    f"{self.wandb_prefix}/disc_loss": disc_loss / (step + 1), 
                  })

    def clean_up(self):
        self.optimizer_g.zero_grad(set_to_none=True)
        self.optimizer_d.zero_grad(set_to_none=True)

        if self.discriminator is not None:
            del self.discriminator
        if self.loss_perceptual is not None:
            del self.loss_perceptual
        torch.cuda.empty_cache()

class SegmentationAutoencoderTrainer(AutoencoderTrainer):
    def __init__(self, 
                 autoencoder: AutoencoderKL,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 patch_discrim_config: dict,
                 auto_encoder_training_config: dict,
                 evaluation_intervall: float,
                 WANDB_LOG_IMAGES: bool,
                 starting_epoch=0,
                 ) -> None:
        super().__init__(autoencoder,
                        train_loader, val_loader,
                        patch_discrim_config, auto_encoder_training_config,
                        evaluation_intervall, 
                        WANDB_LOG_IMAGES, 
                        starting_epoch)
        
        # Traditional Percpetual loss does not make sense for a segmentation to segmentation task
        self.perceptual_weight = 0
        self.loss_perceptual = None
        self.evaluator = MaskAutoencoderEvaluator(self.autoencoder, self.val_loader, self.wandb_prefix)

    
    def get_input_from_batch(self, batch) -> Tuple[torch.Tensor, None]:
        mask = encode_one_hot(batch["mask"].to(device))
        return (mask, None)
