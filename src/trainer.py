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

from src.util import Stopwatch
from src.logging_util import LOGGER
from src.evaluation import DiffusionModelEvaluator
from src.torch_setup import device
from src.synthseg_masks import decode_one_hot, encode_one_hot
from src.evaluation import AutoencoderEvaluator, MaskAutoencoderEvaluator, SegmentationMaskAutoencoderEvaluator
from src.evaluation import SpadeAutoencoderEvaluator, SpadeDiffusionModelEvaluator
from src.directory_management import MODEL_DIRECTORY
from src.custom_autoencoders import IAutoencoder
from torch import Tensor
from typing import Tuple
import os

class AutoencoderTrainer():

    def __init__(self,
                 autoencoder: IAutoencoder,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 patch_discrim_config: dict,
                 auto_encoder_training_config: dict,
                 evaluation_intervall: float,
                 WANDB_LOG_IMAGES: bool,
                 starting_epoch=0,
                 ) -> None:

        self.autoencoder : IAutoencoder = autoencoder
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
    
    def get_input_from_batch(self, batch: dict) -> Tuple[Tensor, ...]:
        return (batch["image"].to(device),)

    # this is important to enable different input and output encodings as well as support additional network inputs
    # like segmentation masks that are not relevant for losses or evaluation
    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return model_input[0]

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
        
        model_input = self.get_input_from_batch(batch)
        ground_truth = self.get_input_for_evaluation_from_batch(model_input, batch)

        # Generator part
        self.optimizer_g.zero_grad(set_to_none=True)
        reconstruction, z_mu, z_sigma = self.autoencoder(*model_input)
        

        recons_loss = self.l1_loss(reconstruction.float(), ground_truth.float())
        loss_g = recons_loss

        loss_g += self.kl_weight * self.KL_loss(z_mu, z_sigma)

        if self.loss_perceptual is not None:
            loss_g += self.perceptual_weight * self.loss_perceptual(reconstruction.float(), ground_truth.float())

        # Adverserial loss determined by the discriminator
        if epoch >= self.autoencoder_warm_up_n_epochs:
            logits_fake = self.discriminator(reconstruction.contiguous().float())[-1]
            generator_loss = self.adv_loss(logits_fake, target_is_real=True, for_discriminator=False)
            loss_g += self.adv_weight * generator_loss

        loss_g.backward()
        self.optimizer_g.step()

        if epoch >= self.autoencoder_warm_up_n_epochs:
            # Discriminator part
            discriminator_loss = self.discriminator_step(reconstruction, ground_truth)


        step_loss += recons_loss.item()
        if epoch >= self.autoencoder_warm_up_n_epochs:
            gen_step_loss += generator_loss.item()
            disc_step_loss += discriminator_loss.item()
        
        return step_loss, gen_step_loss, disc_step_loss
    
    def discriminator_step(self, reconstruction, images_y):
        self.optimizer_d.zero_grad(set_to_none=True)
        
        logits_fake = self.discriminator(reconstruction.contiguous().detach())[-1]
        loss_d_fake = self.adv_loss(logits_fake, target_is_real=False, for_discriminator=True)
        logits_real = self.discriminator(images_y.contiguous().detach())[-1]
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

    def get_input_from_batch(self, batch) -> Tuple[Tensor, ...]:
        mask = encode_one_hot(batch["mask"].to(device))
        return (mask,)

class SegmentationEmbeddingAutoencoderTrainer(SegmentationAutoencoderTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = SegmentationMaskAutoencoderEvaluator(self.autoencoder, self.val_loader, self.wandb_prefix)

    # since the embedding has to be part of the model, we give the raw mask 
    def get_input_from_batch(self, batch) -> Tuple[Tensor, ...]:
        return (batch["mask"].int().to(device),)
    
    # Our decoder should output one-hot encoding, thus we compute loss and other evaluation calculation
    # using one-hot encoding
    def get_input_for_evaluation_from_batch(self, model_input, batch) -> Tensor:
        mask = encode_one_hot(model_input[0])
        return mask

class SpadeAutoencoderTrainer(AutoencoderTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = SpadeAutoencoderEvaluator(self.autoencoder, self.val_loader, self.wandb_prefix)
    
    def get_input_from_batch(self, batch: dict) -> Tuple[Tensor, ...]:
        return (batch["image"].to(device), encode_one_hot(batch["mask"].to(device)))
    
    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return model_input[0] # use the same image as we used as input to model
    

########## DIFFUSION MODEL ###########

class DiffusionModelTrainer():
    def __init__(self,
                 autoencoder: IAutoencoder,
                 unet: DiffusionModelUNet,
                 optimizer_diff: torch.optim.Optimizer, scaler: GradScaler,
                 inferer: LatentDiffusionInferer,
                 train_loader: DataLoader, valid_loader: DataLoader,
                 encoding_shape: torch.Size,
                 diffusion_model_training_config: dict, 
                 evaluation_intervall: float,
                 starting_epoch=0,
                ) -> None:
        self.wandb_prefix = "diffusion/training"
        wandb.define_metric(f"{self.wandb_prefix}/epoch")
        wandb.define_metric(f"{self.wandb_prefix}/*", step_metric=f"{self.wandb_prefix}/epoch")

        self.train_loader = train_loader
        self.val_loader = valid_loader
        self.encoding_shape = encoding_shape
        self.evaluation_intervall= evaluation_intervall
        self.inferer = inferer

        # stateful
        self.optimizer = optimizer_diff
        self.grad_scaler = scaler
        self.autoencoder = autoencoder
        self.diffusion_model = unet

        self.n_epochs = diffusion_model_training_config["n_epochs"]
        
        self.current_epoch = starting_epoch
        assert self.current_epoch <= self.n_epochs

        
        # continue training from checkpoint
        if self.current_epoch != 0:
            self.load_state()
        
        self.evaluator = DiffusionModelEvaluator(self.diffusion_model, self.inferer.scheduler, self.autoencoder,
                                                 self.encoding_shape, self.inferer, self.val_loader, self.train_loader,
                                                 wandb_prefix=self.wandb_prefix
                                                 )

    
    def get_input_from_batch(self, batch: dict) -> dict:
        return  { "inputs": batch["image"].to(device), 
                  "condition": batch["volume"].to(device)
                }
    
    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return model_input["inputs"]
    
    def train(self):
        train_stopwatch = Stopwatch("Training took: ").start()

        for epoch in range(self.current_epoch, self.n_epochs):
            self.autoencoder.eval()
            self.diffusion_model.train()
            
            epoch_loss = 0

            for step, batch in enumerate(self.train_loader):
                epoch_loss += self.do_step(batch, epoch)
            
            self.log_epoch(step, epoch, epoch_loss)


            if epoch + 1 in (np.round(np.arange(0.0, 1.01, self.evaluation_intervall) * self.n_epochs)):
                with Stopwatch("Sampling example images took: "):
                    self.evaluator.log_samples(1)
                
                with Stopwatch("Evaluation metrics took: "):
                    self.evaluator.evaluate()

                with Stopwatch("Saving new state took: "):
                    previously_saved_epoch = self.current_epoch
                    self.current_epoch = epoch + 1
                    self.save_state()
                    self.delete_save(previously_saved_epoch)

        # clean up data
        self.clean_up()

        train_stopwatch.stop().display()

        return self.diffusion_model


    def do_step(self, batch, epoch):
        inputs = self.get_input_from_batch(batch)
        self.optimizer.zero_grad(set_to_none=True)

        with autocast(enabled=True):
            # Generate random noise
            noise = torch.randn(self.encoding_shape).to(device)
            batch_size = self.encoding_shape[0]
            # Create timesteps
            timesteps = torch.randint(
                0, self.inferer.scheduler.num_train_timesteps, (batch_size,), device=device
            ).long()

            # Get model prediction
            noise_pred = self.inferer(
                autoencoder_model=self.autoencoder, 
                diffusion_model=self.diffusion_model, 
                noise=noise,
                timesteps=timesteps,
                **inputs
            )

            loss = F.mse_loss(noise_pred.float(), noise.float())

        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        return loss.item()

        
    def save_state(self):

        state = {f"model_{self.autoencoder.__class__.__name__}": self.autoencoder.state_dict(),
                 f"model_{self.diffusion_model.__class__.__name__}": self.diffusion_model.state_dict(),
                 f"optimizer": self.optimizer.state_dict(),
                 f"scaler": self.grad_scaler.state_dict()
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

        diffusion_name = f"model_{self.diffusion_model.__class__.__name__}"
        assert diffusion_name in state.keys(), f"Diffusion model with class {self.diffusion_model.__class__.__name__} not present in saved state"

        self.autoencoder.load_state_dict(state[autoencoder_name])
        self.diffusion_model.load_state_dict(state[diffusion_name])
        self.optimizer.load_state_dict(state["optimizer"])
        self.grad_scaler.load_state_dict(state["scaler"])
    
    def log_epoch(self, step, epoch, epoch_loss):
        LOGGER.info(f"{epoch}: loss={epoch_loss / (step + 1):.5f}")
        wandb.log({ 
                    f"{self.wandb_prefix}/epoch": epoch,
                    f"{self.wandb_prefix}/loss": epoch_loss / (step + 1),
                  })
    
    def clean_up(self):
        self.optimizer.zero_grad(set_to_none=True)
        del self.grad_scaler
        del self.optimizer
        torch.cuda.empty_cache()

class SpadeDiffusionModelTrainer(DiffusionModelTrainer):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.evaluator = SpadeDiffusionModelEvaluator(
                            self.diffusion_model, self.inferer.scheduler, self.autoencoder,
                            self.encoding_shape, self.inferer,
                            self.val_loader, self.train_loader,
                            wandb_prefix=self.wandb_prefix
                            )
    
    def get_input_from_batch(self, batch: dict) -> dict:
        return  { 
                  "inputs": batch["image"].to(device), 
                  "seg": encode_one_hot(batch["mask"].to(device))
                }