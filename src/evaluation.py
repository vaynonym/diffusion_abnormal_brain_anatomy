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
from torch import Tensor

import os.path
from collections import OrderedDict
import torch.nn.functional as F
import torchvision
from typing import Tuple, Optional

from src.directory_management import PRETRAINED_MODEL_DIRECTORY
from src.torch_setup import device
from src.util import log_image_with_mask, log_image_to_wandb
from src.synthseg_masks import decode_one_hot, encode_one_hot
from src.custom_autoencoders import IAutoencoder

class AutoencoderEvaluator():

    def __init__(self,
                 autoencoder: IAutoencoder,
                 val_loader: Optional[DataLoader],
                 wandb_prefix: str,
                 ) -> None:
        self.autoencoder = autoencoder
        self.wandb_prefix = wandb_prefix
        self.val_loader = val_loader
  
    def prepare_input_for_logging(self, input):
        return torch.clamp(input, 0, 1)
    
    def get_input_from_batch(self, batch) -> Tuple[Tensor, ...]:
        return (batch["image"].to(device),)
    
    # this is important to enable different input and output encodings
    # gets model_input to avoid using memory twice for the same tensor
    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return model_input[0]
    
    def visualize_batch(self, batch):
        with torch.no_grad():
            self.autoencoder.eval()
            model_input = self.get_input_from_batch(batch)  # choose only one of Brats channels
            ground_truth = self.get_input_for_evaluation_from_batch(model_input, batch)

            reconstruction = self.autoencoder.reconstruct(*model_input)
            conditioning=batch["volume"][0].detach().cpu()
            
            wandb.log({f"{self.wandb_prefix}/max_r_intensity": reconstruction.max().item(), f"{self.wandb_prefix}/min_r_intensity": reconstruction.min().item()})

            reconstruction_for_logging = self.prepare_input_for_logging(reconstruction)[0, 0].detach().cpu().numpy()
            image_for_logging = self.prepare_input_for_logging(ground_truth)[0, 0].detach().cpu().numpy()

            log_image_to_wandb(image_for_logging, reconstruction_for_logging, f"{self.wandb_prefix}/reconstruction", True,
                        conditioning_information=conditioning)

    def visualize_batches(self, count):
        for i, batch in enumerate(self.val_loader):
            self.visualize_batch(batch)
            if i >= count:
                break

    def evaluate(self):
        if self.val_loader is None:
            print("Cannot evaluate without validation dataloader")
            return

        self.autoencoder.eval()
        
        metrics = { 
            "MS-SSIM": MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4),
            "SSIM": SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4),
            "MMD": MMDMetric(),
        }
        
        accumulations = [[] for _ in metrics.keys()]


        for _, batch in enumerate(self.val_loader):
            model_input = self.get_input_from_batch(batch)
            ground_truth = self.get_input_for_evaluation_from_batch(model_input, batch)

            with torch.no_grad():
                reconstruction = self.autoencoder.reconstruct(*model_input)

                for metric, accumulator in zip(metrics.values(), accumulations):
                    res = metric(ground_truth, reconstruction)
                    if(len(res.size()) == 0):
                        res = res.unsqueeze(0)
                    accumulator.append(res)

        with torch.no_grad():
            wandb.log(
                {
                    f"{self.wandb_prefix}/{name}": (torch.cat(accumulator, dim=0).mean().item()) for (name, accumulator) in zip(metrics.keys(), accumulations)
                }
            )
        
        return None

class MaskAutoencoderEvaluator(AutoencoderEvaluator):
    def __init__(self, 
                 autoencoder: AutoencoderKL,
                 val_loader: Optional[DataLoader],
                 wandb_prefix: str,
                 ) -> None:
        super().__init__(autoencoder, val_loader, wandb_prefix)

    def prepare_input_for_logging(self, input):
        return decode_one_hot(input)
    
    def get_input_from_batch(self, batch) -> Tuple[Tensor, ...]:
        mask = encode_one_hot(batch["mask"].to(device))
        return (mask,)

    def visualize_batch(self, batch):
        # select first sample from last training batch as example evaluation
        with torch.no_grad():
            self.autoencoder.eval()
            image = super().prepare_input_for_logging(batch["image"].to(device))[0, 0].detach().cpu().numpy()
            original_mask = batch["mask"][0, 0].detach().cpu().numpy()
            conditioning=batch["volume"][0].detach().cpu()

            model_input = self.get_input_from_batch(batch)
            reconstruction = self.autoencoder.reconstruct(*model_input)
            print(reconstruction.shape)
            print
            
            # log intensity values
            wandb.log({f"{self.wandb_prefix}/max_r_intensity": reconstruction.max().item(),
                        f"{self.wandb_prefix}/min_r_intensity": reconstruction.min().item()})

            reconstructed_mask_decoded = self.prepare_input_for_logging(reconstruction)[0, 0].detach().cpu().numpy()

            log_image_with_mask(image=image, 
                                original_mask=original_mask,
                                reconstruction_image=image,
                                reconstruction_mask=reconstructed_mask_decoded,
                                description_prefix=f"{self.wandb_prefix}/masks_reconstruction",
                                conditioning_information=conditioning) 


class SegmentationMaskAutoencoderEvaluator(MaskAutoencoderEvaluator):
    def __init__(self, autoencoder: AutoencoderKL, val_loader: Optional[DataLoader], wandb_prefix: str) -> None:
        super().__init__(autoencoder, val_loader, wandb_prefix)
    
    def get_input_from_batch(self, batch) -> Tuple[Tensor, None]:
        return (batch["mask"].int().to(device), None)

    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        mask = encode_one_hot(model_input[0])
        return (mask)

class SpadeAutoencoderEvaluator(AutoencoderEvaluator):
    def __init__(self, autoencoder: IAutoencoder, val_loader: Optional[DataLoader], wandb_prefix: str) -> None:
        super().__init__(autoencoder, val_loader, wandb_prefix)
    
    def get_input_from_batch(self, batch) -> Tuple[Tensor, ...]:
        return (batch["image"].to(device), encode_one_hot(batch["mask"].to(device)))
    
    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return model_input[0]
    


############### Diffusion Model Evaluation #####################

def evaluate_diffusion_model(diffusion_model: nn.Module,
                             scheduler,
                             autoencoder: AutoencoderKL,
                             latent_shape: Tensor,
                             inferer: LatentDiffusionInferer,
                             val_loader: DataLoader,
                             get_input_from_batch = lambda batch: (batch["image"].to(device), batch["volume"].to(device))
                             ):
    weights_path = os.path.join(PRETRAINED_MODEL_DIRECTORY, "resnet_50.pth")
    assert os.path.exists(weights_path)

    diffusion_model.eval()

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
        for i, batch in enumerate(val_loader):
            real_images, _ = get_input_from_batch(batch)
            with torch.no_grad():
                normalized = medicalNetNormalize(real_images)
                if i < 2:
                    log_image_to_wandb(real_images[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, None)
                    log_image_to_wandb(real_images[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, None)

                    log_image_to_wandb(normalized[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, None)
                    log_image_to_wandb(normalized[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, None)

                real_eval_feats : Tensor = feature_extraction_model(normalized)
                real_features.append(real_eval_feats)

    # calculate validation loss for sample
    # get latent representations of synthetic images (for FID)
    number_images_to_consider = 20
    for _, batch in enumerate(val_loader):
        # Get the real images
        real_images, conditioning = get_input_from_batch(batch)

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