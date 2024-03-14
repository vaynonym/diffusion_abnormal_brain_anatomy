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
from itertools import combinations
from torch.utils.data import RandomSampler, ConcatDataset



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
            LOGGER.warn("Cannot evaluate without validation dataloader")
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
                 autoencoder: IAutoencoder,
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
    def __init__(self, autoencoder: IAutoencoder, val_loader: Optional[DataLoader], wandb_prefix: str) -> None:
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

class DiffusionModelEvaluator():
    def __init__(self, 
                 diffusion_model: nn.Module,
                 scheduler,
                 autoencoder: IAutoencoder,
                 latent_shape: torch.Size,
                 inferer: LatentDiffusionInferer,
                 val_loader: DataLoader,
                 train_loader: DataLoader,
                 wandb_prefix: str) -> None:
        self.diffusion_model = diffusion_model
        self.scheduler = scheduler
        self.autoencoder = autoencoder
        self.latent_shape = latent_shape
        self.inferer = inferer
        self.val_loader = val_loader
        self.train_loader = train_loader

        self.wandb_prefix = wandb_prefix

        self.load_feature_extraction_model()
    
    def load_feature_extraction_model(self):
        weights_path = os.path.join(PRETRAINED_MODEL_DIRECTORY, "resnet_50.pth")
        assert os.path.exists(weights_path)

        self.feature_extraction_model = resnet50(pretrained=False,
                                            shortcut_type="B",
                                            feed_forward=False,
                                            bias_downsample=False,
                                            n_input_channels=1,
                                            spatial_dims=3
                                            ).to(device)
        
        # weights are from a model wrapped in torch.nn.DataParallel
        # so we need to unwrap the weights to load them
        loaded = torch.load(weights_path)
        state_dict = loaded["state_dict"]
        state_dict_without_data_parallel = OrderedDict()
        prefix_len = len("module.")
        for k, v in state_dict.items():
            name = k[prefix_len:] # remove "module." prefix
            state_dict_without_data_parallel[name] = v

        self.feature_extraction_model.load_state_dict(state_dict_without_data_parallel)
    
    def medicalNetNormalize(self, img):
        return nn.functional.interpolate(
            (img - img.mean(dim=(1,2,3,4), keepdim=True)) / img.std(dim=(1,2,3,4), keepdim=True),
            size=(125, 125, 125), mode="trilinear"
            )
    
    def get_input_from_batch(self, batch):
        return (batch["image"].to(device), batch["volume"].to(device))
    
    def get_real_features(self, dataloader: DataLoader):
        real_features = []
        for i, batch in enumerate(dataloader):
            real_images, _ = self.get_input_from_batch(batch)
            with torch.no_grad():
                
                normalized = self.medicalNetNormalize(torch.clamp(real_images, 0., 1.))
                #if i < 2:
                #    log_image_to_wandb(real_images[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, None)
                #    log_image_to_wandb(real_images[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, None)
                #
                #    log_image_to_wandb(normalized[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, None)
                #    log_image_to_wandb(normalized[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, None)

                real_eval_feats : Tensor = self.feature_extraction_model(normalized)
                real_features.append(real_eval_feats)
        
        return torch.vstack(real_features)
    
    def get_additional_input_from_batch(self, batch) -> dict:
        return {"conditioning": batch["volume"].to(device)}
    
    def get_synthetic_images(self, batch):
        latent_noise = torch.randn(self.latent_shape).to(device)    
        additional_inputs = self.get_additional_input_from_batch(batch)
        return self.inferer.sample(input_noise=latent_noise,
                                   autoencoder_model=self.autoencoder,
                                   diffusion_model=self.diffusion_model,
                                   scheduler=self.scheduler, 
                                   **additional_inputs)
    
    def get_synthetic_features(self, count):
        batch_size = self.latent_shape[0]
        synth_features = []

        for i, batch in enumerate(self.train_loader):
            # Get the real images
            self.scheduler.set_timesteps(num_inference_steps=1000)

            # Generate some synthetic images using the defined model
            synthetic_images=self.get_synthetic_images(batch)

            with torch.no_grad():
                synthetic_images = self.medicalNetNormalize(torch.clamp(synthetic_images, 0., 1.))

                # Get the features for the synthetic data
                # Generating images far dominates the time to extract features
                synth_eval_feats = self.feature_extraction_model(synthetic_images)
                synth_features.append(synth_eval_feats)
            
            if (i + 1) * batch_size >= count:
                break

        return torch.vstack(synth_features)
    

    def calculate_diversity_metrics(self, count):

        batch_size_multiplier = 3

        combined_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset])
        combined_dataloader = DataLoader(
                   dataset=combined_dataset,
                   batch_size=self.train_loader.batch_size * batch_size_multiplier,
                   shuffle=False,
                   drop_last=self.train_loader.drop_last, 
                   num_workers=self.train_loader.num_workers,
                   persistent_workers=self.train_loader.persistent_workers,
                   sampler=RandomSampler(combined_dataset, replacement=True, num_samples=count)
                   )
        
        old_latent_shape = self.latent_shape
        # temporarily double batch size
        self.latent_shape = torch.Size([self.latent_shape[0] * batch_size_multiplier, *self.latent_shape[1:]])

        metrics = [
        ("MS-SSIM", MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4), []),
        ("SSIM", SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4), []),
        ("MMD", MMDMetric(), []),
        ]

        current_count = 0
        for i, batch in enumerate(combined_dataloader):
            # Get the real images
            self.scheduler.set_timesteps(num_inference_steps=1000)

            # Generate some synthetic images using the defined model
            synthetic_images=self.get_synthetic_images(batch)
            
            combinations_in_batch = list(combinations(range(self.latent_shape[0]), 2))
            for idx_a, idx_b in combinations_in_batch:
                for _, metric, scores in metrics:
                    res = metric(synthetic_images[idx_a].unsqueeze(0), 
                                 synthetic_images[idx_b].unsqueeze(0))
                    if(len(res.size()) == 0):
                        res = res.unsqueeze(0)
                    scores.append(res)
                
                current_count += 1

                if current_count >= count:
                    break
            
            if current_count >= count:
                break
        
        LOGGER.info(f"Number of batches for diversity metrics: { i + 1}")

        results = {
            f"{self.wandb_prefix}/{name}": torch.cat(scores, dim=0).mean().item() for name, _, scores in metrics
        }

        # return batch size to original value
        self.latent_shape = old_latent_shape

        return results


    
    # fid is calculated on train dataset
    def calculate_fid(self):
        real_features = self.get_real_features(self.train_loader)
        synthetic_features = self.get_synthetic_features(4)

        fid = FIDMetric()(synthetic_features, real_features)
        return fid
    
    def evaluate(self):
        with torch.no_grad():
            self.autoencoder.eval()
            self.diffusion_model.eval()
            fid = self.calculate_fid()
            results = {f"{self.wandb_prefix}/FID": fid.item()}
            diversity_metrics = self.calculate_diversity_metrics(15)

        results.update(diversity_metrics)
        wandb.log(results)
    
    def log_samples(self, count):
        self.autoencoder.eval()
        self.diffusion_model.eval()

        batch_size = self.latent_shape[0]
        for i, batch in enumerate(self.val_loader):
            self.scheduler.set_timesteps(num_inference_steps=1000)

            synthetic_images = self.get_synthetic_images(batch)
            additional_inputs = self.get_additional_input_from_batch(batch)

            # ### Visualise synthetic data
            for batch_index in range(batch_size):
                img = synthetic_images[batch_index, 0].detach().cpu().numpy()  # images

                if "conditioning" in additional_inputs.keys():
                    conditioning = additional_inputs["conditioning"]
                else:
                    conditioning = None
                
                log_image_to_wandb(img, None, f"{self.wandb_prefix}/sample_images", True, conditioning[batch_index, 0].detach().cpu())

                if (i + 1) * batch_size + (batch_index + 1) >= count:
                    break
            
            if (i + 1) * batch_size >= count:
                break

class SpadeDiffusionModelEvaluator(DiffusionModelEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def get_additional_input_from_batch(self, batch) -> dict:
        return {"seg": encode_one_hot(batch["mask"].to(device))}
    
    def log_samples(self, count):
        self.autoencoder.eval()
        self.diffusion_model.eval()

        batch_size = self.latent_shape[0]
        for i, batch in enumerate(self.val_loader):
            input_noise =  torch.randn(self.latent_shape).to(device)
            additional_inputs = self.get_additional_input_from_batch(batch)

            self.scheduler.set_timesteps(num_inference_steps=1000)
            
            synthetic_images = self.inferer.sample(
                                            input_noise=input_noise,
                                            autoencoder_model=self.autoencoder,
                                            diffusion_model=self.diffusion_model,
                                            scheduler=self.scheduler,
                                            **additional_inputs
                                            )

            # ### Visualise synthetic data
            for batch_index in range(batch_size):
                synth_image = synthetic_images[batch_index, 0].detach().cpu().numpy()  # images
                original_image = batch["image"][batch_index, 0].detach().cpu().numpy()
                mask = batch["mask"][batch_index, 0].detach().cpu().numpy()
                
                log_image_with_mask(image=original_image, 
                                    original_mask=mask, 
                                    reconstruction_image=synth_image,
                                    reconstruction_mask=mask,
                                    description_prefix=f"{self.wandb_prefix}/sample_images",
                                    conditioning_information=None)
                
                if (i + 1) * batch_size + (batch_index + 1) >= count:
                    break
                
            
            if (i + 1) * batch_size >= count:
                break