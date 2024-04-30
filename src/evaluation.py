from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from monai.data import DataLoader
from monai.networks.nets.resnet import resnet50

import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset, TensorDataset
from torch import Tensor
import torch.nn.functional as F
from torchmetrics.classification import Dice

import os.path
from collections import OrderedDict
from typing import Tuple, Optional

from src.directory_management import PRETRAINED_MODEL_DIRECTORY
from src.torch_setup import device
from src.util import log_image_with_mask, log_image_to_wandb, Stopwatch
from src.logging_util import LOGGER
from src.synthseg_masks import decode_one_hot, encode_one_hot, get_crop_to_max_ventricle_shape
from src.custom_autoencoders import IAutoencoder
from src.diffusion import ILatentDiffusionInferer
from itertools import combinations
from torch.nn import L1Loss



class AutoencoderEvaluator():

    def __init__(self,
                 autoencoder: IAutoencoder,
                 val_loader: Optional[DataLoader],
                 wandb_prefix: str,
                 crop_to_ventricles=False,
                 ) -> None:
        self.autoencoder = autoencoder
        self.wandb_prefix = wandb_prefix
        self.val_loader = val_loader

        if crop_to_ventricles:
            self.crop_to_ventricles = get_crop_to_max_ventricle_shape(self.val_loader)
        else:
            self.crop_to_ventricles = None
        
        self.metrics = { 
            "MS-SSIM": MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4),
            "SSIM": SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4),
            "MMD": MMDMetric(),
            "L1-Loss": L1Loss(),
        }
  
    def evaluation_preprocessing(self, img):
        img = torch.clamp(img, 0, 1)

        if self.crop_to_ventricles is not None:
            img = self.crop_to_ventricles(img)
        
        return img
    
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

            reconstruction_for_logging = self.evaluation_preprocessing(reconstruction)[0, 0].detach().cpu().numpy()
            image_for_logging = self.evaluation_preprocessing(ground_truth)[0, 0].detach().cpu().numpy()

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
        
        accumulations = [[] for _ in self.metrics.keys()]


        for i, batch in enumerate(self.val_loader):
            with torch.no_grad():
                model_input = self.get_input_from_batch(batch)
                ground_truth = self.get_input_for_evaluation_from_batch(model_input, batch)
                ground_truth = self.evaluation_preprocessing(ground_truth)

                reconstruction = self.autoencoder.reconstruct(*model_input)
                reconstruction = self.evaluation_preprocessing(reconstruction)

                for metric, accumulator in zip(self.metrics.values(), accumulations):
                    # clipping images since some metrics assume data range from 0 to 1
                    res = metric(reconstruction, ground_truth)
                    if(len(res.size()) == 0):
                        res = res.unsqueeze(0)
                    accumulator.append(res)

        with torch.no_grad():
            wandb.log(
                {
                    f"{self.wandb_prefix}/{name}": (torch.cat(accumulator, dim=0).mean().item()) for (name, accumulator) in zip(self.metrics.keys(), accumulations)
                }
            )
        
        return None



# Generates Masks

from torch.nn.functional import softmax
from torchmetrics.classification import MulticlassJaccardIndex as IoU
from src.synthseg_masks import synthseg_classes

class MaskAutoencoderEvaluator(AutoencoderEvaluator):
    def __init__(self, 
                 autoencoder: IAutoencoder,
                 val_loader: Optional[DataLoader],
                 wandb_prefix: str,
                 ) -> None:
        super().__init__(autoencoder, val_loader, wandb_prefix)

        self.metric_Dice = Dice(average='micro').to(device)
        self.metric_IoU = IoU(num_classes=len(synthseg_classes), average="micro").to(device)
        self.metric_CrossEntropy = nn.CrossEntropyLoss().to(device)

        self.metrics = { # x is logits, y is one-hot encoded, so most metrics require preprocessing
            # Segmentation-based metrics
            "IoU": lambda x, y: self.metric_IoU(x, y.argmax(dim=1)),
            "Dice": lambda x, y: self.metric_Dice(x, y.argmax(dim=1)),
            "CrossEntropy": lambda x, y: self.metric_CrossEntropy(x, y.argmax(dim=1))
        }

    def evaluation_preprocessing(self, x): 
        if self.crop_to_ventricles is not None:
            x = self.crop_to_ventricles(x)
        
        return x
    
    def get_input_from_batch(self, batch) -> Tuple[Tensor, ...]:
        mask = encode_one_hot(batch["mask"].to(device))
        return (mask,)
    
    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return model_input[0] # one-hot encoded
    
    def visualize_batch(self, batch):
        # select first sample from last training batch as example evaluation
        with torch.no_grad():
            self.autoencoder.eval()
            image = super().evaluation_preprocessing(batch["image"].to(device))[0, 0].detach().cpu().numpy()
            original_mask = batch["mask"][0, 0].detach().cpu().numpy()
            conditioning=batch["volume"][0, 0].detach().cpu()

            model_input = self.get_input_from_batch(batch)
            reconstruction = self.autoencoder.reconstruct(*model_input)

            # log intensity values
            wandb.log({f"{self.wandb_prefix}/max_r_intensity": reconstruction.max().item(),
                        f"{self.wandb_prefix}/min_r_intensity": reconstruction.min().item()})

            reconstructed_mask_decoded = self.evaluation_preprocessing(decode_one_hot(reconstruction))[0, 0].detach().cpu().numpy()

            log_image_with_mask(image=image, 
                                original_mask=original_mask,
                                reconstruction_image=image,
                                reconstruction_mask=reconstructed_mask_decoded,
                                description_prefix=f"{self.wandb_prefix}/masks_reconstruction",
                                conditioning_information=conditioning) 

# Generates Masks
class MaskEmbeddingAutoencoderEvaluator(MaskAutoencoderEvaluator):
    def __init__(self, autoencoder: IAutoencoder, val_loader: Optional[DataLoader], wandb_prefix: str) -> None:
        super().__init__(autoencoder, val_loader, wandb_prefix)
    
    def get_input_from_batch(self, batch) -> Tuple[Tensor, None]:
        return (batch["mask"].int().to(device), )

    def get_input_for_evaluation_from_batch(self, model_input: Tuple[Tensor, ...], batch: dict) -> Tensor:
        return encode_one_hot(model_input[0])

# Generates Images
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
                 evaluation_scheduler,
                 autoencoder: IAutoencoder,
                 latent_shape: torch.Size,
                 inferer: ILatentDiffusionInferer,
                 val_loader: DataLoader,
                 train_loader: DataLoader,
                 wandb_prefix: str,
                 crop_to_ventricles=False,
                 guidance:Optional[float]=None,
                 ) -> None:
        self.diffusion_model = diffusion_model
        self.evaluation_scheduler = evaluation_scheduler
        self.autoencoder = autoencoder
        self.latent_shape = latent_shape
        self.inferer = inferer
        self.val_loader = val_loader
        self.train_loader = train_loader
        self.guidance = guidance

        self.wandb_prefix = wandb_prefix

        if crop_to_ventricles:
            LOGGER.info("Evaluation is cropping to ventricle shape")
            self.crop_to_ventricles = get_crop_to_max_ventricle_shape(self.train_loader)
        else:
            self.crop_to_ventricles = None

        self.diversity_metrics = [
            ("MS-SSIM", MultiScaleSSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4)),
            ("SSIM", SSIMMetric(spatial_dims=3, data_range=1.0, kernel_size=4)),
            ("MMD", MMDMetric()),
        ]

        self.load_feature_extraction_model()
    
    @torch.no_grad()
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
        self.feature_extraction_model.eval()
    
    @torch.no_grad()
    def medicalNetNormalize(self, img):
        return nn.functional.interpolate(
            (img - img.mean(dim=(1,2,3,4), keepdim=True)) / img.std(dim=(1,2,3,4), keepdim=True),
            size=(125, 125, 125), mode="trilinear"
            )
    
    def get_input_from_batch(self, batch: dict) -> dict:
        return  { "inputs": batch["image"].to(device)}
    
    @torch.no_grad()
    def get_real_features(self, dataloader: DataLoader):
        real_features = []
        for i, batch in enumerate(dataloader):
            real_images = self.get_input_from_batch(batch)["inputs"]
            with torch.no_grad():
                
                normalized = self.medicalNetNormalize(self.image_preprocessing(real_images))
                #if i < 2:
                #    log_image_to_wandb(real_images[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, None)
                #    log_image_to_wandb(real_images[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/pristine", True, None, None)
                #
                #    log_image_to_wandb(normalized[0, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, None)
                #    log_image_to_wandb(normalized[1, 0].detach().cpu().numpy(), None, "MedicalnetNormalize/normalized", True, None, None)

                real_eval_feats : Tensor = self.feature_extraction_model(normalized)
                real_features.append(real_eval_feats)
        
        return torch.vstack(real_features)
    
    @torch.no_grad()
    def get_additional_input_from_batch(self, batch) -> dict:
        output = {"conditioning": batch["volume"].to(device)
                }
        if "class" in batch:
            output["class_labels"] = batch["class"].to(device)
        
        return output
    
    @torch.no_grad()
    def get_synthetic_output(self, batch, use_evaluation_scheduler):
        self.autoencoder.eval()
        self.diffusion_model.eval()

        latent_noise = torch.randn(self.latent_shape).to(device)    
        additional_inputs = self.get_additional_input_from_batch(batch)

        scheduler=self.evaluation_scheduler if use_evaluation_scheduler else self.inferer.scheduler 

        if self.guidance is None or self.guidance == 1.0:
            return self.inferer.sample(input_noise=latent_noise,
                                   autoencoder_model=self.autoencoder,
                                   diffusion_model=self.diffusion_model,
                                   scheduler=scheduler,
                                   verbose=False,
                                   **additional_inputs)
        else:
            # remains constant over time, hence we can double here
            additional_inputs = {k: torch.cat([v] * 2)for k, v in additional_inputs.items()}
            # conditional case
            additional_inputs["conditioning"][:self.latent_shape[0], 0, -1] = 1
            # unconditional case
            additional_inputs["conditioning"][self.latent_shape[0]:, 0, -1] = 0
            additional_inputs["conditioning"][self.latent_shape[0]:, 0, 0:-1] = torch.rand(())

            # since we call diffusion model directly, it uses "context" instead of "conditioning"
            additional_inputs["context"] = additional_inputs["conditioning"]
            del additional_inputs["conditioning"]

            for t in scheduler.timesteps:
                # noise changes each timestep, so we double here
                noise_input = torch.cat([latent_noise] * 2)
                model_output = self.diffusion_model(noise_input, timesteps=torch.Tensor((t,)).to(device), **additional_inputs)
                noise_pred_cond, noise_pred_uncond = model_output.chunk(2) # inverse of torch.cat([x] * 2)
                noise_pred = noise_pred_uncond + self.guidance * (noise_pred_cond - noise_pred_uncond)

                latent_noise, _ = scheduler.step(noise_pred, t, latent_noise)

            return self.autoencoder.decode_stage_2_outputs(latent_noise / self.inferer.scale_factor)
    
    @torch.no_grad()
    def image_preprocessing(self, image):
        with torch.no_grad():
            image = torch.clamp(image, 0, 1)
            if self.crop_to_ventricles is not None:
                image = self.crop_to_ventricles(image)
        
        return image
    
    @torch.no_grad()
    def get_synthetic_features(self, count):
        batch_size = self.latent_shape[0]
        synth_features = []

        for i, batch in enumerate(self.val_loader):
            # Get the real images

            # Generate some synthetic images using the defined model
            with torch.no_grad():
                synthetic_images=self.get_synthetic_output(batch, True)

                synthetic_images = self.medicalNetNormalize(self.image_preprocessing(synthetic_images))

                # Get the features for the synthetic data
                # Generating images far dominates the time to extract features
                synth_eval_feats = self.feature_extraction_model(synthetic_images)
                synth_features.append(synth_eval_feats)
            
            if (i + 1) * batch_size >= count:
                break

        return torch.vstack(synth_features)
    
    @torch.no_grad()
    def calculate_diversity_metrics(self, count):

        batch_size_multiplier = 1

        combined_dataset = ConcatDataset([self.train_loader.dataset, self.val_loader.dataset])
        combined_dataloader = DataLoader(
                   dataset=combined_dataset,
                   batch_size=self.train_loader.batch_size * batch_size_multiplier,
                   shuffle=False,
                   drop_last=self.train_loader.drop_last, 
                   num_workers=1,
                   persistent_workers=self.train_loader.persistent_workers,
                   sampler=RandomSampler(combined_dataset, replacement=True, num_samples=count,)
                   )
        
        old_latent_shape = self.latent_shape
        # temporarily double batch size
        self.latent_shape = torch.Size([self.latent_shape[0] * batch_size_multiplier, *self.latent_shape[1:]])

        diversity_metrics = list(zip(self.diversity_metrics, [[]] * len(self.diversity_metrics)))

        current_count = 0
        for i, batch in enumerate(combined_dataloader):
            # Get the real images
            # Generate some synthetic images using the defined model
            synthetic_images=self.get_synthetic_output(batch, True)
            
            combinations_in_batch = list(combinations(range(self.latent_shape[0]), 2))
            for idx_a, idx_b in combinations_in_batch:
                for (_, metric), scores in diversity_metrics:
                    res = metric(self.image_preprocessing(synthetic_images[idx_a].unsqueeze(0)), 
                                 self.image_preprocessing(synthetic_images[idx_b].unsqueeze(0)))
                    if(len(res.size()) == 0):
                        res = res.unsqueeze(0)
                    if(len(res.size()) == 1):
                        res = res.unsqueeze(0)
                    scores.append(res)
                
                current_count += 1

                if current_count >= count:
                    break
            
            if current_count >= count:
                break
        
        LOGGER.info(f"Number of batches for diversity metrics: { i + 1}")

        results = {
            f"{self.wandb_prefix}/{name}": torch.cat(scores, dim=0).mean().item() for (name, _), scores in diversity_metrics
        }

        # return batch size to original value
        self.latent_shape = old_latent_shape

        return results

    # fid is calculated on train dataset
    @torch.no_grad()
    def calculate_fid(self):
        real_features = self.get_real_features(self.val_loader)
        synthetic_features = self.get_synthetic_features(100)

        fid = FIDMetric()(synthetic_features, real_features)
        return fid
    
    @torch.no_grad()
    def evaluate(self):
        self.autoencoder.eval()
        self.diffusion_model.eval()
        with Stopwatch("FID took: "):
            fid = self.calculate_fid()
        results = {f"{self.wandb_prefix}/FID": fid.item()}
        with Stopwatch("Diversity metrics took: "):
            diversity_metrics = self.calculate_diversity_metrics(100)

        results.update(diversity_metrics)
        wandb.log(results)
    
    @torch.no_grad()
    def log_samples(self, count, use_evaluation_scheduler=False):
        self.autoencoder.eval()
        self.diffusion_model.eval()

        # don't generate more images than necessary
        old_latent_shape = self.latent_shape
        self.latent_shape = torch.Size([min(self.latent_shape[0], count) , *self.latent_shape[1:]])
        
        batch_size = self.latent_shape[0]
        for i, batch in enumerate(self.val_loader):
            reduced_batch = {k:v[:batch_size] for k, v in batch.items()}

            synthetic_images = self.get_synthetic_output(reduced_batch, use_evaluation_scheduler)
            additional_inputs = self.get_additional_input_from_batch(reduced_batch)

            # ### Visualise synthetic data
            for batch_index in range(batch_size):
                img = self.image_preprocessing(synthetic_images[batch_index, 0]).detach().cpu().numpy()  # images

                if "conditioning" in additional_inputs.keys():
                    conditioning = additional_inputs["conditioning"]
                else:
                    conditioning = None
                
                
                
                log_image_to_wandb(img, None, f"{self.wandb_prefix}/sample_images", True, conditioning[batch_index, 0].detach().cpu())

                if i * batch_size + (batch_index + 1) >= count:
                    break
            
            if (i + 1) * batch_size >= count:
                break
        
        self.latent_shape = old_latent_shape

class MaskDiffusionModelEvaluator(DiffusionModelEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        

        self.metric_Dice = Dice(average='micro').to(device)
        self.metric_IoU = IoU(num_classes=len(synthseg_classes), average="micro").to(device)
        self.metric_CrossEntropy = nn.CrossEntropyLoss().to(device)

        self.metrics = { # x is logits, y [0...C) values
            "IoU": lambda x, y: self.metric_IoU(x, y),
            "Dice": lambda x, y: self.metric_Dice(x, y),
            "CrossEntropy": lambda x, y: self.metric_CrossEntropy(x, y)
        }

        self.diversity_metrics = {
            ("Pairwise IoU", lambda x, y: self.metric_IoU(x.argmax(dim=1), y.argmax(dim=1))),
            ("Pairwise Dice", lambda x, y: self.metric_Dice(x.argmax(dim=1), y.argmax(dim=1))),
        }

    def get_input_from_batch(self, batch: dict) -> dict:
        return  { "inputs": batch["mask"].int().to(device)}

    def image_preprocessing(self, image):
        with torch.no_grad():
            if self.crop_to_ventricles is not None:
                image = self.crop_to_ventricles(image)
        
        return image
    
    def get_synthetic_features(self, count):
        pass

    def get_real_features(self, dataloader: DataLoader):
        pass

    def calculate_fid(self):
        pass

    def calculate_reconstruction_metrics(self):

        accumulations = [[] for _ in self.metrics.keys()]


        for i, batch in enumerate(self.val_loader):
            with torch.no_grad():
                # convert to [0...C) indices
                ground_truth = self.image_preprocessing(encode_one_hot(batch["mask"].to(device))).argmax(dim=1)
                reconstruction = self.image_preprocessing(self.get_synthetic_output(batch, True))

                for metric, accumulator in zip(self.metrics.values(), accumulations):
                    # clipping images since some metrics assume data range from 0 to 1
                    res = metric(reconstruction, ground_truth)
                    if(len(res.size()) == 0):
                        res = res.unsqueeze(0)
                    accumulator.append(res)

        with torch.no_grad():
            results = {
                    f"{self.wandb_prefix}/{name}": (torch.cat(accumulator, dim=0).mean().item()) for (name, accumulator) in zip(self.metrics.keys(), accumulations)
                }
        
        return results


    def evaluate(self):
        with torch.no_grad():
            self.autoencoder.eval()
            self.diffusion_model.eval()
            results = {}
            with Stopwatch("Reconstruction  metrics: "):
                reconstruction_metrics = self.calculate_reconstruction_metrics()
                results.update(reconstruction_metrics)
            with Stopwatch("Diversity metrics took: "):
                diversity_metrics = self.calculate_diversity_metrics(100)
                results.update(diversity_metrics)

        wandb.log(results)


    def log_samples(self, count, use_evaluation_scheduler=False):
        self.autoencoder.eval()
        self.diffusion_model.eval()

        # don't generate more images than necessary
        old_latent_shape = self.latent_shape
        self.latent_shape = torch.Size([min(self.latent_shape[0], count) , *self.latent_shape[1:]])

        batch_size = self.latent_shape[0]
        for i, batch in enumerate(self.val_loader):
            reduced_batch = {k:v[:batch_size] for k, v in batch.items()}

            synthetic_output = decode_one_hot(self.get_synthetic_output(reduced_batch, use_evaluation_scheduler))

            # ### Visualise synthetic data
            for batch_index in range(batch_size):
                synth_mask = self.image_preprocessing(synthetic_output)[batch_index, 0].detach().cpu().numpy()  # images
                original_image = super().image_preprocessing(reduced_batch["image"])[batch_index, 0].detach().cpu().numpy()
                ground_truth_mask = self.image_preprocessing(reduced_batch["mask"])[batch_index, 0].detach().cpu().numpy()

                if "volume" in reduced_batch.keys():
                    volume = reduced_batch["volume"][batch_index, 0].detach().cpu()
                
                log_image_with_mask(image=original_image, 
                                    original_mask=ground_truth_mask, 
                                    reconstruction_image=original_image,
                                    reconstruction_mask=synth_mask,
                                    description_prefix=f"{self.wandb_prefix}/sample_images",
                                    conditioning_information=volume)
                
                if i * batch_size + (batch_index + 1) >= count:
                    break
                
            
            if (i + 1) * batch_size >= count:
                break
        
        self.latent_shape = old_latent_shape


class SpadeDiffusionModelEvaluator(DiffusionModelEvaluator):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    @torch.no_grad()
    def get_additional_input_from_batch(self, batch) -> dict:
        output = {"seg": encode_one_hot(batch["mask"].to(device))}

        if "class" in batch:
            output["class_labels"] = batch["class"].to(device)
        
        return output
    
    @torch.no_grad()
    def mask_preprocessing(self, image):
        if self.crop_to_ventricles is not None:
            image = self.crop_to_ventricles(image)
        
        return image

    @torch.no_grad()
    def log_samples(self, count, use_evaluation_scheduler=False):
        self.autoencoder.eval()
        self.diffusion_model.eval()

        # don't generate more images than necessary
        old_latent_shape = self.latent_shape
        self.latent_shape = torch.Size([min(self.latent_shape[0], count) , *self.latent_shape[1:]])

        batch_size = self.latent_shape[0]
        for i, batch in enumerate(self.val_loader):
            reduced_batch = {k:v[:batch_size] for k, v in batch.items()}

            synthetic_images = self.get_synthetic_output(reduced_batch, use_evaluation_scheduler)

            # ### Visualise synthetic data
            for batch_index in range(batch_size):
                synth_image = self.image_preprocessing(synthetic_images[batch_index, 0]).detach().cpu().numpy()  # images
                if "image" in reduced_batch:
                    original_image = self.image_preprocessing(reduced_batch["image"][batch_index, 0]).detach().cpu().numpy()
                else:
                    original_image = torch.zeros_like(synthetic_images[batch_index, 0]).detach().cpu().numpy()

                mask = self.mask_preprocessing(reduced_batch["mask"][batch_index, 0]).detach().cpu().numpy()

                if "volume" in reduced_batch.keys():
                    volume = reduced_batch["volume"][batch_index, 0].detach().cpu()

                class_label = None
                if "class_labels" in batch.keys():
                    class_label = reduced_batch["class_labels"][batch_index, 0].detach().cpu()
                
                log_image_with_mask(image=original_image, 
                                    original_mask=mask, 
                                    reconstruction_image=synth_image,
                                    reconstruction_mask=mask,
                                    description_prefix=f"{self.wandb_prefix}/sample_images",
                                    conditioning_information=volume,
                                    class_label=class_label
                                    )
                
                if i * batch_size + (batch_index + 1) >= count:
                    break
            if (i + 1) * batch_size >= count:
                break

        self.latent_shape = old_latent_shape


def create_fake_volume_dataloader(min_val, max_val, bucket_size, number_samples_per_bucket, classifier_free_guidance, batch_size, image_shape: Optional[torch.Size] = None):
    
    dataset = torch.arange(min_val, max_val, bucket_size).unsqueeze(0).repeat(number_samples_per_bucket, 1).flatten().unsqueeze(1).unsqueeze(2)

    def collate(data):
        volume = torch.stack([x[0] for x in data])
        d = {"volume": volume}
        if image_shape is not None:
            d["image"] = torch.zeros(image_shape), 
            d["mask"] = torch.zeros(image_shape)
        return d

    def collate_guidance(data):
        volume = torch.stack([x[0] for x in data])
        volume = torch.concat([volume, torch.ones_like(volume)], dim=2)
        d = {"volume": volume}
        if image_shape is not None:
            d["image"] = torch.zeros(image_shape), 
            d["mask"] = torch.zeros(image_shape)
        return d

    fake_volume_dataset = TensorDataset(dataset)

    fake_volume_dataloader = DataLoader(fake_volume_dataset,
                                        batch_size=batch_size,
                                        shuffle=True, num_workers=2,
                                        drop_last=True,
                                        persistent_workers=True,
                                        collate_fn=collate_guidance if classifier_free_guidance is not None else collate)

    return fake_volume_dataloader