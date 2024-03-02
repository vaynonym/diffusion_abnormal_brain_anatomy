from src.util import device, Stopwatch
from src.logging_util import LOGGER
import torch
import wandb
from generative.metrics import FIDMetric, MMDMetric, MultiScaleSSIMMetric, SSIMMetric
from generative.inferers import LatentDiffusionInferer
from monai.networks.nets.resnet import resnet50
import torch.nn as nn
import os.path
from collections import OrderedDict
import torch.nn.functional as F


def evaluate_autoencoder(val_loader, autoencoder):
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
        wandb.log(
            {
                f"autoencoder/{name}": (torch.cat(accumulator, dim=0).mean().item()) for (name, accumulator) in zip(metrics.keys(), accumulations)
            }
        )
    
    return None


############### Diffusion Model Evaluation #####################

def evaluate_diffusion_model(diffusion_model, scheduler, autoencoder, latent_shape, inferer: LatentDiffusionInferer, val_loader, model_path):
    assert os.path.exists(model_path), "model_path needs to exist"
    assert os.path.isdir(model_path), "model_path needs to be a directory"
    weights_path = os.path.join(model_path, "resnet_50_23dataset.pth")
    assert os.path.exists(weights_path)

    feature_extraction_model = resnet50(pretrained=False, shortcut_type="B", feed_forward=False, bias_downsample=False, n_input_channels=1, spatial_dims=3).to(device)
    feature_extraction_model.eval()

    loaded = torch.load(weights_path)
    state_dict = loaded["state_dict"]
    state_dict_without_data_parallel = OrderedDict()
    prefix_len = len("module.")
    for k, v in state_dict.items():
        name = k[prefix_len:] # remove "module." prefix
        state_dict_without_data_parallel[name] = v

    feature_extraction_model.load_state_dict(state_dict_without_data_parallel)

    synth_features = []
    real_features = []

    number_images_to_consider = 20

    def medicalNetNormalize(img):
        return (img - img.mean(dim=(1,2,3,4), keepdim=True)) / img.std(dim=(1,2,3,4), keepdim=True)
    
    losses = []

    for step, x in enumerate(val_loader):
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
            real_images = medicalNetNormalize(real_images)
            # Get the features for the real data
            real_eval_feats : torch.Tensor = feature_extraction_model(real_images)
            real_features.append(real_eval_feats)

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

    wandb.log({"training/FID": fid_res.item(),
               "training/valid_loss": mean_loss.item()})
    
    print(f"FID Score: {fid_res.item():.4f}")