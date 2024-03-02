import torch
import wandb
import os
from src.logging_util import LOGGER

def save_model(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer=None):

    state = {"model": model.state_dict()}
    if optimizer is not None:
        state["optimizer"] = optimizer.state_dict()
    torch.save(state, path)

def save_model_as_artifact(wandb_run, model: torch.nn.Module, model_name: str, model_config: dict, path:str):
    model_artifact = wandb.Artifact(model_name, type="model", metadata=dict(model_config))
    file_path = os.path.join(path, model_name + ".pth")
    save_model(model, file_path)
    model_artifact.add_file(file_path)
    wandb_run.log_artifact(model_artifact)

def load_model(model: torch.nn.Module, path: str, optimizer: torch.optim.Optimizer=None):
    state = torch.load(path, map_location=torch.device('cpu'))
    model.load_state_dict(state["model"])

    if optimizer is not None:
        optimizer.load_state_dict(state["optimizer"])

def load_model_from_run_with_matching_config(subconfigs, subconfig_names, project, entity, model, artifact_name, models_path):
    
    filters = {
        "$and": [{f"config.{subconfig_name}": subconfig} for subconfig, subconfig_name in zip(subconfigs, subconfig_names)]
            #  + [{"state": "finished"}],
    }
    runs = wandb.Api().runs(path=f"{entity}/{project}", filters=filters)

    if len(runs) == 0:
        LOGGER.info("Did not find any run with matching config")
        return None
    
    LOGGER.info(f"Found {len(runs)} runs with matching configuration")
    
    fitting_run = None
    fitting_artifact = None
    for run in runs:

        fitting_artifacts = [artifact for artifact in run.logged_artifacts() if artifact.name.startswith(artifact_name)]

        if len(fitting_artifacts) != 0:
            LOGGER.info(f"Found artifact in matching run {run.name}")
            fitting_run = run
            fitting_artifact = fitting_artifacts[0]
            break
    
    if not fitting_run or not fitting_artifact:
        LOGGER.info(f"No run with matching config had artifact {artifact_name}")
        return None
    
    file_name = f"{artifact_name}.pth"

    model_weights = fitting_artifact.get_entry(file_name)
    tmp_path = os.path.join(models_path, "temp")
    model_weights.download(tmp_path)
    downloaded_file_path = os.path.join(tmp_path, file_name)

    load_model(model, downloaded_file_path)
    os.remove(downloaded_file_path)

    return model

def check_dimensions(run_config, auto_encoder_config, diffusion_model_unet_config):
    # Check that the image dimensions match downsampling dimensions
    for image_dim in run_config["input_image_crop_roi"]:
        down_sample_factor_autoencoder = len(auto_encoder_config["num_channels"]) - 1
        down_sample_factor_diffusion_model = len(diffusion_model_unet_config["num_channels"]) - 1
        assert (image_dim % (down_sample_factor_autoencoder * down_sample_factor_diffusion_model)) == 0,\
            f"image dim {image_dim} must be evenly divisible by autoencoder downsampling factor {down_sample_factor_autoencoder} * diffusion dowmsampling factor {down_sample_factor_diffusion_model}"

    LOGGER.info("Image dimensions match downsampling factors! Good to go!")