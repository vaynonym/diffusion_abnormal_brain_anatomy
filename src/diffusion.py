import torch
from generative.inferers import LatentDiffusionInferer
from generative.networks.nets import AutoencoderKL, DiffusionModelUNet
from generative.networks.schedulers import DDPMScheduler

from src.util import log_image_to_wandb
from src.logging_util import LOGGER
from torch.cuda.amp import autocast

from src.torch_setup import device


def get_scale_factor(autoencoder: AutoencoderKL, sample_data: torch.Tensor) -> torch.Tensor:
    # ### Scaling factor
    #
    # As mentioned in Rombach et al. [1] Section 4.3.2 and D.1, the signal-to-noise ratio (induced by the scale of the latent space) can affect the results obtained with the LDM, if the standard deviation of the latent space distribution drifts too much from that of a Gaussian. For this reason, it is best practice to use a scaling factor to adapt this standard deviation.
    #
    # _Note: In case where the latent space is close to a Gaussian distribution, the scaling factor will be close to one, and the results will not differ from those obtained when it is not used._
    #

    # +
    scale_factor = torch.tensor(1)
    with torch.no_grad():
        with autocast(enabled=True):
            z = autoencoder.encode_stage_2_inputs(sample_data.to(device))

            scale_factor = 1 / torch.std(z)
            LOGGER.info(f"Scaling factor set to {scale_factor}")
    return scale_factor