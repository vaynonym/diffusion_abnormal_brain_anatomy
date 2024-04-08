import torch
from src.logging_util import LOGGER
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using device {os.environ.get('CUDA_VISIBLE_DEVICES')} of {torch.cuda.get_device_name(device)}"
              if torch.cuda.is_available() else "cpu")
