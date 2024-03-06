import torch
from src.logging_util import LOGGER

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
LOGGER.info(f"Using {torch.cuda.get_device_name(device) if torch.cuda.is_available() else 'cpu'}")
