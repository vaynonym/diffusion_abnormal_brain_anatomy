import os
from src.logging_util import LOGGER

LOGGER.info("Reading directory information from environment...")
PRETRAINED_MODEL_DIRECTORY = os.environ.get("PRETRAINED_MODEL_DIRECTORY")
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY")
BASE_DIRECTORY = os.environ.get("BASE_DIRECTORY")
OUTPUT_DIRECTORY = os.environ.get("OUTPUT_DIRECTORY")
MODEL_DIRECTORY = os.environ.get("MODEL_DIRECTORY")

for dir_path in [PRETRAINED_MODEL_DIRECTORY,
                 DATA_DIRECTORY,
                 DATA_DIRECTORY,
                 OUTPUT_DIRECTORY,
                 MODEL_DIRECTORY,
                 ]:
    assert os.path.exists(dir_path), f"path {dir_path} needs to exist"
    assert os.path.isdir(dir_path), f"path {dir_path} needs to be a directory"
