import os

PRETRAINED_MODEL_DIRECTORY = os.environ.get("PRETRAINED_MODEL_DIRECTORY")
DATA_DIRECTORY = os.environ.get("DATA_DIRECTORY")
FLAIR_DATASET_DIRECTORY = os.environ.get("FLAIR_DATASET_DIRECTORY")
BASE_DIRECTORY = os.environ.get("BASE_DIRECTORY")
OUTPUT_DIRECTORY = os.environ.get("OUTPUT_DIRECTORY")
MODEL_DIRECTORY = os.environ.get("MODEL_DIRECTORY")

for dir_path in [PRETRAINED_MODEL_DIRECTORY,
                 OUTPUT_DIRECTORY,
                 DATA_DIRECTORY,
                 FLAIR_DATASET_DIRECTORY,
                 MODEL_DIRECTORY,
                 ]:
    assert os.path.exists(dir_path), f"path {dir_path} needs to exist"
    assert os.path.isdir(dir_path), f"path {dir_path} needs to be a directory"


