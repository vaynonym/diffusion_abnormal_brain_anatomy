from src.datasets import RHFlairDataset, SyntheticLDM100K
from monai import transforms
import torch
from src.directory_management import DATA_DIRECTORY
import os


FHFlairDataset_path = "/depict/users/claes/shared/MS_data_orig_pull"
volume_only_transforms = transforms.Compose([
        transforms.Lambdad(keys=["volume"], 
                           func = lambda x: torch.tensor([x], dtype=torch.float32).unsqueeze(0)),
    ])

#train_ds = SyntheticLDM100K(
#    dataset_path=os.path.join(DATA_DIRECTORY, "LDM_100k"),
#    section="training",
#    size=9999,
#    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
#    num_workers=6,
#    seed=0,
#    transform=volume_only_transforms,
#    display_statistics=True,
#)

train_ds = RHFlairDataset(
    dataset_path=FHFlairDataset_path,
    section="training",
    size=10000,
    cache_rate=1.0,  # you may need a few Gb of RAM... Set to 0 otherwise
    num_workers=6,
    seed=0,
    transform=volume_only_transforms,
)

from src.dataset_analysis import log_bucket_counts
import numpy as np

np.set_printoptions(suppress=True)

log_bucket_counts(train_ds)