#!/usr/bin/env python3

# Set up imports
import os
import torch
import torch.nn.functional as F
import torch.multiprocessing
import wandb
from src.torch_setup import device

from monai import transforms
from monai.utils import set_determinism


from src.logging_util import LOGGER


torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")

# for reproducibility purposes set a seed
set_determinism(42)

import numpy as np


from src.synthseg_masks import decode_one_hot, encode_contiguous_labels_one_hot
import math

from src.synthseg_masks import synthseg_class_to_string_map, ventricles
from collections import defaultdict
from src.util import visualize_3d_image_slice_wise
from src.synthseg_masks import encode_one_hot, decode_one_hot_to_consecutive_indices
from tqdm import tqdm
from monai.data.meta_tensor import MetaTensor
from pprint import pprint

prediction_models = {
 "NNU Real": "777",
 "NNU Augmented": "778",
 "NNU Synthetic": "801"
}

dataset_paths = {
    "Real" : "Dataset777_RealBrainFlair",
    "Augmented" : "Dataset778_AugmentedBrainFlair",
    "Synthetic" : "Dataset801_PurelySynthetic",
}

base_path = "/depict/users/tim/private/correct_masks/nnUNet_raw/"

label_path = "labelsTr"

def synthseg_to_binary_ventricle_mask(mask):
    ventricle_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in ventricles]).to(device)

    return torch.isin(mask, ventricle_indices)

def ventricle_only_mask_to_binary_ventricle_mask(mask):
    return mask > 0

def get_volumes_for_dataset(dataset_path):
    full_label_path = os.path.join(base_path, dataset_path, label_path)
    label_file_names = [x for x in os.listdir(full_label_path) if x.endswith(".nii.gz") ]

    MM_3_TO_MLL_CONSTANT = 0.001
    SPACING_CONSTANT = 1.75 * 1.667 * 1.667

    all_volumes = []
    import random
    random.shuffle(label_file_names)

    for label_file_name in tqdm(label_file_names[:1000]):

        mask = transforms.LoadImaged(keys="mask")({"mask": os.path.join(full_label_path, label_file_name)})["mask"].int().unsqueeze(0).unsqueeze(0)
        mask = synthseg_to_binary_ventricle_mask(decode_one_hot(encode_contiguous_labels_one_hot(mask.to(device)))).cpu().detach()

        volume_in_MLL = mask.sum().item() * SPACING_CONSTANT * MM_3_TO_MLL_CONSTANT
        all_volumes.append(volume_in_MLL)
    
    return all_volumes

results = {}

for name, dataset_path in dataset_paths.items():
    volumes_for_dataset = get_volumes_for_dataset(dataset_path)
    volumes_for_dataset.sort()
    results[name] = volumes_for_dataset


import matplotlib.pyplot as plt

colors = ["orange", "green", "purple"]

all_volumes = list(results.values())
labels = list(results.keys())

import seaborn
import pandas as pd

df_results = pd.DataFrame.from_dict(results)


plot = seaborn.displot(data=df_results, kind="kde", cut=0, fill=True)



#lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
#plt.xticks([])
#plt.xlabel("Patients")
plt.xlabel("Ventricular Volume in mL", fontsize="16")
plt.ylabel("Density", fontsize="16")
plot.savefig("../output/volumes_density.png")