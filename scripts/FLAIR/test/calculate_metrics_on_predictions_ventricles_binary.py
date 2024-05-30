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


from src.util import Stopwatch, read_config
from src.logging_util import LOGGER
from src.datasets import get_dataloader


torch.multiprocessing.set_sharing_strategy('file_system')

LOGGER.info(f"Device count: {torch.cuda.device_count()}", )
LOGGER.info(f"Device: {device}")


WANDB_LOG_IMAGES = os.environ.get("WANDB_LOG_IMAGES")
WANDB_RUN_NAME = os.environ.get("WANDB_RUN_NAME")

experiment_config = read_config(os.environ.get("EXPERIMENT_CONFIG"))
run_config = experiment_config["run"]

# for reproducibility purposes set a seed
set_determinism(42)

import numpy as np





from src.directory_management import OUTPUT_DIRECTORY





from src.util import visualize_3d_image_slice_wise
from src.synthseg_masks import encode_one_hot, decode_one_hot_to_consecutive_indices

from tqdm import tqdm

from monai.data.meta_tensor import MetaTensor
from pprint import pprint

prediction_models = {
 "nnU Real": "777",
 "nnU Augmented": "778",
 "nnU Synthetic": "801"
}

dataset_path = "/depict/users/tim/private/correct_masks/nnUNet_raw/Task809_TestRealBrainFlair"
#dataset_path = "/depict/users/tim/private/correct_masks/nnUNet_raw/Task810_Test2RealBrainFlair"
#dataset_path = "/depict/users/tim/private/correct_masks/nnUNet_raw/Task808_ValidationRealBrainFlair"

ground_truth_path = os.path.join(dataset_path, "labelsTr")
image_path = os.path.join(dataset_path, "imagesTr")
synthseg_path = os.path.join(dataset_path, "synthseg_labels")
if not os.path.exists(synthseg_path):
    synthseg_path = None


model_prediction_folder_prefix = "predictions_"

LOGGER.info(ground_truth_path)

label_file_names = [x for x in os.listdir(ground_truth_path) if x.endswith(".nii.gz") ]

from src.synthseg_masks import synthseg_class_to_string_map, ventricles


def synthseg_to_binary_ventricle_mask(mask):
    ventricle_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in ventricles]).to(device)

    return torch.isin(mask, ventricle_indices)

def ventricle_only_mask_to_binary_ventricle_mask(mask):
    return mask > 0

from collections import defaultdict

scores_IoU = defaultdict(lambda: [])
scores_Dice = defaultdict(lambda: [])
scores_MLL_error = defaultdict(lambda: [])
scores_MLL_squared_error = defaultdict(lambda: [])



from torchmetrics.classification import BinaryF1Score, BinaryJaccardIndex as BinaryIoU


dice_score = BinaryF1Score()
iou_score = BinaryIoU()

from src.synthseg_masks import decode_one_hot, encode_contiguous_labels_one_hot

MM_3_TO_MLL_CONSTANT = 0.001
SPACING_CONSTANT = 1.75 * 1.667 * 1.667
import math

volume_outlier_tracking = []

all_volumes = []

for label_file_name in tqdm(label_file_names):

    GT_mask = transforms.LoadImaged(keys="mask")({"mask": os.path.join(ground_truth_path, label_file_name)})["mask"].int().unsqueeze(0).unsqueeze(0)
    GT_mask = ventricle_only_mask_to_binary_ventricle_mask(GT_mask)
    #GT_mask = synthseg_to_binary_ventricle_mask(decode_one_hot(encode_contiguous_labels_one_hot(GT_mask.to(device)))).cpu().detach()

    GT_volume_in_MLL = GT_mask.sum() * SPACING_CONSTANT * MM_3_TO_MLL_CONSTANT

    current_volume_collection = {"Ground Truth": GT_volume_in_MLL}

    if synthseg_path is not None:
        synthseg_mask = transforms.LoadImaged(keys="synthseg_mask")({"synthseg_mask": os.path.join(synthseg_path, label_file_name)})["synthseg_mask"].int().unsqueeze(0).unsqueeze(0)
        synthseg_mask = synthseg_to_binary_ventricle_mask(synthseg_mask.to(device)).cpu().detach().int()

        synthseg_volume_in_MLL = synthseg_mask.sum() * SPACING_CONSTANT * MM_3_TO_MLL_CONSTANT
        scores_Dice["synthseg"].append(dice_score(synthseg_mask, GT_mask).cpu().numpy())
        scores_IoU["synthseg"].append(iou_score(synthseg_mask, GT_mask).cpu().numpy())
        scores_MLL_error["synthseg"].append(abs(GT_volume_in_MLL - synthseg_volume_in_MLL))
        scores_MLL_squared_error["synthseg"].append(math.pow(GT_volume_in_MLL - synthseg_volume_in_MLL, 2))
        current_outlier_tracking = dict()
        current_outlier_tracking["name"] = label_file_name
        current_outlier_tracking["synthseg"] = abs((GT_volume_in_MLL - synthseg_volume_in_MLL).item())

        current_volume_collection["SynthSeg"] = synthseg_volume_in_MLL

    for name, index in prediction_models.items():
        prediction_mask_path = os.path.join(dataset_path, model_prediction_folder_prefix + index, label_file_name)
        prediction_mask = transforms.LoadImaged(keys="mask")({"mask": prediction_mask_path})["mask"].int().unsqueeze(0).unsqueeze(0).to(device)
        prediction_mask = decode_one_hot(encode_contiguous_labels_one_hot(prediction_mask))

        prediction_mask = synthseg_to_binary_ventricle_mask(prediction_mask).cpu().detach().int()

        prediction_volume_in_MLL = prediction_mask.sum() * SPACING_CONSTANT * MM_3_TO_MLL_CONSTANT

        scores_Dice[name].append(dice_score(prediction_mask, GT_mask).cpu().numpy())
        scores_IoU[name].append(iou_score(prediction_mask, GT_mask).cpu().numpy())
        scores_MLL_error[name].append(abs(GT_volume_in_MLL - prediction_volume_in_MLL))
        scores_MLL_squared_error[name].append(math.pow(GT_volume_in_MLL - prediction_volume_in_MLL, 2))

        #visualize_3d_image_slice_wise(prediction_mask[0][0].int(), OUTPUT_DIRECTORY + "/visualized_testset_one" + f"/{label_file_name}_{name}.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
        #                            is_image_mask=True)
        
        current_outlier_tracking[name] = abs((GT_volume_in_MLL - prediction_volume_in_MLL).item())
        current_volume_collection[name] = prediction_volume_in_MLL


    #visualize_3d_image_slice_wise(GT_mask[0][0].int(), OUTPUT_DIRECTORY + "/visualized_testset_one" + f"/{label_file_name}_GT.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
    #                                is_image_mask=True)
    
    #if synthseg_path is not None:
    #    visualize_3d_image_slice_wise(synthseg_mask[0][0].int(), OUTPUT_DIRECTORY + "/visualized_testset_one" + f"/{label_file_name}_synthseg.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
    #                                is_image_mask=True)
    
    #img_file_name = label_file_name[:-len(".nii.gz")] + "_0000.nii.gz"
    #img = transforms.LoadImaged(keys="image")({"image": os.path.join(image_path, img_file_name)})["image"]
    #visualize_3d_image_slice_wise(img, OUTPUT_DIRECTORY + "/visualized_testset_one" + f"/{label_file_name}_image.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
    #                        is_image_mask=False)
    
    volume_outlier_tracking.append(current_outlier_tracking)
    all_volumes.append(current_volume_collection)
    
names = list(prediction_models.keys())

if synthseg_path is not None:
    names.append("synthseg")

for name in names:
    mean_dice = np.mean(scores_Dice[name], axis=0)
    std_dice = np.std(scores_Dice[name], axis=0)
    mean_iou = np.mean(scores_IoU[name], axis=0)
    std_iou = np.std(scores_IoU[name], axis=0)

    mean_MLL_error = np.mean(scores_MLL_error[name], axis=0)
    std_MLL_error = np.std(scores_MLL_error[name], axis=0)
    mean_MLL_squared_error = np.mean(scores_MLL_squared_error[name], axis=0)
    std_MLL_squared_error = np.std(scores_MLL_squared_error[name], axis=0)

    LOGGER.info("=====================")
    LOGGER.info(f"{name}")
    LOGGER.info(f"dice={mean_dice:.03f}+-{std_dice:.03f}, IoU={mean_iou:.03f}+-{std_iou:.03f}")
    LOGGER.info(f"Raw volume in MLL: MAE={mean_MLL_error:.03f}+-{std_MLL_error:.03f}, MSE={mean_MLL_squared_error:.03f}+-{std_MLL_squared_error:.03f}")

volume_outlier_tracking.sort(key= lambda tup: tup["synthseg"], reverse=True)

LOGGER.info(f"Top 3 outlier cases: {volume_outlier_tracking[:3]}")


all_volumes.sort(key=lambda d: d["Ground Truth"])

import matplotlib.pyplot as plt

colors = ["blue", "red", "orange", "green", "purple"]
transparencies = [0.9, 0.6, 0.6, 0.6, 0.6]
markers = ["o", ".", ".", ".", "."]
markers_sizes = [12] + [15] * 4

for i, key in enumerate(all_volumes[0].keys()):
    entries = list(map(lambda volumes: volumes[key], all_volumes))

    plt.plot(range(len(entries)), entries, color=colors[i], label=key, linestyle="",
             marker=markers[i], markersize=markers_sizes[i], alpha=transparencies[i])

lgd = plt.legend(loc="center left", bbox_to_anchor=(1, 0.5))
plt.xticks([])
plt.xlabel("Patients")
plt.ylabel("Ventricular Volume in ml")
plt.savefig("../output/volumes_per_patient.png", bbox_extra_artists=[lgd], bbox_inches="tight")