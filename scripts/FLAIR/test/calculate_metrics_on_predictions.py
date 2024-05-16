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
 "Real": "777",
 "Augmented": "778",
 "Synthetic": "801"
}

dataset_path = "/depict/users/tim/private/correct_masks/nnUNet_raw/Task809_TestRealBrainFlair"
ground_truth_path = os.path.join(dataset_path, "labelsTr")

model_prediction_folder_prefix = "predictions_"

LOGGER.info(ground_truth_path)

label_file_names = [x for x in os.listdir(ground_truth_path) if x.endswith(".nii.gz") ]

from src.synthseg_masks import synthseg_class_to_string_map, ventricles

classes_to_index_test_set = {
    "Background" : 0,
    "right lateral ventricle" : 1,
    "left lateral ventricle" : 2,
    "right inferior lateral ventricle" : 3,
    "left inferior lateral ventricle" : 4,
    "3rd ventricle" : 5,
    "4th ventricle" : 6,
}

index_to_classes_test_set = {
    0: "Background",
    1: "right lateral ventricle",
    2: "left lateral ventricle",
    3: "right inferior lateral ventricle",
    4: "left inferior lateral ventricle",
    5: "3rd ventricle",
    6: "4th ventricle",
}


def reduce_mask_to_ventricles(mask):
    ventricle_indices = torch.tensor([index for index, name in synthseg_class_to_string_map.items() 
                                      if name in ventricles]).to(device)

    return mask * torch.isin(mask, ventricle_indices)

from pprint import pprint

def match_ventricle_labels_from_synthseg(mask):

    og_ventricle_only_mask = reduce_mask_to_ventricles(mask)

    result_ventricle_only_mask = og_ventricle_only_mask.clone()

    # match indices
    ventricle_only_synthseg_classes = {k:v for k, v in synthseg_class_to_string_map.items() if v in ventricles}
    
    for index, name in ventricle_only_synthseg_classes.items():
        result_ventricle_only_mask[og_ventricle_only_mask == index] = classes_to_index_test_set[name]

    return result_ventricle_only_mask

from collections import defaultdict

scores_IoU = defaultdict(lambda: [])
scores_Dice = defaultdict(lambda: [])

from torchmetrics.classification import Dice, MulticlassJaccardIndex as IoU


dice_score = Dice(len(index_to_classes_test_set), ignore_index=0, average=None)
iou_score = IoU(num_classes=len(index_to_classes_test_set), ignore_index=0, average=None)

from src.synthseg_masks import decode_one_hot, encode_contiguous_labels_one_hot


for label_file_name in tqdm(label_file_names):

    GT_mask = transforms.LoadImaged(keys="mask")({"mask": os.path.join(ground_truth_path, label_file_name)})["mask"].int().unsqueeze(0).unsqueeze(0)

    for name, index in prediction_models.items():
        prediction_mask_path = os.path.join(dataset_path, model_prediction_folder_prefix + index, label_file_name)
        prediction_mask = transforms.LoadImaged(keys="mask")({"mask": prediction_mask_path})["mask"].int().unsqueeze(0).unsqueeze(0).to(device)
        prediction_mask = decode_one_hot(encode_contiguous_labels_one_hot(prediction_mask))

        prediction_mask = match_ventricle_labels_from_synthseg(prediction_mask).cpu().detach().int()

        scores_Dice[name].append(dice_score(prediction_mask, GT_mask).cpu().numpy())
        scores_IoU[name].append(iou_score(prediction_mask, GT_mask).cpu().numpy())
        visualize_3d_image_slice_wise(prediction_mask[0][0], OUTPUT_DIRECTORY + f"/{label_file_name}_{name}.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
                                    is_image_mask=True)


    visualize_3d_image_slice_wise(GT_mask[0][0], OUTPUT_DIRECTORY + f"/{label_file_name}_GT.png", description_prefix="", log_to_wandb=False, conditioning_information=None,
                                    is_image_mask=True)
    

for name in prediction_models:
    mean_dice = np.mean(scores_Dice[name], axis=0)
    std_dice = np.std(scores_Dice[name], axis=0)
    mean_iou = np.mean(scores_IoU[name], axis=0)
    std_iou = np.std(scores_IoU[name], axis=0)

    print(mean_dice.shape)
    LOGGER.info("=====================")
    LOGGER.info(f"{name}")
    for c, c_name in index_to_classes_test_set.items():
        if c == 0: continue
        
        LOGGER.info(f"{c_name}: dice={mean_dice[c - 1]:.03f}+-{std_dice[c - 1]:.03f}, IoU={mean_iou[c - 1]:.03f}+-{std_iou[c - 1]:.03f}")

    