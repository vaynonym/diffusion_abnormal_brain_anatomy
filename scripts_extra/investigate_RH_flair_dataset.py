base_path = "/depict/users/claes/shared/MS_data_orig_pull"
import os.path as path

import os

top_level = os.listdir(base_path)

all_paths = []

print(f"Top-level: {len(top_level)}")
print(f"Sample: {top_level[0]}")

number_unexpected_files_sub_level_one = 0
number_unexpected_folders_sub_level_two = 0

number_CSVs = 0
number_jsons = 0
number_images = 0
number_masks = 0
unaccounted_for_files = 0

unexpected_number_of_files = 0


import tqdm

unexpected_number_of_files_samples = []

paths_with_file_missing = []

for dir_path in tqdm.tqdm(top_level):
    sub_level_one_base_path = path.join(base_path, dir_path)
    sub_level_one = os.listdir(sub_level_one_base_path)

    for sub_level_one_path in sub_level_one:
        sub_level_two_base_path = path.join(sub_level_one_base_path, sub_level_one_path)

        if not path.isdir(sub_level_two_base_path):
            number_unexpected_files_sub_level_one += 1
        else:
            sub_level_two = os.listdir(sub_level_two_base_path)
            if len(sub_level_two) != 4:
                unexpected_number_of_files += 1
                if len(unexpected_number_of_files_samples ) < 3:
                    unexpected_number_of_files_samples.append(sub_level_two)
            
            expected_files_present_counter = 0

            for sub_level_two_path in sub_level_two:
                sub_level_three_base_path = path.join(sub_level_two_base_path, sub_level_two_path)
                

                if path.isdir(sub_level_three_base_path):
                    number_unexpected_folders_sub_level_two += 1
                else:
                    if sub_level_two_path == "FLAIR_synthseg_volumes.csv":
                        number_CSVs += 1
                        expected_files_present_counter += 1
                    elif sub_level_two_path == "FLAIR.json":
                        number_jsons += 1
                        expected_files_present_counter += 1
                    elif sub_level_two_path == "FLAIR_synthseg.nii.gz":
                        number_masks += 1
                        expected_files_present_counter += 1
                    elif sub_level_two_path == "FLAIR.nii.gz":
                        number_images += 1
                        expected_files_present_counter += 1
                    else:
                        unaccounted_for_files += 1
                    
            if expected_files_present_counter != 4:
                paths_with_file_missing.append(sub_level_two_base_path)


print(f"number_unexpected_files_sub_level_one : {number_unexpected_files_sub_level_one}") 
print(f"number_unexpected_folders_sub_level_two : {number_unexpected_folders_sub_level_two}") 
print(f"number_CSVs : {number_CSVs}") 
print(f"number_jsons : {number_jsons}") 
print(f"number_images : {number_images}") 
print(f"number_masks : {number_masks}") 
print(f"unaccounted_for_files : {unaccounted_for_files}") 
print(f"unexpected_number_of_files : {unexpected_number_of_files}") 


import pprint

print("Missing files:")
pprint.pp(paths_with_file_missing)

print("Unexpected number of files:")
pprint.pprint(unexpected_number_of_files_samples)

print("All Done!")





mask_file_path = "FLAIR_synthseg.nii.gz"
image_file_path = "FLAIR.nii.gz"
json_path = "FLAIR.json"
volumes_csv_path = "FLAIR_synthseg_volumes.csv"

import tqdm
import json
import pandas as pd

from src.logging_util import LOGGER

def create_dataset_overview(
    self,
    dataset_path,
):
    if not path.isdir(dataset_path):
        raise ValueError("Root directory root_dir must be a directory.")

    self.datalist = []

    skipped_because_of_volumes = []
    skipped_because_2D = []
    skipped_because_no_info_about_acquisition_type = []
    too_many_volumes = []
    MRAcquisitionType = set()

    good_scan_paths = []

    volumes = []

    for rel_patient_path in tqdm.tqdm(os.listdir(dataset_path)):
        patient_path = os.path.join(dataset_path, rel_patient_path)
        for rel_scan_path in os.listdir(patient_path):
            scan_path = os.path.join(patient_path, rel_scan_path)

            image_file = os.path.join(scan_path, image_file_path)
            mask_file = os.path.join(scan_path, mask_file_path)
            json_file = os.path.join(scan_path, json_path)
            volumes_file = os.path.join(scan_path, volumes_csv_path)

            with open(json_file) as f:
                json_f = json.load(f)
                if not "MRAcquisitionType" in json_f:
                    skipped_because_no_info_about_acquisition_type.append(scan_path)
                elif json_f["MRAcquisitionType"] == "2D":
                    skipped_because_2D.append(scan_path)
                    continue
                else:
                    MRAcquisitionType.add(json_f["MRAcquisitionType"])

            volume_df = pd.read_csv(volumes_file)
            assert (volume_df["subject"] == "FLAIR").all()
            volume_df = volume_df.drop("subject", axis=1)
            if volume_df.shape[0] != 1:
                too_many_volumes.append(scan_path)
                continue

            assert volume_df.shape[0] == 1, "CSV should contain only one entry"

            if volume_df.isna().values.any() or volume_df.values.astype(float).__le__(10).any():
                skipped_because_of_volumes.append(scan_path)
                continue

            good_scan_paths.append(scan_path)


    LOGGER.warning(f"skipped_because_2D={pprint.pformat(skipped_because_2D)}")
    LOGGER.warning(f"skipped_because_of_volumes={pprint.pformat(skipped_because_of_volumes)}")
    LOGGER.warning(f"skipped_because_no_info_about_acquisition_type={pprint.pformat(skipped_because_no_info_about_acquisition_type)}")
    LOGGER.warning(f"too_many_volumes={pprint.pformat(too_many_volumes)}")


    LOGGER.warning(f"skipped_because_2D={len(skipped_because_2D)}")
    LOGGER.warning(f"skipped_because_of_volumes={len(skipped_because_of_volumes)}")
    LOGGER.warning(f"skipped_because_no_info_about_acquisition_type={len(skipped_because_no_info_about_acquisition_type)}")
    LOGGER.warning(f"too_many_volumes={len(too_many_volumes)}")
    LOGGER.info(f"Acquisition Types Used: {MRAcquisitionType}")

    LOGGER.info(f"Good scans: {len(good_scan_paths)}")

    from src.directory_management import OUTPUT_DIRECTORY

    with open(os.path.join(OUTPUT_DIRECTORY, "RH_3D_subset_paths"), 'w') as f:
        for scan_path in good_scan_paths:
            f.write(scan_path)
            f.write("\n")
