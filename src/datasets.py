from monai.data import CacheDataset
from monai.transforms import Randomizable
import os.path as path
import pandas as pd
import sys
import csv
import numpy as np

VENTRICLE_KEYS = [
    "left lateral ventricle",
    "left inferior lateral ventricle",
    "right lateral ventricle",
    "right inferior lateral ventricle",
    "3rd ventricle",
    "4th ventricle",
]
    
BRAIN_SIZE_KEY = "total intracranial"

def normalize_volumes(volumes, datalist):
    def normalize(x, min_x, max_x):
        return (x -  min_x) / (max_x - min_x)
    
    max_volume = np.max(volumes)
    min_volume = np.min(volumes)

    def normalize_volume(d):
        d["volume"] = normalize(d["volume"], min_volume, max_volume)
        return d

    return list(map(normalize_volume, datalist))

class SyntheticLDM100K(Randomizable, CacheDataset):

    def __init__(
        self,
        dataset_path,
        section,
        transform,
        seed=0,
        size=1000,
        val_frac=0.2,
        test_frac=0.0,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
        sorted_by_volume=False,
        filter_function=None,
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        self.datalist = []
        volumes = []
        with open(path.join(dataset_path, "participants.tsv"), "r") as participantsCSV:
            reader = csv.reader(participantsCSV, delimiter='\t')
            next(reader) # skip header row
            count = 0
            for line in reader:
                sub_folder_name = line[0]
                image_file_name = sub_folder_name + "_T1w.nii.gz"
                mask_file_name = sub_folder_name + "_T1w.nii.gz"
                volume_file_name = sub_folder_name + "_synthseg_volumes.csv"


                image_path = path.join(dataset_path, "rawdata", sub_folder_name, "anat", image_file_name)
                mask_path = path.join(dataset_path, "masks", sub_folder_name, "anat", mask_file_name)
                volume_path = path.join(dataset_path, "masks", sub_folder_name, "anat", volume_file_name)

                volume_df = pd.read_csv(volume_path)
                assert len(volume_df) == 1, "CSV should contain only one entry"

                # sum over ventricle columns
                ventricle_volume = volume_df[VENTRICLE_KEYS].sum(axis=1)[0]
                brain_volume = volume_df[BRAIN_SIZE_KEY][0]

                relative_volume = ventricle_volume/brain_volume

                self.datalist.append({
                    "image": image_path,
                    "mask": mask_path,
                    "volume": relative_volume
                })
                volumes.append(relative_volume)
                count += 1
                if count >= size:
                    break
        

        self.datalist = normalize_volumes(volumes, self.datalist)

        if filter_function is not None:
            LOGGER.info("Filtering dataset...")
            self.datalist = list(filter(filter_function, self.datalist))

        data = self._generate_data_list()

        if sorted_by_volume:
            LOGGER.info("Sorting the dataset...")
            data = self._sort_by_volume(data)

        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def _sort_by_volume(self, data):
        sorted(data, key=lambda d: d["volume"])
        return data

    def randomize(self, data=None):
        self.rann = self.R.random()

    def _generate_data_list(self):
        data = []
        for d in self.datalist:
            self.randomize()
            if self.section == "training":
                if self.rann < self.val_frac + self.test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= self.val_frac:
                    continue
            elif self.section == "test":
                if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
                    continue
            else:
                raise ValueError(
                    f"Unsupported section: {self.section}, " "available options are ['training', 'validation', 'test']."
                )
            data.append(d)
        return data

from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.logging_util import LOGGER
import torch

def get_dataloader(data: Dataset, run_config: dict):
    if run_config["oversample_large_ventricles"]:
        with torch.no_grad():
            weights = None

            bins = torch.tensor(np.arange(0, 1.01, 0.1), dtype=torch.float)
# ensure that every normal value is within 0 to 1 buckets, essentially making [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0], (1.0, 1.1], ...
            bins[0] = -0.001 

            volumes = torch.tensor([x["volume"][0, 0] for x in data])


            # inverse frequency, menaing we value each bin an equal amount
            hist = torch.histogram(volumes, bins=bins)[0]

            LOGGER.info(f"Histogram per bin: {hist}")
            weights_per_bin = hist.pow(-1)
            LOGGER.info(f"Using weights based on ventricular volume ratio per 0.1-wide bin: {weights_per_bin}")

            weights = [weights_per_bin[i - 1] for i in torch.bucketize(volumes.flatten(), boundaries=bins)]


            return DataLoader(data, 
                    batch_size=run_config["batch_size"],
                    shuffle=False,
                    num_workers=2,
                    drop_last=True,
                    persistent_workers=True,
                    sampler=WeightedRandomSampler(weights=weights, num_samples=len(data), replacement=True),
                    )
    else:
        return DataLoader(data, 
                      batch_size=run_config["batch_size"],
                      shuffle=True,
                      num_workers=2,
                      drop_last=True,
                      persistent_workers=True)




class AbnormalSyntheticMaskDataset(Randomizable, CacheDataset):
    OVERVIEW_FILENAME = "overview.tsv"

    def __init__(
        self,
        dataset_path,
        section,
        transform,
        seed=0,
        size=1000,
        val_frac=0.2,
        test_frac=0.0,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
        sorted_by_volume=False
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        self.datalist = []
        with open(path.join(dataset_path, "overview.tsv"), "r") as participantsCSV:
            reader = csv.reader(participantsCSV, delimiter='\t')
            next(reader) # skip header row
            count = 0
            for line in reader:
                sub_folder_name = line[0]
                volume_value = float(line[1])
                mask_file_name = sub_folder_name + ".nii.gz"


                mask_path = path.join(dataset_path, "data", mask_file_name)

                self.datalist.append({
                    "mask": mask_path,
                    "volume": volume_value
                })
                count += 1
                if count >= size:
                    break

        data = self._generate_data_list()

        if sorted_by_volume:
            data = self._sort_by_volume(data)

        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def _sort_by_volume(self, data):
        sorted(data, key=lambda d: d["volume"])
        return data

    def randomize(self, data=None):
        self.rann = self.R.random()

    def _generate_data_list(self):
        data = []
        for d in self.datalist:
            self.randomize()
            if self.section == "training":
                if self.rann < self.val_frac + self.test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= self.val_frac:
                    continue
            elif self.section == "test":
                if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
                    continue
            else:
                raise ValueError(
                    f"Unsupported section: {self.section}, " "available options are ['training', 'validation', 'test']."
                )
            data.append(d)
        return data


import os
import json
import pprint


class RHFlairDataset(Randomizable, CacheDataset):

    mask_file_path = "FLAIR_synthseg.nii.gz"
    image_file_path = "FLAIR.nii.gz"
    json_path = "FLAIR.json"
    volumes_csv_path = "FLAIR_synthseg_volumes.csv"

    def __init__(
        self,
        dataset_path,
        section,
        transform,
        seed=0,
        size=7000,
        val_frac=0.2,
        test_frac=0.0,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
        sorted_by_volume=False
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        self.datalist = []

        #skipped_because_of_volumes = []
        #skipped_because_2D = []
        #skipped_because_no_info_about_acquisition_type = []
        #too_many_volumes = []
        MRAcquisitionType = set()

        good_scan_paths = []

        volumes = []
        import tqdm

        for rel_patient_path in tqdm.tqdm(os.listdir(dataset_path), desc="Collecting files and volumes..."):
            patient_path = os.path.join(dataset_path, rel_patient_path)
            for rel_scan_path in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, rel_scan_path)

                image_file = os.path.join(scan_path, RHFlairDataset.image_file_path)
                mask_file = os.path.join(scan_path, RHFlairDataset.mask_file_path)
                json_file = os.path.join(scan_path, RHFlairDataset.json_path)
                volumes_file = os.path.join(scan_path, RHFlairDataset.volumes_csv_path)

                with open(json_file) as f:
                    json_f = json.load(f)
                    if not "MRAcquisitionType" in json_f:
                        #skipped_because_no_info_about_acquisition_type.append(scan_path)
                        continue
                    elif json_f["MRAcquisitionType"] == "2D":
                        #skipped_because_2D.append(scan_path)
                        continue
                    else:
                        MRAcquisitionType.add(json_f["MRAcquisitionType"])

                volume_df = pd.read_csv(volumes_file)
                assert (volume_df["subject"] == "FLAIR").all()
                volume_df = volume_df.drop("subject", axis=1)

                if volume_df.shape[0] != 1:
                    #too_many_volumes.append(scan_path)
                    # Use the first row (rows are equivalent)
                    volume_df = volume_df.drop(volume_df.index[range(1, volume_df.shape[0])], axis=0)

                assert volume_df.shape[0] == 1, "CSV should contain only one entry"

                if volume_df.isna().values.any() or volume_df.values.astype(float).__le__(10).any():
                    #skipped_because_of_volumes.append(scan_path)
                    continue


                # sum over ventricle columns
                ventricle_volume = volume_df[VENTRICLE_KEYS].sum(axis=1)[0]
                brain_volume = volume_df[BRAIN_SIZE_KEY][0]

                relative_volume = ventricle_volume/brain_volume

                self.datalist.append({
                    "image": image_file,
                    "image_path": image_file,
                    "mask": mask_file,
                    "mask_path": mask_file,
                    "volume": relative_volume
                })
                volumes.append(relative_volume)

                if len(self.datalist) >= size: break
            if len(self.datalist) >= size: break


        self.datalist = normalize_volumes(volumes, self.datalist)  

        #LOGGER.warning(f"skipped_because_2D={pprint.pformat(skipped_because_2D)}")
        #LOGGER.warning(f"skipped_because_of_volumes={pprint.pformat(skipped_because_of_volumes)}")
        #LOGGER.warning(f"skipped_because_no_info_about_acquisition_type={pprint.pformat(skipped_because_no_info_about_acquisition_type)}")
        #LOGGER.warning(f"too_many_volumes={pprint.pformat(too_many_volumes)}")


        #LOGGER.warning(f"skipped_because_2D={len(skipped_because_2D)}")
        #LOGGER.warning(f"skipped_because_of_volumes={len(skipped_because_of_volumes)}")
        #LOGGER.warning(f"skipped_because_no_info_about_acquisition_type={len(skipped_because_no_info_about_acquisition_type)}")
        #LOGGER.warning(f"too_many_volumes={len(too_many_volumes)}")
        LOGGER.info(f"Acquisition Types used: {MRAcquisitionType}")

        #from src.directory_management import OUTPUT_DIRECTORY

        #with open(os.path.join(OUTPUT_DIRECTORY, "RH_3D_subset_paths"), 'w') as f:
        #    for scan_path in good_scan_paths:
        #        f.write(scan_path)
        #        f.write("\n")

        

        data = self._generate_data_list()

        if sorted_by_volume:
            data = self._sort_by_volume(data)

        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

    def _sort_by_volume(self, data):
        sorted(data, key=lambda d: d["volume"])
        return data

    def randomize(self, data=None):
        self.rann = self.R.random()

    def _generate_data_list(self):
        data = []
        for d in self.datalist:
            self.randomize()
            if self.section == "training":
                if self.rann < self.val_frac + self.test_frac:
                    continue
            elif self.section == "validation":
                if self.rann >= self.val_frac:
                    continue
            elif self.section == "test":
                if self.rann < self.val_frac or self.rann >= self.val_frac + self.test_frac:
                    continue
            else:
                raise ValueError(
                    f"Unsupported section: {self.section}, " "available options are ['training', 'validation', 'test']."
                )
            data.append(d)
        return data