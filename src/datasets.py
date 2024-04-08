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
        sorted_by_volume=False
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
        
        
        def normalize(x, min_x, max_x):
            return (x -  min_x) / (max_x - min_x)
        
        max_volume = np.max(volumes)
        min_volume = np.min(volumes)

        def normalize_volume(d):
            d["volume"] = normalize(d["volume"], min_volume, max_volume)
            return d

        self.datalist = list(map(normalize_volume, self.datalist))

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