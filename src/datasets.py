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
        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )

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