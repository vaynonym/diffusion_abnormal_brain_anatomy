from monai.data import CacheDataset
from monai.transforms import Randomizable
import os.path as path
import pandas as pd
import sys
import csv
import numpy as np
from src.directory_management import DATA_DIRECTORY, FLAIR_DATASET_DIRECTORY
from monai import transforms
from src.transforms import get_crop_around_mask_center
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from src.logging_util import LOGGER
import torch
import random


VENTRICLE_KEYS = [
    "left lateral ventricle",
    "left inferior lateral ventricle",
    "right lateral ventricle",
    "right inferior lateral ventricle",
    "3rd ventricle",
    "4th ventricle",
]
    
BRAIN_SIZE_KEY = "total intracranial"

def normalize_volumes(volumes, datalist, max_volume = None, min_volume = None):
    def normalize(x, min_x, max_x):
        return (x -  min_x) / (max_x - min_x)
    
    if max_volume is None:
        max_volume = np.max(volumes)
    if min_volume is None:
        min_volume = np.min(volumes)

    LOGGER.info(f"Normalizing using max={max_volume:.3f}, min={min_volume:.3f}")

    def normalize_volume(d):
        d["volume"] = normalize(d["volume"], min_volume, max_volume)
        if d["volume"] > 1.000001:
            LOGGER.warn(f"Normalized volume > 1.0 found: {d['volume']}")
        if d["volume"] < 0.0:
            LOGGER.warn(f"Normalized volume < 0.0 found: {d['volume']}")
        return d

    return list(map(normalize_volume, datalist)), max_volume, min_volume

class VolumeDataset():

    def analyze_raw_volumes(self):
        raw_volumes = np.array(list(map(lambda d: d["volume"], self.datalist)))

        LOGGER.info("==== Raw volume statistics ====")
        LOGGER.info(f"Max: {raw_volumes.max()}")
        LOGGER.info(f"Min: {raw_volumes.min()}")
        LOGGER.info(f"Mean: {raw_volumes.mean()}")
        LOGGER.info(f"Std: {raw_volumes.std()}")
        LOGGER.info("===============================")

class SyntheticLDM100K(Randomizable, CacheDataset, VolumeDataset):

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
        display_statistics=False,
        max_volume=None,
        min_volume=None,
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        self.datalist = []
        volumes = []

        LOGGER.info("Loading LDM100k Synthetic Dataset")

        tsv_volumes = []
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

                tsv_ventricle_volume = float(line[3])
                tsv_brain_volume = float(line[4])
                tsv_volumes.append(tsv_ventricle_volume / tsv_brain_volume)


                self.datalist.append({
                    "image": image_path,
                    "mask": mask_path,
                    "volume": relative_volume
                })
                volumes.append(relative_volume)
                count += 1
                if count >= size:
                    break
        
        if display_statistics:
            self.analyze_raw_volumes()
        
        tsv_array = np.array(tsv_volumes)
        LOGGER.info("==== Raw TSV volume statistics ====")
        LOGGER.info(f"Max: {tsv_array.max()}")
        LOGGER.info(f"Min: {tsv_array.min()}")
        LOGGER.info(f"Mean: {tsv_array.mean()}")
        LOGGER.info(f"Std: {tsv_array.std()}")
        LOGGER.info("===============================")
        del tsv_array

        if filter_function is not None:
            LOGGER.info("Filtering dataset...")
            self.datalist = list(filter(filter_function, self.datalist))

        self.datalist, self.max_volume, self.min_volume = normalize_volumes(volumes, self.datalist, max_volume, min_volume)


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

def get_bucket_counts(data: Dataset):    
    bins = torch.tensor(np.arange(0, 1.01, 0.1), dtype=torch.float)
    bins[0] = -0.001 

    volumes = torch.tensor([x["volume"][0, 0] for x in data])


    # inverse frequency, menaing we value each bin an equal amount
    hist = torch.histogram(volumes, bins=bins)[0]

    return hist


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
                      num_workers=4,
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


class RHFlairDataset(Randomizable, CacheDataset, VolumeDataset):

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
        size=10000,
        val_frac=0.2,
        test_frac=0.0,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
        sorted_by_volume=False,
        max_volume=None,
        min_volume=None,
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")
        self.section = section
        self.val_frac = val_frac
        self.test_frac = test_frac
        self.set_random_state(seed=seed)

        self.datalist = []

        skipped_because_of_volumes = []
        skipped_because_2D = []
        skipped_because_no_info_about_acquisition_type = []
        too_many_volumes = []
        MRAcquisitionType = set()

        good_scan_paths = []

        volumes = []
        
        skipped_because_no_slice_thickness = []
        skipped_because_slice_thickness_too_high = []

        IDs_used = []

        col_0s = set()

        import tqdm

        for rel_patient_path in tqdm.tqdm(os.listdir(dataset_path), desc="Collecting files and volumes..."):
            patient_path = os.path.join(dataset_path, rel_patient_path)
            used_at_least_once = False
            for rel_scan_path in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, rel_scan_path)

                image_file = os.path.join(scan_path, RHFlairDataset.image_file_path)
                mask_file = os.path.join(scan_path, RHFlairDataset.mask_file_path)
                json_file = os.path.join(scan_path, RHFlairDataset.json_path)
                volumes_file = os.path.join(scan_path, RHFlairDataset.volumes_csv_path)

                with open(json_file) as f:
                    json_f = json.load(f)
                    if not "MRAcquisitionType" in json_f:
                        skipped_because_no_info_about_acquisition_type.append(scan_path)
                        continue
                    elif json_f["MRAcquisitionType"] == "2D":
                        skipped_because_2D.append(scan_path)
                        continue

                    if "SliceThickness" not in json_f:
                        skipped_because_no_slice_thickness.append(scan_path)
                        continue
                    elif json_f["SliceThickness"] > 2.0:
                        skipped_because_slice_thickness_too_high.append(scan_path)
                        continue
                    
                    MRAcquisitionType.add(json_f["MRAcquisitionType"])

                volume_df = pd.read_csv(volumes_file)
                assert (volume_df.iloc[:, 0] == "FLAIR").all()
                col_0s.add(volume_df.columns[0])
                volume_df = volume_df.drop(volume_df.columns[0], axis=1)

                if volume_df.shape[0] != 1:
                    #too_many_volumes.append(scan_path)
                    # Use the first row (rows are equivalent)
                    volume_df = volume_df.drop(volume_df.index[range(1, volume_df.shape[0])], axis=0)

                assert volume_df.shape[0] == 1, "CSV should contain only one entry"

                if volume_df.isna().values.any() or volume_df.values.astype(float).__le__(10).any():
                    skipped_because_of_volumes.append(scan_path)
                    continue


                # sum over ventricle columns
                ventricle_volume = volume_df[VENTRICLE_KEYS].sum(axis=1)[0]
                brain_volume = volume_df[BRAIN_SIZE_KEY][0]

                relative_volume = ventricle_volume/brain_volume

                self.datalist.append({
                    "image": image_file,
                    "mask": mask_file,
                    "volume": relative_volume
                })
                volumes.append(relative_volume)

                used_at_least_once = True
                if len(self.datalist) >= size: break
            if used_at_least_once:
                IDs_used.append(rel_patient_path)
            if len(self.datalist) >= size: break

        self.analyze_raw_volumes()

        self.datalist, self.max_volume, self.min_volume = normalize_volumes(volumes, self.datalist, max_volume, min_volume)  

        #LOGGER.warning(f"skipped_because_2D={pprint.pformat(skipped_because_2D)}")
        #LOGGER.warning(f"skipped_because_of_volumes={pprint.pformat(skipped_because_of_volumes)}")
        #LOGGER.warning(f"skipped_because_no_info_about_acquisition_type={pprint.pformat(skipped_because_no_info_about_acquisition_type)}")
        #LOGGER.warning(f"too_many_volumes={pprint.pformat(too_many_volumes)}")


        LOGGER.warning(f"skipped_because_2D={len(skipped_because_2D)}")
        LOGGER.warning(f"skipped_because_of_volumes={len(skipped_because_of_volumes)}")
        LOGGER.warning(f"skipped_because_no_info_about_acquisition_type={len(skipped_because_no_info_about_acquisition_type)}")
        LOGGER.warning(f"too_many_volumes={len(too_many_volumes)}")
        LOGGER.warning(f"skipped_because_no_slice_thickness={len(skipped_because_no_slice_thickness)}")
        LOGGER.warning(f"skipped_because_slice_thickness_too_high={len(skipped_because_slice_thickness_too_high)}")
        LOGGER.info(f"Acquisition Types used: {MRAcquisitionType}")

        print("Generate data list")
        data = self._generate_data_list()
        print("Generated data list")

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


dataset_map = {
    "RH_FLAIR": (RHFlairDataset, 
                 {
                    "num_workers": 16,
                    "dataset_path": FLAIR_DATASET_DIRECTORY,

                 }
                ),
    "LDM100K": (SyntheticLDM100K,
                 {
                    "num_workers": 6,
                    "dataset_path": os.path.join(DATA_DIRECTORY, "LDM_100k"),
                 }
                ),
}

def load_dataset_from_config(run_config, section, transforms):
      assert "dataset" in run_config, "Expected key 'dataset' in run_config"
      #assert "target_data" in run_config, "Expected key 'target_data' in run_config"
      assert "dataset_size" in run_config, "Expected key 'dataset_size' in run_config"
      assert "oversample_large_ventricles" in run_config, "Expected key 'oversample_large_ventricles' in run_config"

      if isinstance(run_config["dataset"], str):
        assert run_config["dataset"] in dataset_map, "Dataset name is not supported"
        assert not isinstance(run_config["dataset_size"], Iterable), "If single dataset, need single dataset size"

        dataset_constructor, kwargs = dataset_map[run_config["dataset"]]      
            
        return dataset_constructor(
                section=section,
                size=run_config["dataset_size"],
                cache_rate=0.5,  # you may need a few Gb of RAM... Set to 0 otherwise
                seed=0,
                transform=transforms,
                **kwargs,
        )
      elif isinstance(run_config["dataset"], Iterable):
        assert isinstance(run_config["dataset_size"], Iterable), "If multiple datasets, need multiple dataset sizes"
        assert len(run_config["dataset_size"]) == len(run_config["dataset"]), "Equal length expected"
        assert isinstance(transforms, Iterable)
        assert len(transforms) == len(run_config["dataset"])
        for dataset_instance in run_config["dataset"]:
            assert dataset_instance in dataset_map, "Dataset name is not supported"
        from torch.utils.data import ConcatDataset
        datasets = []
        for i, dataset_instance in enumerate(run_config["dataset"]):
            size = run_config["dataset_size"][i]

            dataset_constructor, kwargs = dataset_map[dataset_instance]      
            
            LOGGER.info(f"Loading dataset {dataset_instance}")

            max_volume = None
            min_volume = None
            if i > 0:
                max_volume = datasets[0].max_volume
                min_volume = datasets[0].min_volume
                

            DS = dataset_constructor(
                    section=section,
                    size=size,
                    cache_rate=0.5,  # you may need a few Gb of RAM... Set to 0 otherwise
                    seed=0,
                    transform=transforms[i],
                    max_volume=max_volume,
                    min_volume=min_volume,
                    **kwargs,
            )

            datasets.append(DS)

        return ConcatDataset(datasets)







from collections.abc import Iterable


def get_default_transforms(run_config, guidance):
    def get_single_DS_transform(DS_str, run_config, guidance):
        if DS_str == "RH_FLAIR":
            return get_default_transforms_RH_FLAIR(target_spacing=run_config["target_spacing"], crop_size=run_config["input_image_crop_roi"], guidance=guidance)
        elif DS_str == "LDM100K":
            return get_default_transforms_LDM100k(target_spacing=run_config["target_spacing"] if "target_spacing" in run_config else run_config["input_image_downsampling_factors"],
                                                crop_size=run_config["input_image_crop_roi"], guidance=guidance)
        else:
            raise Exception(f"Dataset {DS_str} not supported for deault transforms")

    if isinstance(run_config["dataset"], str): # singleton
        return get_single_DS_transform(run_config["dataset"], run_config, guidance)
    elif isinstance(run_config["dataset"], Iterable):
        transforms_train = []
        transforms_valid = []
        for DS_str in run_config["dataset"]:
            train, val = get_single_DS_transform(DS_str, run_config, guidance)
            transforms_train.append(train)
            transforms_valid.append(val)

        return transforms_train, transforms_valid
    else:
        raise Exception("Config input type not supported for default transforms")


def get_default_transforms_RH_FLAIR(target_spacing, crop_size, guidance):
    LOGGER.info("Using default transform from RH_FLAIR")

    base_transforms = [
            transforms.LoadImaged(keys=["image", "mask"]),
            transforms.EnsureChannelFirstd(keys=["image", "mask"]),
            transforms.EnsureTyped(keys=["image", "mask"]),
            transforms.Lambdad(keys=["image"], func=lambda x: x[0, :, :, :].unsqueeze(0)), # select first channel if multiple channels occur

            transforms.Orientationd(keys=["image", "mask"], axcodes="IAR"),# IPR # IAR # axcodes="RAS"
            transforms.Spacingd(keys=["mask"], pixdim=target_spacing, mode=("nearest")),
            transforms.Spacingd(keys=["image"], pixdim=target_spacing, mode=("bilinear")),

            transforms.Lambda(func=get_crop_around_mask_center(crop_size)),
            transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),
            transforms.Lambdad(keys=["volume"], 
                            func = lambda x: (torch.tensor([x, 1.], dtype=torch.float32) if guidance else torch.tensor([x], dtype=torch.float32)).unsqueeze(0)),
        ]

    train_transforms = transforms.Compose([
            *base_transforms,
            # use random conditioning value if unconditioned case
            transforms.RandLambdad(keys=["volume"], prob=0.2 if guidance else 0, func=lambda x: torch.tensor([random.random(), 0.]).unsqueeze(0)),
            ])

    valid_transforms = transforms.Compose(base_transforms)

    return train_transforms, valid_transforms

def get_default_transforms_LDM100k(target_spacing, crop_size, guidance):
    base_transforms = [
        transforms.LoadImaged(keys=["mask", "image"]),
        transforms.EnsureChannelFirstd(keys=["mask", "image"]),
        transforms.EnsureTyped(keys=["mask", "image"]),
        transforms.Orientationd(keys=["mask", "image"], axcodes="IPL"), # axcodes="RAS"
        transforms.Spacingd(keys=["mask"], pixdim=target_spacing, mode=("nearest")),
        transforms.Spacingd(keys=["image"], pixdim=target_spacing, mode=("bilinear")),
        transforms.CenterSpatialCropd(keys=["mask", "image"], roi_size=crop_size),
        transforms.ScaleIntensityRangePercentilesd(keys=["image"], lower=0.5, upper=99.5, b_min=0, b_max=1, clip=True),

        # Use extra conditioning variable to signify conditioned or non-conditioned case
        transforms.Lambdad(keys=["volume"], 
                           func = lambda x: (torch.tensor([x, 1.], dtype=torch.float32) if guidance 
                                             else torch.tensor([x], dtype=torch.float32)).unsqueeze(0)),
    ]


    train_transforms = transforms.Compose([
            *base_transforms,
            # use random conditioning value if unconditioned case
            transforms.RandLambdad(keys=["volume"], prob=0.2 if guidance else 0, func=lambda x: torch.tensor([random.random(), 0.]).unsqueeze(0)),
            ])

    valid_transforms = transforms.Compose(base_transforms)

    return train_transforms, valid_transforms



class RHNPHFlairTestDataset(CacheDataset):
    def __init__(
        self,
        dataset_path,
        transform,
        size=50,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")

        self.datalist = []

        import tqdm
        from typing import List

        def find_file(files: List[str], beginning, ending, keyword):
            if beginning is not None:
                files = [file for file in files if file.startswith(beginning)]
            
            if ending is not None:
                files = [file for file in files if file.endswith(ending)]
            
            if keyword is not None:
                files = [file for file in files if keyword in file]
            
            if len(files) == 0:
                LOGGER.warning(f"Could not find [{beginning}, {ending}, {keyword}] in {files}")
                return None

            
            if len(files) > 1:
                LOGGER.warning(f"Multipl files for [{beginning}, {ending}, {keyword}] in {files}")
            
            return files[0]


        for rel_patient_path in tqdm.tqdm(os.listdir(dataset_path), desc="Collecting files and volumes..."):
            patient_path = os.path.join(dataset_path, rel_patient_path)

            for rel_scan_path in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, rel_scan_path)
                all_files = os.listdir(scan_path)

                mask_file = find_file(all_files, beginning=None, ending=".nii", keyword="Annika")
                if mask_file is None:
                    continue

                image_file = find_file(all_files, beginning=None, ending="FLAIR.nii.gz", keyword=None)
                if image_file is None:
                    continue

                synthseg_mask_file = find_file(all_files, beginning=None, ending="FLAIR_synthseg.nii.gz", keyword=None)
                
                if synthseg_mask_file is None:
                    LOGGER.warn("No synthseg mask found!")


                self.datalist.append({
                    "image": os.path.join(scan_path, image_file),
                    "mask": os.path.join(scan_path, mask_file),
                    "case_identifier": f"{rel_patient_path}-{rel_scan_path}",
                    "synthseg_mask": os.path.join(scan_path, synthseg_mask_file)
                })

                if len(self.datalist) >= size: break
            if len(self.datalist) >= size: break


        print("Generate data list")
        data = [x for x in self.datalist]
        print("Generated data list")

        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )




class RH3DFlairTestDataset(CacheDataset):
    IGNORE_LIST = []

    def __init__(
        self,
        dataset_path,
        transform,
        size=50,
        cache_num=sys.maxsize,
        cache_rate=1.0,
        num_workers=0,
    ):
        if not path.isdir(dataset_path):
            raise ValueError("Root directory root_dir must be a directory.")

        self.datalist = []

        import tqdm
        from typing import List

        def find_file(files: List[str], beginning, ending, keyword):
            if beginning is not None:
                files = [file for file in files if file.startswith(beginning)]
            
            if ending is not None:
                files = [file for file in files if file.endswith(ending)]
            
            if keyword is not None:
                files = [file for file in files if keyword in file]
            
            if len(files) == 0:
                LOGGER.warning(f"Could not find [{beginning}, {ending}, {keyword}] in {files}")
                return None

            
            if len(files) > 1:
                LOGGER.warning(f"Multipl files for [{beginning}, {ending}, {keyword}] in {files}")
            
            return files[0]

        prefix_segmentations = "derivatives/delineations"
        prefix_images = "rawdata"
        image_additional_subpath = "anat"

        for rel_patient_path in tqdm.tqdm(os.listdir(os.path.join(dataset_path, prefix_segmentations)), desc="Collecting files and volumes..."):
            patient_path = os.path.join(dataset_path, prefix_segmentations, rel_patient_path)
            
            assert len(os.listdir(patient_path)) == 1

            if rel_patient_path in RH3DFlairTestDataset.IGNORE_LIST:
                continue

            for rel_scan_path in os.listdir(patient_path):
                scan_path = os.path.join(patient_path, rel_scan_path)
                all_files = os.listdir(scan_path)

                mask_file = find_file(all_files, beginning=None, ending=None, keyword="expert")

                if mask_file is None:
                    mask_file = find_file(all_files, beginning=None, ending=None, keyword="student")
                    if mask_file is None:
                        LOGGER.warn(f"Unexpected missing segmentation in {all_files}")
                
                image_folder = os.path.join(dataset_path, prefix_images, rel_patient_path, rel_scan_path, image_additional_subpath)

                image_file = find_file(os.listdir(image_folder), beginning=None, ending="FLAIR.nii.gz", keyword=None)
                synthseg_mask_file = find_file(os.listdir(image_folder), beginning=None, ending="FLAIR_synthseg_mask.nii.gz", keyword=None)

                if image_file is None:
                    LOGGER.warn("Unexpected missing image")
                    raise Exception("Unexpected missing image")

                if synthseg_mask_file is None:
                    raise Exception("Unexpected missing synthseg mask")

                self.datalist.append({
                    "image": os.path.join(image_folder, image_file),
                    "mask": os.path.join(scan_path, mask_file),
                    "synthseg_mask": os.path.join(image_folder, synthseg_mask_file),
                    "case_identifier": f"{rel_patient_path}-{rel_scan_path}",
                })

                if len(self.datalist) >= size: break
            if len(self.datalist) >= size: break


        data = [x for x in self.datalist]

        super().__init__(
            data,
            transform,
            cache_num=cache_num,
            cache_rate=cache_rate,
            num_workers=num_workers,
        )
