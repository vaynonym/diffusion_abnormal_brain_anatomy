from src.datasets import get_bucket_counts
from src.directory_management import OUTPUT_DIRECTORY
from src.logging_util import LOGGER
import matplotlib.pyplot as plt
import numpy as np
import os.path as path

from torch.utils.data import Dataset
import pprint


def log_bucket_counts(data: Dataset):
    bucket_counts = get_bucket_counts(data)
    
    LOGGER.info(f"Bucket counts: {pprint.pformat(bucket_counts.detach().cpu().numpy())}")

    plt.bar(np.arange(0, 1.0, 0.1) + 0.05, height=bucket_counts, width=0.1)
    plt.xlabel("Normalized Volume Ratio", fontsize=16)
    plt.ylabel("Instances in Dataset", fontsize=16)
    plt.savefig(path.join(OUTPUT_DIRECTORY, "dataset_volumes_bar_chart"))
    plt.cla()
    plt.clf()