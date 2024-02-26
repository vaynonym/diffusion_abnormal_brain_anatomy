#!/usr/bin/env python3
import os

TEST_FILE_PATH = "/homes/tim/thesis/own_experiments/data/LDM_100k/file_paths.txt"
OUT_FILE_PATH = "/homes/tim/thesis/own_experiments/data/LDM_100k/output_paths.txt"
OUT_VOLUME_PATH = "/homes/tim/thesis/own_experiments/data/LDM_100k/volume_output_paths.txt"


synthseg_path = "/homes/tim/synthseg/SynthSeg/scripts/commands/SynthSeg_predict.py"

os.system(f'python3 {synthseg_path} --i {TEST_FILE_PATH} --o {OUT_FILE_PATH} --vol {OUT_VOLUME_PATH} --robust')