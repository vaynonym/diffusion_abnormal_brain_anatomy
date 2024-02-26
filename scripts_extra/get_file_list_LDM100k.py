import csv
import os.path as path

FOLDER_PATH = "/homes/tim/thesis/own_experiments/data/LDM_100k"
PARTICIPANTS_FILE = path.join(FOLDER_PATH, "participants.tsv")
RESULT_FILE = path.join(FOLDER_PATH, "file_paths.txt")
SAMPLE_SIZE = 10000


with open(PARTICIPANTS_FILE) as participantsCSV:
    with open(RESULT_FILE, mode="w") as result_file:
        reader = csv.reader(participantsCSV, delimiter='\t')
        next(reader) # skip header row
        count = 0
        for line in reader:
            sub_folder_name = line[0]
            file_name = sub_folder_name + "_T1w.nii.gz"
            result_file.write(path.join(FOLDER_PATH, "rawdata", sub_folder_name, "anat", file_name) + "\n")

            count += 1
            if count >= SAMPLE_SIZE:
                break
