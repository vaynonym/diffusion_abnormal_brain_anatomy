import csv
import os.path as path

FOLDER_PATH = "/homes/tim/thesis/own_experiments/data/LDM_100k"
PARTICIPANTS_FILE = path.join(FOLDER_PATH, "participants.tsv")
RESULT_FILE_INPUT = path.join(FOLDER_PATH, "file_paths.txt")
RESULT_FILE_OUTPUT = path.join(FOLDER_PATH, "output_paths.txt")
RESULT_FILE_VOLUME_OUTPUT = path.join(FOLDER_PATH, "volume_output_paths.txt")

SAMPLE_SIZE = 10000


with open(PARTICIPANTS_FILE) as participantsCSV:
    with open(RESULT_FILE_INPUT, mode="w") as result_file_input:
        with open(RESULT_FILE_OUTPUT, mode="w") as result_file_output:
            with open(RESULT_FILE_VOLUME_OUTPUT, mode="w") as result_file_volume_output:

                reader = csv.reader(participantsCSV, delimiter='\t')
                next(reader) # skip header row
                count = 0
                for line in reader:
                    sub_folder_name = line[0]
                    input_file_name = sub_folder_name + "_T1w.nii.gz"
                    output_file_name = sub_folder_name + "_T1w.nii.gz"
                    volume_file_name = sub_folder_name + "_synthseg_volumes.csv"

                    result_file_input.write(path.join(FOLDER_PATH, "rawdata", sub_folder_name, "anat", input_file_name) + "\n")
                    result_file_output.write(path.join(FOLDER_PATH, "masks", sub_folder_name, "anat", output_file_name) + "\n")
                    result_file_volume_output.write(path.join(FOLDER_PATH, "masks", sub_folder_name, "anat", volume_file_name) + "\n")


                    count += 1
                    if count >= SAMPLE_SIZE:
                        break
