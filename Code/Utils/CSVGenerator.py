import csv
import os
from pathlib import Path

from tqdm import tqdm


def checkCSV(dataset_Path, csv_FileName, subjects, overwrite=False):
    if overwrite:
        if os.path.isfile(dataset_Path + csv_FileName):
            os.remove(dataset_Path + csv_FileName)
    if not os.path.isfile(dataset_Path + csv_FileName):
        print(" CSV File missing..\nGenerating new CVS File..")
        GenerateCSV(dataset_Path=dataset_Path, csv_FileName=csv_FileName, subjects=subjects)
        print(" CSV File Created!")
    else:
        print("\n Dataset file available")


def GenerateCSV(dataset_Path, csv_FileName, subjects=None):
    with open(dataset_Path + csv_FileName, 'w') as f:
        writer = csv.writer(f)
        dataset_Path = Path(dataset_Path)
        if subjects is None:
            for file_name in tqdm(sorted(dataset_Path.glob("*.nii.gz"))):
                ssim = float(file_name.name.split(".nii.gz")[0].split("-")[-1])
                writer.writerow([file_name.name, ssim])
        else:
            for subject_id in tqdm(subjects):
                for file_name in sorted(dataset_Path.glob(str("*" + subject_id + "*"))):
                    ssim = float(file_name.name.split(".nii.gz")[0].split("-")[-1])
                    writer.writerow([file_name.name, ssim])
