from pathlib import Path
import csv
from tqdm import tqdm


def GenerateCSV(datasetPath, csv_FileName):
    dataset_Path = Path(datasetPath)

    with open(datasetPath + csv_FileName, 'w') as f:
        writer = csv.writer(f)

        for file_name in tqdm(sorted(dataset_Path.glob("*.nii.gz"))):
            ssim = float(file_name.name.split(".nii.gz")[0].split("-")[-1])
            writer.writerow([file_name.name, ssim])
