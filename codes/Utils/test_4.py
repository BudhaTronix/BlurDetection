from tqdm import tqdm
from pathlib import Path
import torchio as tio
import time
import numpy as np


output = []
path = "/media/hdd_storage/Budha/Dataset/Isotropic"
inpPath = Path(path)
for file_name in sorted(inpPath.glob("*.nii.gz")):
    temp = str(file_name.name)
    #sigma = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
    #fileName = temp.replace(sigma, "")
    fileName = temp
    if fileName not in output:
        output.append(fileName)


print("Total Files : ", len(output))

output = []
for file_name in sorted(inpPath.glob("*.nii.gz")):
    temp = str(file_name.name)
    sigma = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
    fileName = temp.replace(sigma, "")
    if fileName not in output:
        output.append(fileName)

print("Total Subjects : ", len(output))