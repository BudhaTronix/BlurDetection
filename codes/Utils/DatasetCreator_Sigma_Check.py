from tqdm import tqdm
from pathlib import Path
import torchio as tio
import time
import numpy as np

from motion import MotionCorrupter

output = []
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Regression_T1/"
in_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
out_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Regression_T1/"

mainPath = Path(in_path)
inpPath = Path(path)
for file_name in sorted(inpPath.glob("*.nii.gz")):
    temp = str(file_name.name)
    fileName = temp
    if fileName not in output:
        output.append(fileName)

print("Total Files : ", len(output))

output = []
for file_name in sorted(inpPath.glob("*T1*.nii.gz")):
    temp = str(file_name.name)
    sigma = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
    fileName = temp.replace(sigma, "")
    if fileName not in output:
        output.append(fileName)

print("Total Subjects : ", len(output))
#####################################################################################################################################

for subject in output:
    c = 0
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        temp = str(file_name.name)
        sigma = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(sigma, "")
        if fileName == subject:
            c = c + 1
    print(subject, " - ", c)
    if c <= 10:
        no_of_corr = 10 - c
        for file_name in sorted(mainPath.glob("*T1*.nii.gz")):
            if file_name.name == str(subject) + ".nii.gz":
                for i in range(no_of_corr):
                    print("Starting to corrupt ", file_name.name, "    Iteration Number - ", i, "/", no_of_corr)
                    subject = tio.Subject(image=tio.ScalarImage(file_name))
                    sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))
                    moco = MotionCorrupter(mode=2, n_threads=5, mu=0.0, sigma=sigma, random_sigma=False)
                    transforms = [tio.Lambda(moco.perform, p=1)]
                    transform = tio.Compose(transforms)
                    img = transform(subject["image"][tio.DATA])[1:2, :, :, :]
                    temp = tio.ScalarImage(tensor=img)
                    temp.save(out_path + str(file_name.name.split(".nii.gz")[0]) + "-" + str(sigma[0]) + '.nii.gz',
                              squeeze=True)
