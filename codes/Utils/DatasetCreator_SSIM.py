import skimage.metrics as ski
import torchio as tio
from pathlib import Path
import os

path_main = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
path_reg = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/SSIM/"

regPath = Path(path_reg)
mainPath = Path(path_main)
f = open("details_2.txt", "w+")
f.write("Filename,Sigma,SSIM\r\n")
for reg_fileName in regPath.glob("*.nii.gz"):
    regFilePath = reg_fileName
    temp = str(reg_fileName.name)
    si = reg_fileName.name.split(".nii.gz")[0].split("-")[-1]
    sigma = str("-" + reg_fileName.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
    reg_fileName = temp.replace(sigma, "")
    for main_file_name in mainPath.glob("*T1*.nii.gz"):
        if main_file_name.name == str(reg_fileName + ".nii.gz"):
            subject_reg = tio.Subject(image=tio.ScalarImage(regFilePath))
            subject_main = tio.Subject(image=tio.ScalarImage(main_file_name))
            P = subject_reg["image"][tio.DATA].squeeze(0).float()
            GT = subject_main["image"][tio.DATA].squeeze(0).float()
            GT = (GT - GT.min()) / (GT.max() - GT.min())
            P = (P - P.min()) / (P.max() - P.min())
            ssim = ski.structural_similarity(GT.numpy(), P.numpy())
            tempStr = str(reg_fileName + "-" + str(ssim) + ".nii.gz")
            temp = str(reg_fileName) + "," + str(si) + "," + str(ssim) + "\r\n"
            print(temp)
            f.write(temp)
            os.rename(regFilePath, Path(regFilePath.parent, tempStr))

f.close()
