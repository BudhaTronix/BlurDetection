import shutil
import numpy as np
import os
from pathlib import Path

source_dir = '/media/hdd_storage/Budha/Dataset/SSIM/'
target_dir = '/media/hdd_storage/Budha/Dataset/TrainDataset/'

file_names = os.listdir(source_dir)



def returnClass(no_of_class, array):
    class_intervals = 1 / no_of_class
    array[array == 0] = 0
    array[array >= 1] = no_of_class - 1
    for i in range(no_of_class - 1, -1, -1):
        array[(array > (class_intervals * i)) & (array <= (class_intervals * (i + 1)))] = i
    return array
def viewBalancedSubjects(inpPath):
    output = []
    subject_ctr = 1
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        temp = str(file_name.name)
        ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(ssim, "")
        fileName = fileName.split("_")[0]
        if fileName not in output:
            temp = []
            tempFiles = []
            for sub_file_name in sorted(inpPath.glob(str("*" + fileName + "*"))):
                temp.append(float(sub_file_name.name.split(".nii.gz")[0].split("-")[-1]))
                tempFiles.append(sub_file_name)

            temp = returnClass(4, np.array(temp))
            temp = temp.astype(int)
            bin_count = np.bincount(temp)

            if len(bin_count) == 4 and (bin_count[0] == bin_count[1] == bin_count[2] == bin_count[3]):
                print(fileName, ":", len(temp), ", ", bin_count)
                output.append(fileName)
                print("Subject Name : ", fileName)
                print("Files associated with it: ", tempFiles)
                for fnames in tempFiles:
                    print(os.path.join(source_dir, fnames.name))
                    shutil.move(os.path.join(source_dir, fnames.name), target_dir)
                print("Subject list Updated to : ", len(output))
                print("#" * 100)



inpPath = "/media/hdd_storage/Budha/Dataset/SSIM/"
inpPath = Path(inpPath)
viewBalancedSubjects(inpPath)