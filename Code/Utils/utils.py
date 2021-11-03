import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def getSubjects(inpPath):
    output = []
    selected = []
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        temp = str(file_name.name)
        ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(ssim, "")
        fileName = fileName.split("_")[0]
        if fileName not in output:
            output.append(fileName)
            temp = []
            for sub_file_name in sorted(inpPath.glob(str("*" + fileName + "*"))):
                temp.append(float(sub_file_name.name.split(".nii.gz")[0].split("-")[-1]))
            temp = returnClass(4, np.array(temp))
            temp = temp.astype(int)
            bin_count = np.bincount(temp)
            if len(bin_count) == 4 and bin_count[0] == bin_count[1] == bin_count[2] == bin_count[3]:
                selected.append(fileName)
    print("Total Number of Subjects in Dataset:", len(output))
    print("Total Number of Subjects Selected  :", len(output))
    return output, selected


def returnClass(no_of_class, array):
    class_intervals = 1 / no_of_class
    array[array <= 0] = 0
    array[array >= 1] = no_of_class - 1
    for i in range(no_of_class - 1, -1, -1):
        array[(array > (class_intervals * i)) & (array <= (class_intervals * (i + 1)))] = i
    return array


def plot_bar(array, no_of_class=4):
    array = returnClass(no_of_class, np.array(array))
    array = array.astype(int)
    bin_count = np.bincount(array)
    print("Bin Count : ", bin_count)
    fig, ax = plt.subplots()
    # langs = ['0', '1', '2', '3']
    langs = [str(x) for x in range(0, no_of_class)]
    print("Y Tiles   : ", langs)
    bin = bin_count
    ax.set(title="SSIM distribution",
           xlabel="SSIM Classes",
           ylabel="Count")
    ax.bar(langs, bin)
    plt.show()


def disp(imgReg, imgOrig, text):
    slices = []
    slices.append(imgReg[:, 100:101, :, :].squeeze(0))
    slices.append(imgOrig[:, 100:101, :, :].squeeze(0))
    fig, axes = plt.subplots(1, len(slices))
    for j, slice in enumerate(slices):
        axes[j].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Labels: " + text)
    plt.show()


def display_dataset_dist(inpPath, no_of_class=4):
    for i in range(no_of_class - 1, -1, -1):
        print(i)
    inpPath = Path(inpPath)
    ssims = []
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        ssim = float(file_name.name.split(".nii.gz")[0].split("-")[-1])
        ssims.append(ssim)
    plot_bar(np.array(ssims), no_of_class=no_of_class)


def checkFilePaths():
    print(os.getcwd())
    # os.mkdir('model_W1')
