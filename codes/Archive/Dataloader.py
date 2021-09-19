from __future__ import division
from __future__ import print_function

import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torchio as tio


def datasetCreation(inpPath, val_split=0.5, batch_size=32, debug=True, plot=True):
    inpPath = Path(inpPath)
    l = w = []
    random_seed = 42
    random.seed(random_seed)

    files = []
    labels = []
    i = 0
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        imgReg = tio.ScalarImage(file_name)[tio.DATA].permute(0, 3, 1, 2)
        ssim = float(file_name.name.split(".nii.gz")[0].split("-")[-1])
        l.append(len(imgReg.squeeze()))
        w.append(len(imgReg.squeeze()[0]))
        files.append(imgReg)
        labels.append(ssim)
        i += 1
        if i == 64:
            break

    # Define the transformation required to the subjects
    transform = tio.CropOrPad((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w))))))

    # TODO: Introduce random shuffling to the main dataset

    for i in range(0, len(files)):
        files[i] = transform(files[i])

    # Training Validation Split
    val_files = files[0:int(len(files) * val_split)]
    val_labels = labels[0:int(len(labels) * val_split)]

    train_files = files[int(len(files) * val_split):len(files)]
    train_labels = labels[int(len(labels) * val_split):len(labels)]

    # Training Validation Dataset Creation
    train_dataset = torch.utils.data.Dataset((train_files, train_labels))
    val_dataset = torch.utils.data.Dataset(val_files, val_labels)

    if debug:
        print("\n#################### RETRIEVING INFORMATION ####################")
        print("Total Subjects: ", len(files))
        print("Subjects in Training: ", len(train_files))
        print("Subjects in Validation: ", len(val_files))
        print("\n#################### LOADING DATASET ####################")

    train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    dataloader = [train, val]

    if plot:
        labels1 = [row[1] for row in train_files]
        fig, ax = plt.subplots()
        ax.set(title="SSIM distribution - Training",
               xlabel="SSIM",
               ylabel="Count")
        ax.hist(labels1, bins=4, range=[0, 1])
        plt.show()

        labels2 = [row[1] for row in val_files]
        fig, ax = plt.subplots()
        ax.set(title="SSIM distribution - Validation",
               xlabel="SSIM",
               ylabel="Count")
        ax.hist(labels2, bins=4, range=[0, 1])
        plt.show()

        labels3 = labels1 + labels2
        fig, ax = plt.subplots()
        ax.set(title="SSIM distribution - Total",
               xlabel="SSIM",
               ylabel="Count")
        ax.hist(labels3, bins=4, range=[0, 1])
        plt.show()

    return dataloader


"""inp_Path = str("/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Dataset/")
dataset = datasetCreation(inpPath=inp_Path)"""
