from __future__ import division
from __future__ import print_function

import sys
import csv
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchio as tio

sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Utils/')
import pytorch_ssim


def disp(imgReg, imgOrig, text):
    slices = []
    slices.append(imgReg[:, 100:101, :, :].squeeze(0))
    slices.append(imgOrig[:, 100:101, :, :].squeeze(0))
    fig, axes = plt.subplots(1, len(slices))
    for j, slice in enumerate(slices):
        axes[j].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Labels: " + text)
    plt.show()


def RandDatasetCreation(inpPath, mainPath, outPath):
    inpPath = Path(inpPath)
    main_Path = Path(mainPath)
    output = []
    random_seed = 42
    samples_per_subject = 40
    random.seed(random_seed)
    buffer = 20
    no_of_retry = 10
    idx = flag = 0
    ctr_1 = ctr_2 = ctr_3 = ctr_4 = retry = 0
    c_1 = c_2 = c_3 = c_4 = []
    q = 0
    no_files_perSubject = 200
    csvFileName = 'result.csv'

    for file_name in sorted(main_Path.glob("*T1*.nii.gz")):
        output.append(file_name.name.replace(".nii.gz", ""))

    with open(outPath + csvFileName, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'SSIM', 'Axis', 'Counter'])

    while idx < len(output):
        if retry >= no_of_retry:
            l = (ctr_1, ctr_2, ctr_3, ctr_4)
            print("Subject :", output[idx], "   Values :", ctr_1, ctr_2, ctr_3, ctr_4)
            # get index of smallest item in list
            X = min(l) # Getting the minimum no of items of each class
            if X == 0:
                print("Subject Removed")  # Subject is removed if one of the classes have 0 items
            else:
                c_1 = c_1[0:X]
                c_2 = c_2[0:X]
                c_3 = c_3[0:X]
                c_4 = c_4[0:X]
            retry = 0
            idx += 1
        if ctr_1 == ctr_2 == ctr_3 == ctr_4 == int(samples_per_subject / 4) or flag == 1:  # Check - all class are equal
            idx += 1
            flag = ctr_1 = ctr_2 = ctr_3 = ctr_4 = retry = 0
            c_1 = c_2 = c_3 = c_4 = []
            print("New Subject Selected: ", output[idx])
        else:
            if not q == 0:
                retry += 1
                print("Iteration :", retry)  # Printing the number of iterations performed

        subject_id = output[idx]
        writeFile = []
        q = 1
        j = 1

        for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
            if flag == 1:
                break

            imgReg = tio.ScalarImage(file_name)[tio.DATA]
            imgOrig = tio.ScalarImage(main_Path / str(subject_id + ".nii.gz"))[tio.DATA]
            imgReg = (imgReg - imgReg.min()) / (imgReg.max() - imgReg.min())
            imgOrig = (imgOrig - imgOrig.min()) / (imgOrig.max() - imgOrig.min())

            ctr = 0
            while not (ctr_1 == ctr_2 == ctr_3 == ctr_4 == int(
                    samples_per_subject / 4)) and ctr < no_files_perSubject and retry < no_of_retry:
                store = True
                # Randomly select the axis
                axis = random.randint(0, 2)
                if axis == 0:
                    imgReg_op = imgReg
                    imgOrig_op = imgOrig
                if axis == 1:
                    imgReg_op = imgReg.permute(0, 2, 3, 1)
                    imgOrig_op = imgOrig.permute(0, 2, 3, 1)
                if axis == 2:
                    imgReg_op = imgReg.permute(0, 3, 2, 1)
                    imgOrig_op = imgOrig.permute(0, 3, 2, 1)
                if torch.cuda.is_available():
                    imgReg_ssim = imgReg_op.cuda()
                    imgOrig_ssim = imgOrig_op.cuda()

                # Calculate the SSIM
                ssim = pytorch_ssim.ssim(imgOrig_ssim.double(), imgReg_ssim.double()).mean(0).mean(1).mean(
                    1).detach().cpu()

                # Randomly select a slice from the images
                slice = random.randint((0 + buffer), len(imgReg_op.squeeze()) - buffer)

                subject = imgReg_op[:, slice:(slice + 1), :, :].squeeze(0).squeeze(0)
                filename = subject_id + "_" + str(j) + "-" + str(ssim[slice].item()) + '.nii.gz'
                out_filename = outPath + filename

                if 0 <= ssim[slice].item() <= .25 and ctr_1 < int(samples_per_subject / 4):
                    c_1.append([subject, out_filename, ssim[slice].item()])
                    ctr_1 += 1
                elif 0.25 < ssim[slice].item() <= .5 and ctr_2 < int(samples_per_subject / 4):
                    c_2.append([subject, out_filename, ssim[slice].item()])
                    ctr_2 += 1
                elif 0.5 < ssim[slice].item() <= .75 and ctr_3 < int(samples_per_subject / 4):
                    c_3.append([subject, out_filename, ssim[slice].item()])
                    ctr_3 += 1
                elif 0.75 < ssim[slice].item() <= 1.0 and ctr_4 < int(samples_per_subject / 4):
                    c_4.append([subject, out_filename, ssim[slice].item()])
                    ctr_4 += 1
                else:
                    store = False
                    ctr += 1

                if store:
                    writeFile.append([filename, ssim[slice].item(), axis, slice])
                    j += 1
                if ctr_1 == ctr_2 == ctr_3 == ctr_4 == int(samples_per_subject / 4):
                    flag = 1

        with open(outPath + csvFileName, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(writeFile)
        files = c_1 + c_2 + c_3 + c_4

        subjects = [row[0] for row in files]
        filenames = [row[1] for row in files]

        i = 0
        for subject in subjects:
            temp = tio.ScalarImage(tensor=subject.unsqueeze(2).unsqueeze(0))
            temp.save(filenames[i], squeeze=True)
            i += 1


out_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/TestDataset/"
main_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
inp_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/SSIM/"
RandDatasetCreation(inpPath=inp_Path, mainPath=main_Path, outPath=out_path)
