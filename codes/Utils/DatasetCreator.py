from __future__ import division
from __future__ import print_function

import sys
import csv
import random
from pathlib import Path
import matplotlib.pyplot as plt
import torch
import torchio as tio

# sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Utils/')
import pytorch_ssim


def singleSubjectCreation(fileName, inpPath, mainPath, outPath, total_subjects,
                          no_of_subjects, class_range_low, class_range_high):
    inpPath = Path(inpPath)
    mainPath = Path(mainPath)
    buffer = 0
    csv_update_counter = total_subjects + 1
    imgOrig = tio.ScalarImage(mainPath / str(fileName + ".nii.gz"))[tio.DATA]
    imgOrig = (imgOrig - imgOrig.min()) / (imgOrig.max() - imgOrig.min())
    for file_name in sorted(inpPath.glob(str("*" + fileName + "*"))):
        if no_of_subjects < 1:
            break
        imgReg = tio.ScalarImage(file_name)[tio.DATA]
        imgReg = (imgReg - imgReg.min()) / (imgReg.max() - imgReg.min())
        for axis in range(0, 3):
            if no_of_subjects < 1:
                break
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

            for slice in range(buffer, len(imgReg_op.squeeze()) - buffer):
                if no_of_subjects < 1:
                    break
                # Get the slice image of the subject
                image = imgReg_op[:, slice:(slice + 1), :, :].squeeze(0).squeeze(0)
                # Check if all classes are filled up
                if class_range_low <= ssim[slice].item() <= class_range_high:
                    filename = fileName + "_" + str(csv_update_counter) + "-" + str(ssim[slice].item()) + '.nii.gz'
                    out_filename = outPath + filename
                    print("Saving file: ", out_filename, "  Counter : ", no_of_subjects)
                    temp = tio.ScalarImage(tensor=image.unsqueeze(2).unsqueeze(0))
                    temp.save(out_filename, squeeze=True)
                    csv_update_counter += 1
                    no_of_subjects -= 1


def RandDatasetCreation(inpPath, mainPath, outPath):
    inpPath = Path(inpPath)
    main_Path = Path(mainPath)
    output = []
    random_seed = 42
    samples_per_subject = 100
    random.seed(random_seed)
    buffer = 20
    no_of_retry = 10
    idx = flag = 0
    ctr_1 = ctr_2 = ctr_3 = ctr_4 = retry = 0
    c_1 = c_2 = c_3 = c_4 = []
    first_time_in_loop = True  # Used for entry check -
    no_files_perSubject = 100
    csvFileName = 'result.csv'

    # Get the value for number of files per class
    samples_per_class = int(samples_per_subject / 4)

    created = []
    OP = Path(outPath)
    for file_name in sorted(OP.glob("*.nii.gz")):
        temp = str(file_name.name)
        ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(ssim, "")
        fileName = fileName.split("_")[0]
        if fileName not in created:
            created.append(fileName)

    for file_name in sorted(main_Path.glob("*T1*.nii.gz")):
        output.append(file_name.name.replace(".nii.gz", ""))

    print("Number of Subjects in Main list   : ", len(output))
    print("Number of Subjects in Created list: ", len(created))

    output = list(set(output) - set(created))
    print("Number of Subjects in New list: ", len(output))


    with open(outPath + csvFileName, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['Filename', 'SSIM', 'Axis', 'Counter'])
    idx = 0
    while idx < len(output):
        if retry >= no_of_retry:
            retry = 0
            idx += 1
        if flag == 1:  # Check - all class are equal
            idx += 1
            flag = ctr_1 = ctr_2 = ctr_3 = ctr_4 = retry = 0
            c_1 = c_2 = c_3 = c_4 = []
            print("New Subject Selected: ", output[idx], "  ID Values: ", idx)
        else:
            if first_time_in_loop:
                print("New Subject Selected: ", output[idx], "  ID Values: ", idx)
            else:
                retry += 1
                print("Iteration :", retry)  # Printing the number of iterations performed

        subject_id = output[idx]
        writeFile = []
        first_time_in_loop = False
        csv_update_counter = 1

        for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
            if flag == 1:
                break
            # Read the images - Corrupted and Original
            imgReg = tio.ScalarImage(file_name)[tio.DATA]
            imgOrig = tio.ScalarImage(main_Path / str(subject_id + ".nii.gz"))[tio.DATA]

            # Normalizing the images
            imgReg = (imgReg - imgReg.min()) / (imgReg.max() - imgReg.min())
            imgOrig = (imgOrig - imgOrig.min()) / (imgOrig.max() - imgOrig.min())

            ctr = 0  # Counter to check number of files per subject
            while ctr < no_files_perSubject and retry < no_of_retry:
                store = True
                # Randomly select the axis of the image
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
                slice = random.randint(buffer, len(imgReg_op.squeeze()) - buffer)

                # Get the slice image of the subject
                image = imgReg_op[:, slice:(slice + 1), :, :].squeeze(0).squeeze(0)
                filename = subject_id + "_" + str(csv_update_counter) + "-" + str(ssim[slice].item()) + '.nii.gz'
                out_filename = outPath + filename

                # Check if all classes are filled up
                if 0 <= ssim[slice].item() <= .25 and ctr_1 < samples_per_class:
                    c_1.append([image, out_filename, ssim[slice].item()])
                    ctr_1 += 1
                elif 0.25 < ssim[slice].item() <= .5 and ctr_2 < samples_per_class:
                    c_2.append([image, out_filename, ssim[slice].item()])
                    ctr_2 += 1
                elif 0.5 < ssim[slice].item() <= .75 and ctr_3 < samples_per_class:
                    c_3.append([image, out_filename, ssim[slice].item()])
                    ctr_3 += 1
                elif 0.75 < ssim[slice].item() <= 1.0 and ctr_4 < samples_per_class:
                    c_4.append([image, out_filename, ssim[slice].item()])
                    ctr_4 += 1
                else:
                    store = False  # Change the store flag to False - File saving turned off
                    ctr += 1  # Increase counter of files per subject

                if store:
                    writeFile.append([filename, ssim[slice].item(), axis, slice])
                    csv_update_counter += 1  # Increase counter of number of files stored
                if ctr_1 == ctr_2 == ctr_3 == ctr_4 == samples_per_class:
                    flag = 1  # Flag changed to update classes are full

        with open(outPath + csvFileName, 'a') as f:
            writer = csv.writer(f)
            writer.writerows(writeFile)
        if len(c_1) == len(c_2) == len(c_3) == len(c_4):
            files = c_1 + c_2 + c_3 + c_4

            subjects = [row[0] for row in files]
            filenames = [row[1] for row in files]

            i = 0
            for subject in subjects:
                temp = tio.ScalarImage(tensor=subject.unsqueeze(2).unsqueeze(0))
                temp.save(filenames[i], squeeze=True)
                i += 1


out_path = "/media/hdd_storage/Budha/Dataset/SSIM/"
main_Path = "/media/hdd_storage/Budha/Dataset/Isotropic"
inp_Path = "/media/hdd_storage/Budha/Dataset/Regression"
RandDatasetCreation(inpPath=inp_Path, mainPath=main_Path, outPath=out_path)
"""fileName = "IXI002-Guys-0828-T1"
singleSubjectCreation(fileName, inp_Path, main_Path, out_path, total_subjects=99, no_of_subjects=1, class_range_low=0.0,
                      class_range_high=0.25)
"""