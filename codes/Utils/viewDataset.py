from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from DatasetCreator import singleSubjectCreation


def getClassbounds(classVal):
    low = high = -1
    if classVal == 0:
        low = 0
        high = 0.25
    elif classVal == 1:
        low = 0.26
        high = 0.5
    elif classVal == 2:
        low = 0.5
        high = 0.75
    elif classVal == 3:
        low = 0.75
        high = 1

    return low, high


def getSubjects(inpPath):
    output = []
    out_path = "/media/hdd_storage/Budha/Dataset/SSIM/"
    main_Path = "/media/hdd_storage/Budha/Dataset/Isotropic"
    inp_Path = "/media/hdd_storage/Budha/Dataset/Regression"
    subject_ctr = 1
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
            print(subject_ctr, " - ", fileName, ":", len(temp), ", ", bin_count)

            if len(bin_count) == 4:
                for i in range(0, 4):
                    if bin_count[i] < 25:
                        count = 25 - bin_count[i]
                        low, high = getClassbounds(i)
                        print("Number if files to create: ", count, "  Class:", i)
                        singleSubjectCreation(fileName, inp_Path, main_Path, out_path, total_subjects=len(temp), no_of_subjects=count,
                                              class_range_low=low,
                                              class_range_high=high)
            else:
                print("Code to be updated")


            subject_ctr += 1
    return output


def returnClass(no_of_class, array):
    class_intervals = 1 / no_of_class
    array[array == 0] = 0
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
    temp_title = "SSIM distribution of " + str(no_of_class) + " Classes"
    ax.set(title=temp_title,
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


def subjectsDatasetCreated(inpPath, origPath):
    inpPath = Path(inpPath)
    origPath = Path(origPath)
    created = []
    original = []
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        temp = str(file_name.name)
        ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(ssim, "")
        fileName = fileName.split("_")[0]
        if fileName not in created:
            created.append(fileName)

    for file_name in sorted(origPath.glob("*.nii.gz")):
        temp = str(file_name.name)
        ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(ssim, "")
        fileName = fileName.split("_")[0]
        if fileName not in original:
            original.append(fileName)

    for i in range(0,582):
        print(i, " - ",created[i],original[i])
    return created, original, list(set(original) - set(created))


def subjectDeleter(inpPath):
    inpPath = Path(inpPath)
    for file_name in tqdm(sorted(inpPath.glob("*.nii.gz"))):
        ctr = 0
        for file_name_search in sorted(inpPath.glob("*.nii.gz")):
            print(file_name, file_name_search)
            if file_name_search == file_name:
                ctr += 1
        if ctr > 1:
            print("Filename: ", file_name, "  Count: ", ctr)

inpPath = "/media/hdd_storage/Budha/Dataset/SSIM/"
origPath = "/media/hdd_storage/Budha/Dataset/Regression"
#a, b, c = subjectsDatasetCreated(inpPath, origPath)
#print(len(a),len(b),len(c))
#for i in range(3,6):
#    display_dataset_dist(inpPath, no_of_class=i)
#inpPath = Path(inpPath)
#getSubjects(inpPath)
subjectDeleter(inpPath)