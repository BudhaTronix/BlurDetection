import random
from pathlib import Path

import numpy as np
import torch
import torchio as tio
from torch.utils.data.sampler import SubsetRandomSampler


def datasetCreator(disp):
    path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/SSIM/"
    inpPath = Path(path)

    subjects = []

    if disp: print("Loading Test Dataset.......")
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        subject = tio.Subject(image=tio.ScalarImage(file_name),
                              label=[float(str(file_name.name).split(".nii.gz")[0].split("-")[-1])])
        subjects.append(subject)
        break
    dataset = tio.SubjectsDataset(subjects)
    dataset = subjecttodataset(dataset)

    return dataset


def subjecttodataset(dataset):
    patch_size = (230, 230, 134)
    patch_per_vol = 1  # n_slices
    patch_qlen = patch_per_vol * 2
    sampler = tio.data.UniformSampler(patch_size)
    dataset = tio.Queue(
        subjects_dataset=dataset,
        max_length=patch_qlen,
        samples_per_volume=patch_per_vol,
        sampler=sampler,
        num_workers=0,
        # start_background=True
    )

    dataset_size = len(dataset)
    indices = list(range(dataset_size))

    train_sampler = SubsetRandomSampler(indices)
    dataset = torch.utils.data.DataLoader(dataset, batch_size=1,
                                          sampler=train_sampler, num_workers=4)

    return dataset


def ModelTest(disp=False, preCreated=False, dataset="", tollerance=0.3,
              model_path='/project/mukhopad/tmp/BlurDetection_tmp/model_weights/RESNET101_DataLoader_Reg_T1.pth'):
    if preCreated:
        dataset = subjecttodataset(dataset)
    else:
        dataset = datasetCreator(disp=disp)
    device = "cuda"
    # PATH = '../model_weights/BlurDetection_ModelWeights_SinlgeGPU_RESNET_MultiClass_DataLoader_Reg_T1.pth'
    PATH = model_path
    model = torch.load(PATH)
    model.eval()
    model.to(device)
    overall_acc = 0.0
    count = 0
    for batch in dataset:
        image_batch = batch["image"][tio.DATA].permute(1, 0, 2, 3, 4).squeeze(0)  # [64,134,230,230]
        labels_batch = batch["label"][0]
        select_orientation = random.randint(1, 1)
        if select_orientation == 1:
            image_batch = image_batch  # Coronal
        elif select_orientation == 2:
            image_batch = image_batch.permute(0, 2, 3, 1)  # Transverse
        elif select_orientation == 3:
            image_batch = image_batch.permute(0, 3, 2, 1)  # Axial

        n_correct = 0
        n_wrong = 0
        for i in range(0, len(image_batch[0])):
            with torch.no_grad():
                inputs = image_batch[:, i:(i + 1), :, :].float()
                inputs = inputs / np.linalg.norm(inputs)  # Gaussian Normalization
                outputs = model(inputs.to(device))
                abs_delta = np.abs(outputs.item() - labels_batch.item())
                # abs_delta = np.abs(outputs.squeeze() - labels_batch)
                pct = tollerance
                max_allow = np.abs(pct * labels_batch.item())
                if disp:
                    print("predicted = %0.4f  actual = %0.4f \
                     delta = %0.4f  max_allow = %0.4f : " % (outputs.item(), \
                                                             labels_batch.item(), abs_delta, max_allow), end="")
                if abs_delta < max_allow:
                    if disp: print("correct")
                    n_correct += 1
                else:
                    if disp: print("wrong")
                    n_wrong += 1

        acc = (n_correct * 1.0) / (n_correct + n_wrong)
        if disp: print("Batch Accuracy :", acc)
        count += 1
        overall_acc += acc
    if disp: print("\nOverall Accuracy of 5 Subjects:", overall_acc / count)
    return overall_acc / count

ModelTest()
