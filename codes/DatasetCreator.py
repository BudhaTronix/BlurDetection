import os
import torchio as tio
import torch
from pathlib import Path
from motion import MotionCorrupter
from ImageTransformer import transform_subject, transform_subject_reality
import numpy as np
import nibabel as nib
import random
from pathlib import Path
from torchvision import transforms as transforms

def create_subjectlist(inpPath):
    l = 1000
    b = 1000
    h = 1000
    inpPath = Path(inpPath)
    """"
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        testdata = (nib.load(file_name)).get_fdata()
        if (len(testdata) < l): l = len(testdata)
        if (len(testdata[0]) < b): b = len(testdata[0])
        if (len(testdata[0][0]) < h): h = len(testdata[0][0])
    print("Maximum Size allowed:", l, b, h)"""
    #transform = transforms.ToTensor()
    subjects = []
    #for file_name in sorted(inpPath.glob("*.nii.gz")):
    print("\n Loading Dataset")
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        #file_path = os.path.join(inpPath, file_name)
        subject = tio.Subject(
            #image = tio.ScalarImage(file_name),
            #image = tio.ScalarImage(tensor=(transform((nib.load(file_name)).get_fdata()[:l,:b,:h]).unsqueeze(0))),
            image = tio.ScalarImage(file_name),
            filename = str(file_name.name),
        )

        subjects.append(subject)

    return subjects

def datasetCreator_mode1(in_path,out_path):
    in_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
    out_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed/"
    subjects_dataset = create_subjectlist(in_path)
    print("\n Corrupting Dataset")
    i=1

    for s in subjects_dataset:
        name = s["filename"]
        select = random.randint(0, 4)
        img = transform_subject(select, s["image"][tio.DATA].squeeze(1))[0:1, :, :, :].float()
        data =  np.float32(np.abs(img.cpu().numpy().squeeze()))
        nib.save(nib.Nifti1Image(data, None), out_path +str(name.split(".nii.gz")[0])+"-"+str(select)+'.nii.gz')
        print('...corruption ', str(i+1), " done for ... ", s)

        break


def datasetCreator_mode2(in_path,out_path):
    in_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
    out_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed/"
    subjects_dataset = create_subjectlist(in_path)
    print("\n Corrupting Dataset")
    i = 0
    l = 134
    b = 230
    h = 230
    for s in subjects_dataset:
        name = s["filename"]
        select = random.randint(0, 4)
        img = transform_subject_reality(select, s["image"][tio.DATA].squeeze(1))[0:1, :b, :h, :l].float()
        #data =  np.float32(np.abs(img.cpu().numpy().squeeze()))
        #nib.save(nib.Nifti1Image(data, None), out_path +str(name.split(".nii.gz")[0])+"-"+str(select)+'.nii.gz')
        temp = tio.ScalarImage(tensor=img)
        temp.save(out_path +str(name.split(".nii.gz")[0])+"-"+str(select)+'.nii.gz',squeeze=True)
        print('...corruption ', str(i+1), " done for ... ", s)

datasetCreator_mode2("","")