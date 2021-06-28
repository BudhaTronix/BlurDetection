import torchio as tio
from motion import MotionCorrupter
from ImageTransformer import transform_subject, transform_subject_reality
import numpy as np
import nibabel as nib
import random
from pathlib import Path
from tqdm import tqdm
from test_3 import func
from torchvision import transforms as transforms

def create_subjectlist(inpPath):
    subjects = []
    inpPath = Path(inpPath)
    print("\n Loading Dataset")
    for file_name in sorted(inpPath.glob("*T1*.nii.gz")):
        subject = tio.Subject(
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


def datasetCreator_mode2_classification(in_path,out_path):
    in_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
    out_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Classification_FullVolume/"
    subjects_dataset = create_subjectlist(in_path)
    print("\n Corrupting Dataset")
    i = 1
    for s in subjects_dataset:
        name = s["filename"]
        select = random.randint(0, 4)
        img = transform_subject_reality(select, s["image"][tio.DATA].squeeze(1))[1:2, :, :, :].float()
        temp = tio.ScalarImage(tensor=img)
        temp.save(out_path +str(name.split(".nii.gz")[0])+"-"+str(select)+'.nii.gz',squeeze=True)
        print('...corruption ', str(i), " done for ... ", s)
        i = i + 1

def datasetCreator_mode2_regression(in_path, out_path):
    in_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
    out_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Regression_T1/"

    #subjects_dataset = create_subjectlist(in_path)
    subjects_dataset = func()
    print("\n Corrupting Dataset....")
    n_threads = 5
    mu = 0.0
    for i in range(1,2):
        print('Corruption Iteration: ', str(i))
        for s in tqdm(subjects_dataset):
            name = s["filename"]
            sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))
            moco = MotionCorrupter(mode=2, n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=False)
            transforms = [tio.Lambda(moco.perform, p=1)]
            transform = tio.Compose(transforms)
            img = transform(s["image"][tio.DATA])[1:2, :, :, :]
            temp = tio.ScalarImage(tensor=img)
            temp.save(out_path + str(name.split(".nii.gz")[0]) + "-" + str(sigma[0]) + '.nii.gz', squeeze=True)



datasetCreator_mode2_regression("","")
#datasetCreator_mode2_classification("","")