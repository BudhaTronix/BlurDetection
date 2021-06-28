import torch
from pathlib import Path
import numpy as np
import torchio as tio
from torch.utils.data.sampler import SubsetRandomSampler
from tqdm import tqdm
from torchvision import models
import random
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Regression_T1/"
inpPath = Path(path)
output = []
patch_size = (230, 230, 134)
patch_per_vol = 1  # n_slices
patch_qlen = patch_per_vol * 4
unt = 0
subjects = []
device = "cuda:3"
print("Loading Dataset.......")

for file_name in sorted(inpPath.glob("*.nii.gz")):
    #print("File Name in Train :",file_name)
    subject = tio.Subject(image=tio.ScalarImage(file_name),label=[float(str(file_name.name).split(".nii.gz")[0].split("-")[-1])])
    subjects.append(subject)

dataset = tio.SubjectsDataset(subjects)

sampler = tio.data.UniformSampler(patch_size)
dataset = tio.Queue(
    subjects_dataset=dataset,
    max_length=patch_qlen,
    samples_per_volume=patch_per_vol,
    sampler=sampler,
    num_workers=0,
    #start_background=True
)

dataset_size = len(dataset)
indices = list(range(dataset_size))

shuffle_dataset = False
random_seed = 42

if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)

train_sampler = SubsetRandomSampler(indices)

batch_size = 1
test_datasetLoader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                    sampler=train_sampler, num_workers=4)

PATH = '../model_weights/BlurDetection_ModelWeights_SinlgeGPU_RESNET_MultiClass_DataLoader_Reg_T1.pth'
model = torch.load(PATH)
#model.load_state_dict(torch.load(PATH))
model.eval()
model.to(device)

for batch in test_datasetLoader:
    image_batch = batch["image"][tio.DATA].permute(1, 0, 2, 3, 4).squeeze(0)  # [64,134,230,230]
    labels_batch = batch["label"][0]
    select_orientation = random.randint(1, 3)
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
            print()
            abs_delta = np.abs(outputs.item() - labels_batch.item())
            pct = 0.1
            max_allow = np.abs(pct * labels_batch.item())
            print("predicted = %0.4f  actual = %0.4f \
             delta = %0.4f  max_allow = %0.4f : " % (outputs.item(), \
                                                     labels_batch.item(), abs_delta, max_allow), end="")
            if abs_delta < max_allow:
                print("correct")
                n_correct += 1
            else:
                print("wrong")
                n_wrong += 1

        acc = (n_correct * 1.0) / (n_correct + n_wrong)
        print("ACCURACY :", acc)


