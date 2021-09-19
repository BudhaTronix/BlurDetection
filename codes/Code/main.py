import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
from torch.utils.data import SubsetRandomSampler
from torchvision import models
from tqdm import tqdm

from train import trainModel

sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Utils/')
from CSVGenerator import GenerateCSV
from Dataloader import CustomDataset
from Test import testModel

# GPU Setup
device_id = 6
os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
device = "cuda"

# Define Model
model = models.resnet101(pretrained=True)
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
params_to_update = model.parameters()
num_classes = 1
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, num_classes)
optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
criterion = nn.MSELoss().to(device)
model.to(device)

# Define Paths
dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Dataset/"
test_dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/TestDataset/"
csv_FileName = "Dataset.csv"
modelPath = '../../model_weights/RESNET101.pth'
modelPath_bestweights = '../../model_weights/RESNET101_bestWeights.pth'

# Configuration
val_split = 0.3
shuffle_dataset = True
random_seed = 42
plot = False
num_epochs = 1000
batch_size = 32
debug = True
log = True  # Make it true to log in Tensorboard
writer = ""
l = w = []
transformation = False  # False - Custom transformation, True - Automatic
transfrom_val = (1,230,230)  # Set this value if you want custom transformation

print(" Tensorboard Logging : ", log)
print(" Validation Split    : ", val_split * 100, "%")

# Check for csv File
if not os.path.isfile(dataset_Path + csv_FileName):
    print(" CSV File missing..\nGenerating new CVS File..")
    GenerateCSV(datasetPath=dataset_Path, csv_FileName=csv_FileName)
    print(" CSV File Created!")
else:
    print("\n Dataset file available")

# Get Transformation Vales
if transformation:
    print("\n Scanning Dataset to get transformation values")
    inpPath = Path(dataset_Path)
    for file_name in tqdm(sorted(inpPath.glob("*.nii.gz"))):
        imgReg = tio.ScalarImage(file_name)[tio.DATA].permute(0, 3, 1, 2)
        l.append(len(imgReg.squeeze()))
        w.append(len(imgReg.squeeze()[0]))

    transform = tio.CropOrPad((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w))))))
    print(" Transformation Values are : ",
          str((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w)))))))
else:
    print(" Custom Transformation Values : ", transfrom_val)
    transform = tio.CropOrPad(transfrom_val)

# Load dataset
dataset = CustomDataset(dataset_path=dataset_Path, csv_file=dataset_Path + csv_FileName, transform=transform)
print(" Dataset loaded")

# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(val_split * dataset_size))
if shuffle_dataset:
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
dataloader = [train_loader, validation_loader]

# Train and Validate
print("\n Model Training Started..")
trainModel(modelPath=modelPath, dataloaders=dataloader, modelPath_bestweight=modelPath_bestweights,
           num_epochs=num_epochs, model=model,
           criterion=criterion, optimizer=optimizer, log=log)


# Load Test Data
print("\n Loading Test Data")
GenerateCSV(datasetPath=test_dataset_Path, csv_FileName=csv_FileName)
dataset = CustomDataset(dataset_path=test_dataset_Path, csv_file=dataset_Path + csv_FileName, transform=transform)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, log=log)

# Test Model with saved weights
print("Testing model with saved weights")
testModel(model, test_loader, modelPath,)

# Test Model with best weights
print("Testing model with best weights")
testModel(model, test_loader, modelPath_bestweights)