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

from Train import trainModel
from Dataloader import CustomDataset
from Test import testModel

try:
    from ..Utils.CSVGenerator import checkCSV
    from ..Utils.utils import getSubjects
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Utils/')
    from CSVGenerator import checkCSV
    from utils import getSubjects


# GPU Setup
# device_id = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
# device = "cuda"


class BlurDetection:
    def __init__(self, dataset_path, test_dataset_Path, model_Path, model_bestweight_Path):
        """
        Args:

        """
        # Define Paths
        self.dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Dataset_2/"
        self.test_dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/TestDataset/"
        self.modelPath = '../../model_weights/RESNET101.pth'
        self.modelPath_bestweights = '../../model_weights/RESNET101_bestWeights.pth'

        # Configuration
        self.val_split = 0.3
        self.shuffle_dataset = True
        self.random_seed = 42
        self.plot = False
        self.num_epochs = 3000
        self.batch_size = 300
        self.debug = True
        self.log = True  # Make it true to log in Tensorboard

        self.transformation = False  # False - Custom transformation, True - Automatic
        self.transform_val = (1, 230, 230)  # Set this value if you want custom transformation

    @staticmethod
    def defineModel():
        model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
        parallel_net = nn.DataParallel(model, device_ids=[2, 3, 4])
        return parallel_net

    @staticmethod
    def defineOptim(model):
        params_to_update = model.parameters()
        optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
        return optimizer

    @staticmethod
    def defineLoss():
        criterion = nn.MSELoss()
        return criterion

    def getTrainValSubjects(self):
        # Retrieve the subjects having equal distribution of class
        inpPath = Path(self.dataset_Path)
        _, filtered_subjects = getSubjects(inpPath)

        # Random shuffle the subjects
        if self.shuffle_dataset:
            np.random.seed(self.random_seed)
            np.random.shuffle(filtered_subjects)

        # Split Subjects to Train and Validation
        train_subs = filtered_subjects[int(self.val_split * len(filtered_subjects)): len(filtered_subjects)]
        val_subs = filtered_subjects[0: int(self.val_split * len(filtered_subjects))]

        return train_subs, val_subs

    def getTransformation(self):
        if self.transformation:
            l = w = []
            print("\n Scanning Dataset to get transformation values")
            inpPath = Path(self.dataset_Path)
            for file_name in tqdm(sorted(inpPath.glob("*.nii.gz"))):
                imgReg = tio.ScalarImage(file_name)[tio.DATA].permute(0, 3, 1, 2)
                l.append(len(imgReg.squeeze()))
                w.append(len(imgReg.squeeze()[0]))

            transform = tio.CropOrPad((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w))))))
            print(" Transformation Values are : ",
                  str((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w)))))))
        else:
            print(" Custom Transformation Values : ", self.transform_val)
            transform = tio.CropOrPad(self.transform_val)
        return transform

    def getDataloader(self, dataset_Path, transform, csv_FileName):
        dataset = CustomDataset(dataset_path=dataset_Path, csv_file=dataset_Path + csv_FileName, transform=transform)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        return loader

    def trainModel(self, useSaveWeights, transform_val):
        train_subs, val_subs = self.getTrainValSubjects()
        # Training and Validation Section
        checkCSV(dataset_Path=self.dataset_Path, csv_FileName="train.csv", subjects=train_subs, overwrite=True)
        checkCSV(dataset_Path=self.dataset_Path, csv_FileName="val.csv", subjects=val_subs, overwrite=True)

        train_loader = self.getDataloader(dataset_Path=self.dataset_Path, csv_FileName="train.csv",
                                          transform=transform_val)
        validation_loader = self.getDataloader(dataset_Path=self.dataset_Path, csv_FileName="val.csv",
                                               transform=transform_val)

        if self.debug:
            print("No. of Training Subjects   :", len(train_subs))
            print("No. of Validation Subjects :", len(val_subs))

        dataloader = [train_loader, validation_loader]

        if useSaveWeights:
            model = torch.load(self.modelPath_bestweights)
        else:
            model = self.defineModel()
        optimizer = self.defineOptim(model)
        criterion = self.defineLoss()
        print("\n Model Training Started..")
        trainModel(modelPath=self.modelPath, dataloaders=dataloader, modelPath_bestweight=self.modelPath_bestweights,
                   num_epochs=self.num_epochs, model=model,
                   criterion=criterion, optimizer=optimizer, log=self.log)

    def test(self, transform_val):
        print("\n Loading Test Data")
        checkCSV(dataset_Path=self.test_dataset_Path, csv_FileName="test.csv", subjects=None, overwrite=True)
        test_loader = self.getDataloader(dataset_Path=self.test_dataset_Path, csv_FileName="test.csv",
                                         transform=transform_val)
        no_class = 4
        print("Testing model with saved weights")
        testModel(dataloaders=test_loader, modelPath=self.modelPath, debug=self.debug, no_class=no_class)

        # Test Model with best weights
        # print("Testing model with best weights")
        # testModel(dataloaders=test_loader, modelPath=self.modelPath_bestweights, debug=self.debug, no_class=no_class)
