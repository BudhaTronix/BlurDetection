import os
import sys
from pathlib import Path

import tempfile
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
from Test import testModel, getModelOP

try:
    from ..Utils.CSVGenerator import checkCSV
    from ..Utils.utils import getSubjects
except ImportError:
    sys.path.insert(1, '../Utils/')
    from CSVGenerator import checkCSV
    from utils import getSubjects

# To make the model deterministic
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(42)


# GPU Setup
# device_id = 5
# os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
# device = "cuda"

class BlurDetection:
    def __init__(self, data, system_to_run, model_selection):
        """
        Args:

        """
        # Define Paths
        self.dataset_Path = data[system_to_run]["Dataset_Path"]
        self.test_dataset_Path = data[system_to_run]["Test_Dataset_Path"]
        self.modelPath = data[system_to_run]["model_Path"][str(model_selection)]
        self.modelPath_bestweights = data[system_to_run]["model_bestweight_Path"][str(model_selection)]
        self.log_dir_Path = data[system_to_run]["log_dir"][str(model_selection)]
        self.temp_Test_Path = data[system_to_run]["tempdirTestDataset"]

        # Configuration
        self.model_selection = model_selection
        self.useExistingWeights = False  # Use existing weights
        self.val_split = 0.3
        self.shuffle_dataset = True
        self.random_seed = 42
        self.plot = False
        self.num_epochs = 50
        self.batch_size = 256
        self.debug = True
        self.log = True  # Make it true to log in Tensorboard
        self.transformation = False  # False - Custom transformation, True - Automatic
        self.transform_val = (1, 224, 224)  # Set this value if you want custom transformation
        self.multiGPUTraining = False
        self.deviceIDs = [0, 1, 2]
        self.delete_dir = False  # Set to true if you want to delete the temp directory of test dataset
        self.useModel = True  # Set to False for testing model against GT

        print("Current temp directory:", tempfile.gettempdir())
        tempfile.tempdir = data[system_to_run]["tempdir"]
        print("Temp directory after change:", tempfile.gettempdir())

    def defineModel(self):
        model = ""
        if self.model_selection == 1:
            model = models.resnet18(pretrained=True)
        elif self.model_selection == 2:
            model = models.resnet50(pretrained=True)
        elif self.model_selection == 3:
            model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        if self.multiGPUTraining:
            model = nn.DataParallel(model, device_ids=self.deviceIDs)
        else:
            model = model.to(self.getDevice())
        return model

    @staticmethod
    def defineOptim(model):
        params_to_update = model.parameters()
        optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
        return optimizer

    def defineLoss(self):
        criterion = nn.MSELoss()  # Try with L1 loss
        criterion = criterion.to(self.getDevice())
        return criterion

    @staticmethod
    def getDevice():
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return device

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

    def getTrainValdataloader(self):
        transform_val = self.getTransformation()
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
        return dataloader

    def trainModel(self):
        dataloader = self.getTrainValdataloader()
        if self.useExistingWeights:
            model = torch.load(self.modelPath_bestweights)
        else:
            model = self.defineModel()
        optimizer = self.defineOptim(model)
        criterion = self.defineLoss()
        print("\n Model Training Started..")
        trainModel(modelPath=self.modelPath, dataloaders=dataloader,
                   modelPath_bestweight=self.modelPath_bestweights,
                   num_epochs=self.num_epochs, model=model,
                   criterion=criterion, optimizer=optimizer, log=self.log,
                   log_dir=self.log_dir_Path, device=self.getDevice())

    def test(self):
        transform_val = self.getTransformation()
        print("\n Loading Test Data")
        checkCSV(dataset_Path=self.test_dataset_Path, csv_FileName="test.csv", subjects=None, overwrite=True)
        test_loader = self.getDataloader(dataset_Path=self.test_dataset_Path, csv_FileName="test.csv",
                                         transform=transform_val)
        no_class = 4
        print("Testing model with saved weights")
        testModel(dataloaders=test_loader, modelPath=self.modelPath,
                  debug=self.debug, no_class=no_class, device=self.getDevice())

    def testModelScript_Dataloader_Image(self, niftyFile=None, Subject=None, fileName=None):
        """
            Please provide a TorchIO Subject
        """
        if niftyFile is not None:
            imgReg = tio.ScalarImage(niftyFile)[tio.DATA].squeeze()
            filename = str(os.path.basename(niftyFile)).replace(".nii.gz", "")
            self.testModelScript_Dataloader_Image(Subject=imgReg, fileName=filename)
        if Subject is not None:
            # Create a temporary directory
            try:
                os.mkdir(self.temp_Test_Path)
            except OSError:
                print("Creation of temporary directory %s failed" % self.temp_Test_Path)
            else:
                print("Successfully created temporary directory %s " % self.temp_Test_Path)

            # Traverse through the dataset to get the ssim values
            for axis in range(0, 3):
                if axis == 1:
                    Subject = Subject  # Coronal
                elif axis == 2:
                    Subject = Subject.permute(0, 2, 1)  # Transverse
                elif axis == 3:
                    Subject = Subject.permute(2, 0, 1)  # Axial
                for i in range(0, len(Subject[0])):
                    image = Subject[i:(i + 1), :, :].unsqueeze(3)
                    temp = tio.ScalarImage(tensor=image)
                    temp.save(self.temp_Test_Path + str(fileName) + "-" + str(axis) + "_" + str(i) + '.nii.gz',
                              squeeze=True)

            self.testModelScript_Dataloader(test_dataset_Path=self.temp_Test_Path, generateCSV=True)
            # Delete the temporary directory
            if self.delete_dir:
                try:
                    os.rmdir(self.temp_Test_Path)
                except OSError:
                    print("Deletion of the directory %s failed" % self.temp_Test_Path)
                else:
                    print("Successfully deleted the directory %s" % self.temp_Test_Path)

    def testModelScript_Dataloader(self, test_dataset_Path, csv_FileName="test.csv", generateCSV=False):

        checkCSV(dataset_Path=test_dataset_Path, csv_FileName=csv_FileName, subjects=None, overwrite=generateCSV)
        dataset = CustomDataset(dataset_path=test_dataset_Path, csv_file=test_dataset_Path + csv_FileName,
                                transform=self.transform, useModel=self.useModel)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=512)

        if self.useModel:
            getModelOP(dataloaders=test_loader, modelPath=self.modelPath, debug=True)
        else:
            # Test Model with saved weights
            print("Testing model with saved weights")
            testModel(dataloaders=test_loader, no_class=self.no_of_class, modelPath=self.modelPath, debug=False)
