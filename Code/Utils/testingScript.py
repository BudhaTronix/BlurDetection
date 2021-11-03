import os
import sys

import torch
import torch.nn as nn
import torchio as tio
from torchvision import models

from CSVGenerator import checkCSV

try:
    from Code.src.Dataloader import CustomDataset
    from Code.src.Test import testModel, getModelOP, getModelOP_filePath
except ImportError:
    sys.path.insert(1, '..')
    from Dataloader import CustomDataset
    from Test import testModel  # , getModelOP, getModelOP_filePath


# os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class TestingScript:
    def __init__(self):
        self.dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/TestDataset/"
        self.csv_FileName = "test.csv"
        self.modelPath = '/project/mukhopad/tmp/BlurDetection_tmp/model_weights/R101.pth'
        self.modelPath_bestweights = '../../model_weights/RESNET18_bestWeights.pth'

        # Path for File to be tested
        # self.filePath = "/media/hdd_storage/Budha/Dataset/Test/T2W_TSE.nii.gz"

        # Define Temporary Directory
        self.tempPath = "/home/budha/tmp/data/"
        self.delete_dir = False

        # Define Transformation Value
        self.transform = tio.CropOrPad((1, 230, 230))

        # Define the number of class to split for testing
        self.no_of_class = 4

        # set the bartch size
        self.batch_size = 64

        # Test Model or Use Model
        self.useModel = False  # Set to False for testing model against GT

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
                os.mkdir(self.tempPath)
            except OSError:
                print("Creation of temporary directory %s failed" % self.tempPath)
            else:
                print("Successfully created temporary directory %s " % self.tempPath)

            # Traverse through the dataset to get the ssim values
            for axis in range(2, 3):
                if axis == 0:
                    Subject = Subject  # Coronal
                elif axis == 1:
                    Subject = Subject.permute(0, 2, 1)  # Transverse
                elif axis == 2:
                    Subject = Subject.permute(2, 0, 1)  # Axial
                for i in range(0, len(Subject)):
                    image = Subject[i:(i + 1), :, :].unsqueeze(3)
                    temp = tio.ScalarImage(tensor=image)
                    print(temp.shape)
                    temp.save(self.tempPath + str(fileName) + "-" + str(axis) + "_" + str(i) + '.nii.gz', squeeze=True)

            getModelOP_filePath(self.tempPath, self.modelPath, self.transform)
            # self.testModelScript_Dataloader(test_dataset_Path=self.tempPath, generateCSV=True)

            # Delete the temporary directory
            if self.delete_dir:
                try:
                    os.rmdir(self.tempPath)
                except OSError:
                    print("Deletion of the directory %s failed" % self.tempPath)
                else:
                    print("Successfully deleted the directory %s" % self.tempPath)

    def testModelScript_Dataloader(self, test_dataset_Path, csv_FileName="test.csv", generateCSV=False):
        checkCSV(dataset_Path=test_dataset_Path, csv_FileName=csv_FileName, subjects=None, overwrite=generateCSV)
        dataset = CustomDataset(dataset_path=test_dataset_Path, csv_file=test_dataset_Path + csv_FileName,
                                transform=self.transform, useModel=self.useModel)
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size)

        device = torch.device('cuda:2')
        model = models.resnet101(pretrained=False)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        model.load_state_dict(torch.load(self.modelPath, map_location=device))
        # model.load_state_dict(torch.load(self.modelPath))

        if self.useModel:
            # To be used for model without GT
            getModelOP(dataloaders=test_loader, modelPath=self.modelPath, debug=True)
        else:
            # o be used for model with GT
            print("Testing model with saved weights")
            testModel(dataloaders=test_loader, no_class=self.no_of_class, model=model, debug=False)

    def main(self):
        # Use this function if you have a dataset created
        self.testModelScript_Dataloader(self.dataset_Path, self.csv_FileName)

        # Use this function if you have a dataset created
        # testModelScript_Dataloader_Image(Subject_Name=None, Subject_directory=None, modelPath)


test = TestingScript()
test.main()
