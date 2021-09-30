import sys
import torch
import os
import torchio as tio
from CSVGenerator import GenerateCSV, checkCSV

try:
    from ..Code.Dataloader import CustomDataset
    from ..Code.Test import testModel, getModelOP
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/')
    from Code.Dataloader import CustomDataset
    from Code.Test import testModel, getModelOP

os.environ["CUDA_VISIBLE_DEVICES"] = "6"


class TestingScript:
    def __init__(self):
        self.dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Dataset_2/"
        self.csv_FileName = "test.csv"
        self.modelPath = '../../model_weights/R1.pth'
        self.modelPath_bestweights = '../../model_weights/R1_bw.pth'

        # Path for File to be tested
        self.filePath = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/temp/IXI662-Guys-1120-T1.nii.gz"

        # Define Temporary Directory
        self.tempPath = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/temp/tmp/"
        self.delete_dir = False

        # Define Transformation Value
        self.transform = tio.CropOrPad((1, 230, 230))

        # Define the number of class to split for testing
        self.no_of_class = 4

        # Test Model or Use Model
        self.useModel = True # Set to False for testing model against GT

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
                    temp.save(self.tempPath + str(fileName) + "-" + str(axis) + "_" + str(i) + '.nii.gz', squeeze=True)

            self.testModelScript_Dataloader(test_dataset_Path=self.tempPath, generateCSV=True)
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
        test_loader = torch.utils.data.DataLoader(dataset, batch_size=512)

        if self.useModel:
            getModelOP(dataloaders=test_loader, modelPath=self.modelPath, debug=True)
        else:
            # Test Model with saved weights
            print("Testing model with saved weights")
            testModel(dataloaders=test_loader, no_class=self.no_of_class, modelPath=self.modelPath, debug=False)

    def main(self):
        self.testModelScript_Dataloader_Image(niftyFile=self.filePath)

        # Use this function if you have a dataset created
        # testModelScript_Dataloader(dataset_Path, csv_FileName, modelPath, transform, generateCSV=False)

        # Use this function if you have a dataset created
        # testModelScript_Dataloader_Image(Subject_Name=None, Subject_directory=None, modelPath)


# test = TestingScript()
# test.main()
