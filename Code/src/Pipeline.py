import tempfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
from torch.utils.data import SubsetRandomSampler
from torchvision import models
from tqdm import tqdm
from pathlib import Path
from Code.src.Dataloader import CustomDataset
from Code.src.Test import testModel, testModel_Image
from Code.src.Train import trainModel
from Code.Utils.CSVGenerator import checkCSV
from Code.Utils.utils import getSubjects


class BlurDetection:
    def __init__(self, data, system_to_run, model_selection, deviceIds,
                 enableMultiGPU, defaultGPUID, epochs, Tensorboard, batch_size,
                 validation_split, num_class_confusionMatrix, testFile, output_path):
        """
        Args:

        """
        # Define Paths
        self.system_to_run = data[system_to_run]
        self.dataset_Path = data[system_to_run]["Dataset_Path"]
        self.test_dataset_Path = data[system_to_run]["Test_Dataset_Path"]
        self.modelPath = data[system_to_run]["model_Path"][str(model_selection)]
        self.modelPath_bestweights = data[system_to_run]["model_bestweight_Path"][str(model_selection)]
        self.log_dir_Path = data[system_to_run]["log_dir"][str(model_selection)]
        self.temp_Test_Path = data[system_to_run]["tempdirTestDataset"]
        self.testFile = testFile
        self.output_path = output_path

        # Configuration
        self.defaultGPU = defaultGPUID
        self.OverWriteCSVFile = False  # Set it to true if you want to use existing CSV file
        self.model_selection = model_selection
        self.useExistingWeights = False  # Use existing weights
        self.val_split = validation_split
        self.shuffle_dataset = True
        self.random_seed = 42
        self.plot = False
        self.num_epochs = epochs
        self.batch_size = batch_size
        self.debug = True
        self.log = Tensorboard  # Make it true to log in Tensorboard
        self.transformation = False  # False - Custom transformation, True - Automatic
        self.transform_val = (1, 224, 224)  # Set this value if you want custom transformation
        self.multiGPUTraining = enableMultiGPU
        self.deviceIDs = deviceIds
        self.delete_dir = False  # Set to true if you want to delete the temp directory of test dataset
        self.useModel = False  # Set to False for testing model against GT
        self.class_cfm = num_class_confusionMatrix

        print("Current temp directory:", tempfile.gettempdir())
        tempfile.tempdir = data[system_to_run]["tempdir"]
        print("Temp directory after change:", tempfile.gettempdir())

        # To make the model deterministic
        torch.set_num_threads(1)
        torch.manual_seed(self.random_seed)
        np.random.seed(self.random_seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True
        torch.cuda.manual_seed(self.random_seed)

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
        if self.multiGPUTraining and torch.cuda.device_count() > 1:
            return nn.DataParallel(model, device_ids=self.deviceIDs)
        return model

    @staticmethod
    def defineOptim(model):
        params_to_update = model.parameters()
        optimizer = optim.SGD(params=params_to_update, lr=0.001, momentum=0.9)
        return optimizer

    @staticmethod
    def defineLoss():
        criterion = nn.MSELoss()  # Try with L1 loss
        return criterion

    def getDevice(self):
        if self.multiGPUTraining:
            return torch.device("cuda:" + str(self.deviceIDs[0]) if torch.cuda.is_available() else 'cpu')
        return torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')

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
        return torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

    def getTrainValdataloader(self):
        train_subs, val_subs = self.getTrainValSubjects()
        # Training and Validation Section
        checkCSV(dataset_Path=self.dataset_Path, csv_FileName="train.csv", subjects=train_subs,
                 overwrite=self.OverWriteCSVFile)
        checkCSV(dataset_Path=self.dataset_Path, csv_FileName="val.csv", subjects=val_subs,
                 overwrite=self.OverWriteCSVFile)

        train_loader = self.getDataloader(dataset_Path=self.dataset_Path, csv_FileName="train.csv",
                                          transform=self.getTransformation())
        validation_loader = self.getDataloader(dataset_Path=self.dataset_Path, csv_FileName="val.csv",
                                               transform=self.getTransformation())
        if self.debug:
            print("No. of Training Subjects   :", len(train_subs))
            print("No. of Validation Subjects :", len(val_subs))
        return [train_loader, validation_loader]

    def getTestdataloader(self):
        checkCSV(dataset_Path=self.test_dataset_Path, csv_FileName="test.csv", subjects=None,
                 overwrite=self.OverWriteCSVFile)
        test_loader = self.getDataloader(dataset_Path=self.dataset_Path, csv_FileName="train.csv",
                                         transform=self.getTransformation())
        return test_loader

    def train(self):
        dataloader = self.getTrainValdataloader()
        model = self.defineModel()
        if self.useExistingWeights:
            model.load_state_dict(torch.load(self.modelPath, map_location=self.getDevice()))

        optimizer = self.defineOptim(model)
        criterion = self.defineLoss()
        print("\n Model Training Started..")
        trainModel(modelPath=self.modelPath, dataloaders=dataloader,
                   modelPath_bestweight=self.modelPath_bestweights,
                   num_epochs=self.num_epochs, model=model,
                   criterion=criterion, optimizer=optimizer, log=self.log,
                   log_dir=self.log_dir_Path, device=self.getDevice(), isMultiGPU=self.multiGPUTraining)

    def test(self):
        test_loader = self.getTestdataloader()
        model = self.defineModel()
        model.load_state_dict(torch.load(self.modelPath_bestweights, map_location=self.getDevice()))
        print("Testing model with saved weights")
        testModel(dataloaders=test_loader, model=model,
                  debug=False, no_class=self.class_cfm, device=self.getDevice())

    def test_singleFile(self, transform_Images, custom_model_path=None):
        transform = None
        if custom_model_path is not None:
            self.modelPath_bestweights = custom_model_path
        if transform_Images:
            transform = self.getTransformation()
        try:
            model = self.defineModel()
            model.load_state_dict(torch.load(self.modelPath_bestweights, map_location=self.getDevice()))
        except:
            print("Cannot load model weights, trying to load model")
            model = torch.load(self.modelPath_bestweights)
        print("Testing model with saved weights")
        testModel_Image(niftyFilePath=self.testFile, model=model,
                        transform=transform, output_path=self.output_path)
