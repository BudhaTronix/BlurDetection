from __future__ import division
from __future__ import print_function

import copy
import random
import tempfile
import time
from datetime import datetime
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import torchvision
from torch.cuda.amp import autocast
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
from models.ResNet import resnet101

print("Current temp directory:", tempfile.gettempdir())
tempfile.tempdir = "/home/mukhopad/tmp"
print("Temp directory after change:", tempfile.gettempdir())
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

#from pytorch_lightning.callbacks import Callback
# To make the model deterministic
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(42)
torch.autograd.set_detect_anomaly(True)


##############################################################################
class BlurDetection:
    def __init__(self, model_name="resnet", num_classes=1, batch_size=2, num_epochs=2, device="cuda"):
        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.model_name = model_name

        # Number of classes in the dataset
        self.num_classes = num_classes

        # Batch size for training (change depending on how much memory you have)
        self.batch_size = batch_size

        # Number of epochs to train for
        self.num_epochs = num_epochs

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = False

        # Model Path
        self.PATH = '/home/budha/PycharmProjects/BlurDetection/model_weights/BlurDetection_ModelWeights_SinlgeGPU_RESNET101_MultiClass_DataLoader_Reg_T1.pth'

        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

        TBLOGDIR = "runs/BlurDetection/Training/RegressionModel_T1_Densenet/{}".format(start_time)
        self.writer = SummaryWriter(TBLOGDIR)

        self.device = device

    ##############################################################################
    """DATASET CREATION"""

    @property
    def datasetCreation(self):
        print("\n#################### RETRIVING INFORMATION ####################")
        #path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Regression_T1/"
        path = "/media/hdd_storage/Budha/Dataset/Regression/"
        inpPath = Path(path)
        output = []
        patch_size = (230, 230, 134)
        patch_per_vol = 1  # n_slices
        patch_qlen = patch_per_vol * 2
        val_split = .3
        shuffle_dataset = False
        random_seed = 42
        batch_size = self.batch_size

        for file_name in sorted(inpPath.glob("*.nii.gz")):
            temp = str(file_name.name)
            sigma = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
            fileName = temp.replace(sigma, "")
            if fileName not in output:
                output.append(fileName)

        print("Total Subjects: ", len(output)+5)
        Val = int(len(output) * val_split)
        Train = len(output) - Val
        Test = 5
        print("Subjects in Training: ", Train)
        print("Subjects in Validation: ", Val)
        print("Subjects in Test: ", Test)

        count = 0
        train_subjects = []
        val_subjects = []
        print("\n#################### LOADING DATASET ####################")
        for subject_id in tqdm(output):
            if (count < Train):
                for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                    subject = tio.Subject(image=tio.ScalarImage(file_name),
                                          label=[float(str(file_name.name).split(".nii.gz")[0].split("-")[-1])])
                    train_subjects.append(subject)
            else:
                for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                    subject = tio.Subject(image=tio.ScalarImage(file_name),
                                          label=[float(str(file_name.name).split(".nii.gz")[0].split("-")[-1])])
                    val_subjects.append(subject)
            count += 1

        print("Total files in Training: ", len(train_subjects))
        print("Total files in Validation: ", len(val_subjects))
        print("Total files in Testing: ", Test*5)

        train_dataset = tio.SubjectsDataset(train_subjects)
        val_dataset = tio.SubjectsDataset(val_subjects)

        sampler = tio.data.UniformSampler(patch_size)
        train_dataset = tio.Queue(
            subjects_dataset=train_dataset,
            max_length=patch_qlen,
            samples_per_volume=patch_per_vol,
            sampler=sampler,
            num_workers=0,
            # start_background=True
        )

        val_dataset = tio.Queue(
            subjects_dataset=val_dataset,
            max_length=patch_qlen,
            samples_per_volume=patch_per_vol,
            sampler=sampler,
            num_workers=0,
            # start_background=True
        )

        train_dataset_size = len(train_dataset)
        val_dataset_size = len(val_dataset)

        train_indices = list(range(train_dataset_size))
        val_indices = list(range(val_dataset_size))

        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(train_indices)
            np.random.shuffle(val_indices)

        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
                                            sampler=train_sampler, num_workers=0)
        val = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size,
                                          sampler=valid_sampler, num_workers=0)

        dataloader = []

        dataloader.append(train)
        dataloader.append(val)

        return dataloader

    ###################################################################################
    #os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
    #os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'

    ####################################################################################

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=10, is_inception=False):
        print("\n#################### MODEL - TRAIN & VALIDATION ####################")
        since = time.time()
        val_acc_history = []
        train_loss_history = []
        # model = torch.nn.DataParallel(model, device_ids=[6, 5])
        patience = 5
        precision = 5
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_val_loss = 100000.0
        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs))
            print('-' * 10)
            # Each epoch has a training and validation phase
            for phase in [0, 1]:
                if phase == 0:
                    print("Model In Training mode")
                    model.train()  # Set model to training mode
                else:
                    print("Model In Validation mode")
                    model.eval()  # Set model to evaluate mode

                running_loss = 0.0
                running_corrects = 0
                counter = 0
                # Iterate over data.
                for batch in tqdm(dataloaders[phase]):
                    image_batch = batch["image"][tio.DATA].squeeze(1)
                    labels_batch = batch["label"][0]

                    select_orientation = random.randint(1, 3)
                    if select_orientation == 1:
                        image_batch = image_batch  # Coronal
                    elif select_orientation == 2:
                        image_batch = image_batch.permute(0, 2, 3, 1)  # Transverse
                    elif select_orientation == 3:
                        image_batch = image_batch.permute(0, 3, 2, 1)  # Axial

                    for i in range(0, len(image_batch[0])):
                        inputs = image_batch[:, i:(i + 1), :, :].float()
                        optimizer.zero_grad()
                        # forward
                        with torch.set_grad_enabled(phase == 0):
                            if is_inception and phase == 0:
                                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                outputs, aux_outputs = model(inputs)
                                loss1 = criterion(outputs, labels_batch.unsqueeze(1).float().to(self.device))
                                loss2 = criterion(aux_outputs, labels_batch.unsqueeze(1).float().to(self.device))
                                loss = loss1 + 0.4 * loss2
                            else:
                                with autocast(enabled=True):
                                    # inputs = (inputs-inputs.min())/(inputs.max()-inputs.min())  #Min Max normalization
                                    inputs = inputs / np.linalg.norm(inputs)  # Gaussian Normalization
                                    outputs = model(inputs.to(self.device))
                                    loss = criterion(outputs, labels_batch.unsqueeze(1).float().to(self.device))
                                    counter = counter + 1

                            # backward + optimize only if in training phase
                            if phase == 0:
                                loss.backward()
                                optimizer.step()

                            # statistics
                            running_loss += loss.item() * len(image_batch)
                            running_corrects += torch.sum(outputs.cpu().squeeze() == labels_batch.float())  # .data.to(self.device))

                epoch_loss = running_loss / counter
                epoch_acc = running_corrects.double() / counter

                # self.writer.add_scalar("Acc/Epoch", epoch_acc, epoch)
                if phase == 0:
                    mode = "Train"
                    self.writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
                    epoch_loss = round(epoch_loss, precision)
                    train_loss_history.append(epoch_loss)
                    if(epoch%5==0):
                        torch.save(model, self.PATH)

                else:
                    mode = "Val"
                    self.writer.add_scalar("Loss/val", epoch_loss, epoch)

                print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == 1 and (epoch_acc.item() >= best_acc or epoch_loss < best_val_loss):
                    print("Saving the best model weights")
                    best_acc = epoch_acc.item()
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == 1:
                    val_acc_history.append(epoch_acc.data)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print("Saving the model")
        torch.save(model, self.PATH)

        # load best model weights
        model.load_state_dict(best_model_wts)
        PATH = '../../model_weights/BlurDetection_ModelWeights_SinlgeGPU_RESNET_MultiClass_DataLoader_Reg_T1_bestWeights.pth'
        torch.save(model, PATH)

        return model, val_acc_history

    #################################################################################
    def set_parameter_requires_grad(self, model, feature_extracting):
        if feature_extracting:
            for param in model.parameters():
                param.requires_grad = False

    #################################################################################
    def initialize_model(self, model_name, num_classes, feature_extract, use_pretrained=True):
        # Initialize these variables which will be set in this if statement. Each of these
        #   variables is model specific.
        model_ft = None
        input_size = 0

        if model_name == "resnet":
            """ Resnet18
            """
            #model_ft = models.resnet18(pretrained=use_pretrained)
            #model_ft = models.resnet101(pretrained=use_pretrained)
            model_ft = resnet101(pretrained=use_pretrained)
            model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            #model_ft = model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
            print(num_ftrs, num_classes)
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "alexnet":
            """ Alexnet
            """
            model_ft = models.alexnet(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "vgg":
            """ VGG11_bn
            """
            model_ft = models.vgg11_bn(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier[6].in_features
            model_ft.classifier[6] = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "squeezenet":
            """ Squeezenet
            """
            model_ft = models.squeezenet1_0(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            model_ft.classifier[1] = nn.Conv2d(512, num_classes, kernel_size=(1, 1), stride=(1, 1))
            model_ft.num_classes = num_classes
            input_size = 224

        elif model_name == "densenet":
            """ Densenet
            """
            model_ft = models.densenet121(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.classifier.in_features
            model_ft.classifier = nn.Linear(num_ftrs, num_classes)
            input_size = 224

        elif model_name == "inception":
            """ Inception v3
            Be careful, expects (299,299) sized images and has auxiliary output
            """
            model_ft = models.inception_v3(pretrained=use_pretrained)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            # Handle the auxilary net
            num_ftrs = model_ft.AuxLogits.fc.in_features
            model_ft.AuxLogits.fc = nn.Linear(num_ftrs, num_classes)
            # Handle the primary net
            num_ftrs = model_ft.fc.in_features
            model_ft.fc = nn.Linear(num_ftrs, num_classes)
            input_size = 299

        else:
            print("Invalid model name, exiting...")
            exit()

        return model_ft, input_size

    #################################################################################

    def callFunction(self):
        # Initialize the model for this run
        model_ft, input_size = self.initialize_model(self.model_name, self.num_classes, self.feature_extract)

        # Print the model we just instantiated
        print(model_ft)

        # Send the model to GPU
        model_ft = model_ft.to(self.device)

        params_to_update = model_ft.parameters()
        print("Params to learn:")
        if self.feature_extract:
            params_to_update = []
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    params_to_update.append(param)
                    print("\t", name)
        else:
            for name, param in model_ft.named_parameters():
                if param.requires_grad == True:
                    print("\t", name)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9, nesterov=True)
        #########################################################################

        # Setup the loss fxn
        # criterion = nn.CrossEntropyLoss() #- Use this for multiple class
        # criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.L1Loss()
        criterion = nn.MSELoss()

        # Train and evaluate
        dataloader = self.datasetCreation
        model_ft, hist = self.train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs=self.num_epochs,
                                          is_inception=(self.model_name == "inception"))
        self.writer.flush()
        self.writer.close()


a = BlurDetection()
a.callFunction()
