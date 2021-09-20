from __future__ import division
from __future__ import print_function

import copy
import os
import random
import tempfile
import time
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchio as tio
import torchvision
from torch.cuda.amp import autocast, GradScaler
from torch.utils.data.sampler import SubsetRandomSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from tqdm import tqdm
import sys

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/')

sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Utils/')
import pytorch_ssim

# from models.ResNet import resnet18

print("Current temp directory:", tempfile.gettempdir())
tempfile.tempdir = "/home/mukhopad/tmp"
print("Temp directory after change:", tempfile.gettempdir())
print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# To make the model deterministic
torch.manual_seed(42)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = True
torch.cuda.manual_seed(42)
#torch.autograd.set_detect_anomaly(True)
torch.cuda.empty_cache()
os.environ["CUDA_VISIBLE_DEVICES"] = "6"

log = False
debug = True
##############################################################################
class BlurDetection:
    def __init__(self, model_name="resnet", num_classes=1, batch_size=128, num_epochs=1000, device="cuda"):
        # Models to choose from [resnet, alexnet, vgg, squeezenet, densenet, inception]
        self.model_name = model_name

        # Number of classes in the dataset
        self.num_classes = num_classes

        # Batch size for training (change depending on how much memory you have)
        self.batch_size = batch_size

        # Number of epochs to train for
        self.num_epochs = num_epochs

        # Path of dataset
        self.path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/SSIM/"
        # self.path = "/media/hdd_storage/Budha/Dataset/Regression/"

        # Flag for feature extracting. When False, we finetune the whole model,
        #   when True we only update the reshaped layer params
        self.feature_extract = False

        # Model Path
        self.PATH = '../../model_weights/RESNET101_DataLoader_Reg_T1.pth'

        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")

        if log:
            TBLOGDIR = "runs/BlurDetection/Training/Regression_T1_RESNET_SSIM/{}".format(start_time)
            self.writer = SummaryWriter(TBLOGDIR)

        self.device = device

    ##############################################################################
    """DATASET CREATION"""

    @property
    def datasetCreation(self):
        inpPath = Path(self.path)
        main_Path = Path("/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/")
        output = []
        val_split = .5
        random_seed = 42
        mem_batch = 0.01
        batch_size = self.batch_size
        random.seed(42)

        for file_name in sorted(inpPath.glob("*.nii.gz")):
            temp = str(file_name.name)
            ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
            fileName = temp.replace(ssim, "")
            if fileName not in output:
                output.append(fileName)

        val_subjects = output[0:int(len(output)*(val_split))]
        train_subjects = output[int(len(output)*val_split):len(output)]

        if debug:
            print("\n#################### RETRIEVING INFORMATION ####################")
            print("Total Subjects: ", len(output))
            print("Subjects in Training: ", len(train_subjects))
            print("Subjects in Validation: ", len(val_subjects))
            print("\n#################### LOADING DATASET ####################")

        train_files = []
        t_len = 0
        random.shuffle(train_subjects)
        train_batch = train_subjects[0:int(len(train_subjects)*mem_batch)]
        for subject_id in train_batch:
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                t = tio.ScalarImage(file_name)
                t_len = t_len + len(t.data[0]) + len(t.data[0][0]) + len(t.data[0][0][0])
                break
        t_len = t_len*10
        print("Training batch files - ",t_len)
        flag = ctr_1 = ctr_2 = ctr_3 = ctr_4 = ctr_5 = 0
        l = w = []
        for subject_id in tqdm(train_subjects):
            if flag == 1:
                break
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                imgReg = tio.ScalarImage(file_name)[tio.DATA]
                imgOrig = tio.ScalarImage(main_Path / str(subject_id + ".nii.gz"))[tio.DATA]
                for j in range(3):
                    if j == 0:
                        imgReg_op = imgReg
                        imgOrig_op = imgOrig
                    if j == 1:
                        imgReg_op = imgReg.permute(0, 2, 3, 1)
                        imgOrig_op = imgOrig.permute(0, 2, 3, 1)
                    if j == 2:
                        imgReg_op = imgReg.permute(0, 3, 2, 1)
                        imgOrig_op = imgOrig.permute(0, 3, 2, 1)
                    if torch.cuda.is_available():
                        imgReg_ssim = imgReg_op.cuda()
                        imgOrig_ssim = imgOrig_op.cuda()
                    ssim = pytorch_ssim.ssim(imgOrig_ssim.double(), imgReg_ssim.double()).mean(0).mean(1).mean(
                        1).detach().cpu()
                    c = 0
                    for i in range(0, len(imgReg_op.squeeze())):
                        subject = imgReg_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)
                        if 0 <= ssim[i] <= .2 and ctr_1 < int(t_len / 5):
                            train_files.append([subject,ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_1 += 1
                        elif 0.2 < ssim[i] <= .4 and ctr_2 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_2 += 1
                        elif 0.4 < ssim[i] <= .6 and ctr_3 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_3 += 1
                        elif 0.6 < ssim[i] <= .8 and ctr_4 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_4 += 1
                        elif 0.8 < ssim[i] <= 1 and ctr_5 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_5 += 1
                        if(ctr_5 < int(t_len / 5) and c<3):
                            slice = random.randint(0, len(imgReg_op.squeeze()))
                            ssim[0] = 1.0
                            subject = imgOrig_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)

                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_5 += 1
                            c += 1
                        if ctr_1 == ctr_2 == ctr_3 == ctr_4 == ctr_5 == int(t_len / 5):
                            flag = 1
                            break
                #flag = 1
        #_, labels = train_files
        fig, ax = plt.subplots()
        ax.set(title="SSIM distribution",
               xlabel="SSIM",
               ylabel="Count")
        ax.hist(labels.numpy(), bins=5)
        plt.show()
        del imgReg_ssim
        del imgOrig_ssim
        torch.cuda.empty_cache()
        print("Total files in Training: ", len(train_files))
        transform = tio.CropOrPad((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w))))))
        for i in range(len(train_files)):
            train_files[i] = [transform(train_files[i].__getitem__(0).unsqueeze(0).unsqueeze(0)).squeeze() ,train_files[i].__getitem__(1)]
        #################

        random.shuffle(val_subjects)
        val_batch = val_subjects[0:int(len(val_subjects) * mem_batch)]
        v_len = 0
        for subject_id in val_batch:
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                t = tio.ScalarImage(file_name)
                v_len = v_len + len(t.data[0]) + len(t.data[0][0]) + len(t.data[0][0][0])
                break
        print("Validation batch files - ", v_len*10)
        v_len = v_len*10
        val_files = []
        flag = ctr_1 = ctr_2 = ctr_3 = ctr_4 = ctr_5 = 0
        for subject_id in tqdm(val_subjects):
            if flag == 1:
                break
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                imgReg = tio.ScalarImage(file_name)[tio.DATA]
                imgOrig = tio.ScalarImage(main_Path / str(subject_id + ".nii.gz"))[tio.DATA]
                for j in range(3):
                    if j == 0:
                        imgReg_op = imgReg
                        imgOrig_op = imgOrig
                    if j == 1:
                        imgReg_op = imgReg.permute(0, 2, 3, 1)
                        imgOrig_op = imgOrig.permute(0, 2, 3, 1)
                    if j == 2:
                        imgReg_op = imgReg.permute(0, 3, 2, 1)
                        imgOrig_op = imgOrig.permute(0, 3, 2, 1)
                    if torch.cuda.is_available():
                        imgReg_ssim = imgReg_op.cuda()
                        imgOrig_ssim = imgOrig_op.cuda()
                    ssim = pytorch_ssim.ssim(imgOrig_ssim.double(), imgReg_ssim.double()).mean(0).mean(1).mean(
                        1).detach().cpu()

                    for i in range(0, len(imgReg_op.squeeze())):
                        subject = imgReg_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)
                        if 0 <= ssim[i] <= .2 and ctr_1 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_1 += 1
                        elif 0.2 < ssim[i] <= .4 and ctr_2 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_2 += 1
                        elif 0.4 < ssim[i] <= .6 and ctr_3 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_3 += 1
                        elif 0.6 < ssim[i] <= .8 and ctr_4 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_4 += 1
                        elif 0.8 < ssim[i] <= 1 and ctr_5 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_5 += 1
                        if(ctr_5 < int(v_len / 5) and c<3):
                            slice = random.randint(0, len(imgReg_op.squeeze()))
                            ssim[0] = 1.0
                            subject = imgOrig_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)
                            val_files.append([subject, ssim[i]])
                            ctr_5 += 1
                            c += 1
                        if ctr_1 == ctr_2 == ctr_3 == ctr_4 == ctr_5 == int(v_len / 5):
                            flag = 1
                            break
                #flag = 1
        for i in range(len(val_files)):
            val_files[i] = [transform(val_files[i].__getitem__(0).unsqueeze(0).unsqueeze(0)).squeeze(),
                              val_files[i].__getitem__(1)]
        del imgReg_ssim
        del imgOrig_ssim
        torch.cuda.empty_cache()
        if debug:
            print("Total files in Validation: ", len(val_files))

        train = torch.utils.data.DataLoader(train_files, batch_size=batch_size, shuffle=True)
        val = torch.utils.data.DataLoader(val_files, batch_size=batch_size, shuffle=True)
        dataloader = [train, val]
        torch.cuda.empty_cache()
        return dataloader

    ###################################################################################
    os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
    os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'

    ####################################################################################

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=10, is_inception=False):
        print("\n#################### MODEL - TRAIN & VALIDATION ####################")
        # initialize the early_stopping object
        #patience = 20
        disp = True
        dist = True
        #early_stopping = EarlyStopping(patience=patience, verbose=True)
        scaler = GradScaler()
        since = time.time()
        val_acc_history = []
        train_loss_history = []
        # model = torch.nn.DataParallel(model, device_ids=[6, 5])
        precision = 4
        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        best_val_loss = 100000.0
        flag = True
        dataloaders = None
        for epoch in range(num_epochs):
            if flag:
                if epoch%10 == 0:
                    dataloaders = None
                    torch.cuda.empty_cache()
                    dataloaders = self.datasetCreation
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
                        image_batch, labels_batch = batch
                        image_batch = image_batch.unsqueeze(1)

                        if disp:
                            epi_img_data = image_batch[0:6,:,:,:]
                            slices = []
                            for i in range(0, len(epi_img_data)):
                                slices.append(epi_img_data[i, :, :, :])
                            fig, axes = plt.subplots(1, len(slices))
                            for j, slice in enumerate(slices):
                                axes[j].imshow(slice.T, cmap="gray", origin="lower")
                            plt.suptitle("Labels: " + str(labels_batch[0:6].numpy()))
                            plt.show()
                        if dist:
                            fig, ax = plt.subplots()
                            ax.set(title="SSIM distribution",
                                   xlabel="SSIM",
                                   ylabel="Count")
                            ax.hist(labels_batch.numpy(), bins=5)
                            plt.show()

                        optimizer.zero_grad()
                        # forward
                        with torch.set_grad_enabled(phase == 0):
                            if is_inception and phase == 0:
                                # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                                outputs, aux_outputs = model(image_batch)
                                loss1 = criterion(outputs, labels_batch.unsqueeze(1).float().to(self.device))
                                loss2 = criterion(aux_outputs, labels_batch.unsqueeze(1).float().to(self.device))
                                loss = loss1 + 0.4 * loss2
                            else:
                                with autocast(enabled=True):
                                    image_batch = (image_batch-image_batch.min())/(image_batch.max()-image_batch.min())  #Min Max normalization
                                    #image_batch = image_batch / np.linalg.norm(image_batch)  # Gaussian Normalization
                                    outputs = model(image_batch.float().to(self.device))
                                    loss = criterion(outputs.squeeze(1).float(), labels_batch.float().to(self.device))
                                    counter = counter + 1

                            # backward + optimize only if in training phase
                            if phase == 0:
                                scaler.scale(loss).backward()
                                scaler.step(optimizer)
                                scaler.update()

                            # statistics
                            running_loss += loss.item()
                            running_corrects += torch.sum(
                                np.around(outputs.detach().cpu().squeeze(), decimals=precision) == np.around(labels_batch, decimals=precision))

                    epoch_loss = running_loss / counter
                    epoch_acc = running_corrects.double() / counter
                    if log:
                        self.writer.add_scalar("Acc/Epoch", epoch_acc, epoch)
                    if phase == 0:
                        mode = "Train"
                        if log:
                            self.writer.add_scalar("Loss/Epoch", epoch_loss, epoch)
                        # epoch_loss = round(epoch_loss, precision)
                        # train_loss_history.append(epoch_loss)
                    else:
                        mode = "Val"
                        if log:
                            self.writer.add_scalar("Loss/val", epoch_loss, epoch)
                        """
                        early_stopping(epoch_loss, model)
                        if early_stopping.early_stop:
                            print("Early stopping")
                            print("Saving the best model weights")
                            best_acc = epoch_acc.item()
                            best_model_wts = copy.deepcopy(model.state_dict())
                            flag = False
                            break
                        """

                    print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

                    # deep copy the model
                    if phase == 1 and (epoch_acc.item() >= best_acc or epoch_loss < best_val_loss):
                        print("Saving the best model weights")
                        best_acc = epoch_acc.item()
                        best_model_wts = copy.deepcopy(model.state_dict())
                    # if phase == 1:
                    # val_acc_history.append(epoch_acc.data)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))
        print("Saving the model")
        torch.save(model, self.PATH)

        # load best model weights
        model.load_state_dict(best_model_wts)
        PATH = '../../model_weights/RESNET_Reg_T1_bestWeights.pth'
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
            model_ft = models.resnet101(pretrained=use_pretrained)
            # model_ft = models.resnet101(pretrained=use_pretrained)
            # model_ft = resnet18(pretrained=use_pretrained)
            model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            # model_ft = model_ft.fc.register_forward_hook(lambda m, inp, out: F.dropout(out, p=0.5, training=m.training))
            self.set_parameter_requires_grad(model_ft, feature_extract) # Not required
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
        criterion = nn.MSELoss().to(self.device)

        # Train and evaluate
        #dataloader, testSet = self.datasetCreation
        dataloader = ""
        model_ft, hist = self.train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs=self.num_epochs,
                                          is_inception=(self.model_name == "inception"))
        #acc = ModelTest(disp=True, preCreated=True, dataset=testSet, model_path=self.PATH)
        #print("Test Accuracy on 5 Subjects : ", acc)
        if log:
            self.writer.flush()
            self.writer.close()


a = BlurDetection()
a.callFunction()
