from __future__ import print_function
from __future__ import division
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from pathlib import Path
from torch.utils.data.sampler import SubsetRandomSampler
from motion import MotionCorrupter
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.tensorboard import SummaryWriter
import time
import os
import copy

print("PyTorch Version: ", torch.__version__)
print("Torchvision Version: ", torchvision.__version__)

# To make the model deterministic
torch.manual_seed(41)
np.random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.cuda.manual_seed(42)


##############################################################################
class BlurDetection:
    def __init__(self, model_name="resnet", num_classes=1, batch_size=4, num_epochs=10, corruptionDegree=10,
                 device="cuda:4"):
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
        self.PATH = '../model_weights/BlurDetection_ModelWeights_SinlgeGPU_RESNET.pth'

        self.corruptionDegree = corruptionDegree

        self.writer = SummaryWriter('runs/BlurDetection')

        self.device = device

    ##############################################################################
    """DATASET CREATION"""

    @property
    def datasetCreation(self):
        t1_inp_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1_mini/"

        print("##########Dataset Loader################")
        inpPath = Path(t1_inp_path)
        torch.manual_seed(42)
        moco = MotionCorrupter(degrees=self.corruptionDegree)
        prob_corrupt = 0.75
        patch_size = (256, 256, 1)
        patch_per_vol = 146  # n_slices
        patch_qlen = patch_per_vol * 4
        moco_transforms = [
            tio.Lambda(moco.perform, p=prob_corrupt),
            tio.Lambda(moco.prepare, p=1)
        ]
        moco_transform = tio.Compose(moco_transforms)

        subjects = []
        for file_name in sorted(inpPath.glob("*.nii.gz")):
            subject = tio.Subject(image=tio.ScalarImage(file_name))
            subjects.append(subject)
        dataset = tio.SubjectsDataset(subjects, transform=moco_transform)

        # split here with scikit
        sampler = tio.data.UniformSampler(patch_size)
        dataset = tio.Queue(
            subjects_dataset=dataset,
            max_length=patch_qlen,
            samples_per_volume=patch_per_vol,
            sampler=sampler,
            num_workers=0,
            # start_background=False
        )
        print('Number of subjects in T1 dataset:', len(dataset))
        print("########################################\n\n")

        validation_split = .1
        shuffle_dataset = True
        random_seed = 42
        batch_size = 4
        # Creating data indices for training and validation splits:
        dataset_size = len(dataset)
        indices = list(range(dataset_size))
        split = int(np.floor(validation_split * dataset_size))
        if shuffle_dataset:
            np.random.seed(random_seed)
            np.random.shuffle(indices)
        train_indices, val_indices = indices[split:], indices[:split]

        # Creating PT data samplers and loaders:
        train_sampler = SubsetRandomSampler(train_indices)
        valid_sampler = SubsetRandomSampler(val_indices)

        train = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                            sampler=train_sampler)
        val = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                          sampler=valid_sampler)
        dataloader = []
        dataloader.append(train)
        dataloader.append(val)

        return dataloader

    ###################################################################################
    """TRANSFORMATIONS"""
    """

    transform_0 = tio.Compose([tio.RandomAffine(image_interpolation='nearest')])
    transform_1 = tio.Compose([tio.RandomGhosting((4, 10), (0, 1, 2), (.5, 1)), tio.RandomBlur((1, 2))])
    # transform_3 = transform
    preprocess = transforms.Compose([
        # transforms.Resize(256),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485], std=[0.229]),
    ])
    """

    ###################################################################################
    os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
    os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'

    ####################################################################################

    def train_model(self, model, dataloaders, criterion, optimizer, num_epochs=25, is_inception=False):
        since = time.time()
        val_acc_history = []

        best_model_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0

        for epoch in range(num_epochs):
            print('Epoch {}/{}'.format(epoch, num_epochs - 1))
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
                c = 0
                # Iterate over data.
                for batch in tqdm(dataloaders[phase]):
                    batch_img = batch['image'][tio.DATA].float()

                    batch_lbl = []
                    for i in range(self.batch_size):
                        # if phase == 1:
                        #   print(batch_img.shape)
                        if torch.equal(batch_img[i, 0], batch_img[i, 1]):
                            batch_lbl.append([0])
                        else:
                            batch_lbl.append([1])
                    batch_img = batch_img[:, 1, ...].unsqueeze(1).squeeze(-1)

                    labels = torch.Tensor(batch_lbl).float().to(self.device)
                    inputs = batch_img

                    optimizer.zero_grad()

                    # forward
                    with torch.set_grad_enabled(phase == 0):
                        if is_inception and phase == 0:
                            # From https://discuss.pytorch.org/t/how-to-optimize-inception-model-with-auxiliary-classifiers/7958
                            outputs, aux_outputs = model(inputs)
                            loss1 = criterion(outputs, labels)
                            loss2 = criterion(aux_outputs, labels)
                            loss = loss1 + 0.4 * loss2
                        else:
                            with autocast(enabled=True):
                                outputs = model(inputs.to(self.device))
                                loss = criterion(outputs, labels)
                                # writer.add_scalar("Loss/train", loss, epoch)
                                counter = counter + 1

                        _, preds = torch.max(outputs, 1)

                        # backward + optimize only if in training phase
                        if phase == 0:
                            loss.backward()
                            optimizer.step()

                        # statistics
                        running_loss += loss.item()  # * batch_img.shape[0]
                        running_corrects += torch.sum(preds == labels.data)

                # epoch_loss = running_loss / len(dataloaders[phase].dataset)
                # epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
                epoch_loss = running_loss / counter
                epoch_acc = running_corrects.double() / counter
                if phase == 0:
                    mode = "Train"
                    self.writer.add_scalar("Loss/train", epoch_loss, epoch)
                else:
                    mode = "Val"
                    self.writer.add_scalar("Loss/val", epoch_loss, epoch)
                    img_grid = torchvision.utils.make_grid(inputs)
                    self.writer.add_image('four images ', img_grid)
                    """
                    import torchvision.utils as vutils
                    writer.add_image('{}/output'.format(section),
                     vutils.make_grid(outputs[0, 0, ..., int(AUGMENTATION_PARAMETERS['patch_size'] / 2)],
                                      normalize=True,
                                      scale_each=True),
                     epoch)

                    """
                print('{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))

                # deep copy the model
                if phase == '1' and epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                if phase == '1':
                    val_acc_history.append(epoch_acc)

        time_elapsed = time.time() - since
        print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
        print('Best val Acc: {:4f}'.format(best_acc))

        # load best model weights
        model.load_state_dict(best_model_wts)
        torch.save(model, self.PATH)
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
            model_ft = models.resnet18(pretrained=use_pretrained)
            model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            self.set_parameter_requires_grad(model_ft, feature_extract)
            num_ftrs = model_ft.fc.in_features
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
        optimizer_ft = optim.SGD(params_to_update, lr=0.001, momentum=0.9)
        #########################################################################

        # Setup the loss fxn
        # criterion = nn.CrossEntropyLoss() - Use this for multiple class
        criterion = nn.BCEWithLogitsLoss()
        # criterion = nn.L1Loss()

        # Train and evaluate
        dataloader = self.datasetCreation
        model_ft, hist = self.train_model(model_ft, dataloader, criterion, optimizer_ft, num_epochs=self.num_epochs,
                                          is_inception=(self.model_name == "inception"))
        self.writer.flush()
        self.writer.close()