import os
import sys
import torch
import numpy as np
from torch import nn
import torchio as tio
from torch.nn import functional as F
from torch.utils.data import SubsetRandomSampler
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchvision import models
from torchvision.datasets import MNIST
from torchvision import transforms
import pytorch_lightning as pl

#sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Code/')
from Dataloader import CustomDatasetLight

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        backbone = models.resnet101(pretrained=True)
        backbone.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_filters = backbone.fc.in_features
        layers = list(backbone.children())[:-1]
        self.feature_extractor = nn.Sequential(*layers)
        num_target_classes = 1
        self.classifier = nn.Linear(num_filters, num_target_classes)


    def forward(self, x):
        self.feature_extractor.eval()
        with torch.no_grad():
            representations = self.feature_extractor(x).flatten(1)
        x = self.classifier(representations)
        return x

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        x, y = train_batch

        z = self.forward(x.float())
        loss = F.mse_loss(z.float(), y.float())
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        x, y = val_batch
        z = self.forward(x.float())
        loss = F.mse_loss(z.float(), y.float())
        self.log('val_loss', loss)


# data
dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Dataset/"
test_dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/TestDataset/"
csv_FileName = "Dataset.csv"
modelPath = '../../model_weights/RESNET101.pth'
modelPath_bestweights = '../../model_weights/RESNET101_bestWeights.pth'
transfrom_val = (1, 230, 230)
transform = tio.CropOrPad(transfrom_val)
shuffle_dataset = True
random_seed = 42
val_split = 0.3
batch_size = 32

dataset = CustomDatasetLight(dataset_path=dataset_Path, csv_file=dataset_Path + csv_FileName, transform=transform)
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
val_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         sampler=valid_sampler)

# model
model = LitAutoEncoder()

# training
trainer = pl.Trainer(gpus=1, num_nodes=1, limit_train_batches=0.5)
trainer.fit(model, train_loader, val_loader)
