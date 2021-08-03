#from __future__ import print_function
#from __future__ import division
import torchio as tio
from tqdm import tqdm
from pathlib import Path
import torch.nn as nn
from torchvision import models
from models import ResNet
num_classes = 6
model_ft = ResNet.resnet18(pretrained=True)
model_ft.conv1 = nn.Conv2d(1, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)
#set_parameter_requires_grad(model_ft, feature_extract)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, num_classes)
#input_size = 224

path = "/media/hdd_storage/Budha/Dataset/Isotropic"
inpPath = Path(path)
subjects = []
for file_name in tqdm(sorted(inpPath.glob("*T1*.nii.gz"))):
    subjects.append(tio.Subject(image=tio.ScalarImage(file_name), label=[0]))
a = subjects[1]["image"][tio.DATA]
#a = torch.cat((subjects[1]["image"][tio.DATA], subjects[3]["image"][tio.DATA]), 0)
#for i in range(5):
#    a = torch.cat((a,a),0)
print(model_ft(a.float()[:,0:1]))