import torch
import torchio as tio
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm
from models_D import D_Net
import numpy as np
from pathlib import Path
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler

##############################################################################
"""DATASET CREATION"""
t1_inp_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1/"

print("##########Dataset Loader################")
inpPath = Path(t1_inp_path)
subjects = []
for file_name in sorted(inpPath.glob("*.nii.gz")):
    subject = tio.Subject(image = tio.ScalarImage(file_name))
    subjects.append(subject)
dataset = tio.SubjectsDataset(subjects)
print('Number of subjects in T1 dataset:', len(dataset))
print("########################################\n\n")

validation_split = .1
shuffle_dataset = True
random_seed= 42
batch_size = 1
# Creating data indices for training and validation splits:
dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))
if shuffle_dataset :
    np.random.seed(random_seed)
    np.random.shuffle(indices)
train_indices, val_indices = indices[split:], indices[:split]

# Creating PT data samplers and loaders:
train_sampler = SubsetRandomSampler(train_indices)
valid_sampler = SubsetRandomSampler(val_indices)

train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                           sampler=train_sampler)
validation_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                                sampler=valid_sampler)
###################################################################################
"""PRINTING"""
#print("##########Sample Printing###############")
#sample_subject = dataset[0]
#show_slices(sample_subject['image'])
#print("Label : ",sample_subject['label'])
#print("########################################\n\n")
###################################################################################
"""TRANSFORMATIONS"""
transform_1 = tio.Compose([tio.RandomGhosting((4, 10), (0, 1, 2), (.5, 1)), tio.RandomBlur((1, 2))])


###################################################################################
"""TRAINING"""
print("\n################# TRAINING ########################")
device = "cuda:6"
net = D_Net().to(device)
model = D_Net()

net.to(device)
PATH = '../model_weights/BlurDetection_ModelWeights_SinlgeGPU.pth'
patch_size = 64,128,128
patch_overlap = 0
epochs = 10

criterion = torch.nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
print(len(train_loader))

for epoch in range(epochs):
    running_loss = 0.0
    counter = 0
    print("\n################# Epoch :", (epoch+1), " ########################")
    for j in tqdm(range(len(train_loader))):
        one_batch = next(iter(train_loader))
        batch_img = one_batch['image'][tio.DATA]#.float()
        batch_lbl = np.random.randint(2)
        batch_img = batch_img.permute(0,1,4,2,3)
        batch_img = tio.Subject(image=tio.ScalarImage(tensor=batch_img.squeeze(0)))

        if (batch_lbl == 1):
            batch_img = transform_1(batch_img)
            batch_lbl = [1]
        else: batch_lbl = [0]
        grid_sampler = tio.inference.GridSampler(batch_img, patch_size, patch_overlap, )
        patch_loader = torch.utils.data.DataLoader(grid_sampler,batch_size=1,shuffle=True)  # To be tested: batch_size as 1 and more. If works for more, make a param

        scaler = GradScaler(enabled=True)
        for patches_batch in patch_loader:
            input_tensor = patches_batch['image'][tio.DATA].float().to(device)
            optimizer.zero_grad()
            with autocast(enabled=True):
                output = net(input_tensor)
                loss = criterion(output, batch_lbl.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            counter += 1

    print("Loss :",running_loss/counter)
    torch.save(net, PATH)

torch.save(net, PATH)

#################################################################################
"""VALIDATION"""
print("\n################# VALIDATION ########################")

def model_validation(batch_img, patch_size, patch_overlap):
    grid_sampler = tio.inference.GridSampler(batch_img, patch_size, patch_overlap, )
    patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=1,
                                               shuffle=True)

    running_loss = 0.0
    counter = 0
    for patches_batch in patch_loader:
        input_tensor = patches_batch['image'][tio.DATA].float().to(device)
        optimizer.zero_grad()
        with autocast(enabled=True):
            output = net(input_tensor)
            loss = criterion(output, torch.Tensor(batch_lbl).to(device))
        running_loss += loss.item()
        counter += 1

    return (running_loss / counter)

running_loss = 0.0
with torch.no_grad():
    for j in tqdm(range(len(validation_loader))):
        one_batch = next(iter(validation_loader))
        batch_img = one_batch['image'][tio.DATA]  # .float()
        batch_lbl = np.random.randint(2)
        batch_img = tio.Subject(image=tio.ScalarImage(tensor=batch_img.squeeze(0)))
        if (batch_lbl == 1):
            batch_img = transform_1(batch_img)
            batch_lbl = [1]
        else:
            batch_lbl = [0]

        loss = model_validation(batch_img,patch_size,patch_overlap)
        running_loss = running_loss + loss
         # To be tested: batch_size as 1 and more. If works for more, make a param
print("Loss of Model :", running_loss/len(validation_loader))
