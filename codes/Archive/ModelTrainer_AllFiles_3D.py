import os
import random
import torch
import torchio as tio
from codes.ImageTransformer import transform_subject
from models.models_D import D_Net
import torch.optim as optim
from tqdm import tqdm
model_path = 'model_U_Net.pth'
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1/"
from torch.cuda.amp import autocast, GradScaler

device = "cuda:6"
net = D_Net().to(device)
criterion = torch.nn.BCELoss() # MSE/L1
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
patch_size = 64,128,128
patch_overlap = 0
def train_model(subject):
    batch_img = subject
    grid_sampler = tio.inference.GridSampler(batch_img, patch_size, patch_overlap,)
    patch_loader = torch.utils.data.DataLoader(grid_sampler,batch_size=1)  # To be tested: batch_size as 1 and more. If works for more, make a param
    batch_lbl = torch.Tensor(subject['label']).float().to(device)
    running_loss = 0.0
    counter = 0
    scaler = GradScaler(enabled=True)
    for patches_batch in tqdm(patch_loader):
        input_tensor = patches_batch['image'][tio.DATA].float().to(device)
        #print(input_tensor.shape)
        optimizer.zero_grad()
        with autocast(enabled=True):
            output = net(input_tensor)
            loss = criterion(output, batch_lbl)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        running_loss += loss.item()
        counter += 1

    torch.save(net.state_dict(), model_path)
    return running_loss/counter


print("Training Started")
for epoch in range(20):
    print("Epoch :",(epoch+1))
    arr = os.listdir(path)
    random.shuffle(arr)
    running_loss = 0.0
    for file_name in arr:
        #print(os.path.join(path, file_name))
        file_path = os.path.join(path, file_name)
        subject = tio.Subject(
            image=tio.ScalarImage(file_path),
            label=[0],
        )
        subject_blur = tio.Subject(
            image=transform_subject(3, tio.ScalarImage(file_path)),
            label=[1],
        )


        loss_subject = train_model(subject)
        loss_blur = train_model(subject_blur)

        running_loss = running_loss + (loss_subject + loss_blur)/ 2

    print("Loss :", running_loss/len(arr))



