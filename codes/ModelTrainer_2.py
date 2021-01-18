import torch
import torchio as tio
from models_D import D_Net
from torchvision import transforms
import numpy as np

from DatasetLoader import datasetLoader
import torch.optim as optim

#img_path = 'IXI002-Guys-0828-T1.nii.gz'
t1_inp_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1_mini/"
t2_inp_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T2/"
transform_type = 3
training_split_ratio = 0.9

print("##########Dataset Loader################")
T1_Subs,T1_dataset = datasetLoader(t1_inp_path)
print('Number of subjects in T1 dataset:', len(T1_dataset))
#T2_Subs,T2_dataset = datasetLoader(t2_inp_path)
#print('Number of subjects in T2 dataset:', len(T2_dataset))
print("########################################\n\n")


#Printing
print("##########Sample Printing###############")
sample_subject = T1_dataset[0]
#show_slices(sample_subject['image'])
print("Label : ",sample_subject['label'])
print("########################################\n\n")

num_subjects = len(T1_dataset)
num_training_subjects = int(training_split_ratio * num_subjects)

np.random.shuffle(T1_Subs)

training_subjects = T1_Subs[:num_training_subjects]
validation_subjects = T1_Subs[num_training_subjects:]

training_set = tio.SubjectsDataset(
    training_subjects)

validation_set = tio.SubjectsDataset(
    validation_subjects)

print("Training Set",len(training_set))
print("Validation Set",len(validation_set))

training_batch_size = 1
validation_batch_size = 1

training_loader = torch.utils.data.DataLoader(
    training_set,
    batch_size=training_batch_size,
    shuffle=True,
    #num_workers=torch.multiprocessing.cpu_count(),
)

validation_loader = torch.utils.data.DataLoader(
    validation_set,
    batch_size=validation_batch_size,
    #num_workers=torch.multiprocessing.cpu_count(),
)


device = "cpu"#"cuda:6"
net = D_Net().to(device)

PATH = './cifar_net.pth'
net.load_state_dict(torch.load(PATH))
#criterion = torch.nn.BCEWithLogitsLoss()
criterion = torch.nn.BCELoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
#optimizer = optim.Adam(net.parameters(), lr=0.001)
print(len(training_loader))
for epoch in range(30):
    running_loss = 0.0
    print("\n################# Epoch :", (epoch+1), " ########################")

    for j in range(len(training_loader)):
        #print("\nBatch ", (j+1), ":")
        one_batch = next(iter(training_loader))
        batch_img = one_batch['image'][tio.DATA].float()
        loop_size = batch_img.size()[-1]
        for i in range(int(loop_size/64)):
            optimizer.zero_grad()
            batch_img = one_batch['image'][tio.DATA].float()
            #print(batch_img)
            start = i * 64
            end = (i+1) * 64
            if (i>0):
                start += 1
                end += 1
            #print("START - END ", start, end)
            batch_lbl = torch.Tensor(one_batch['label']).float().to(device)
            batch_img = batch_img.reshape([1, 1, loop_size, 256, 256])
            test = batch_img[:, :, start:end , :, :]
            t = test.squeeze()
            compose = transforms.Compose([transforms.Resize((128, 128))])
            t = compose(t)
            t = t.unsqueeze(0)
            t = t.unsqueeze(0)
            output = net(t.to(device))
            loss = criterion(output, batch_lbl)
            if(output>0.5): output = 1
            else: output = 0
            #print("Output: ", output, "  Label:", batch_lbl.item(), "   Loss:", loss.item())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
    print("Loss :", running_loss/len(training_loader))
    torch.save(net.state_dict(), PATH)

PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)