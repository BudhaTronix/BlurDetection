import sys
from DatasetLoader import datasetLoader
from models_D import D_Net
import torchio as tio
import torch.nn
from tqdm import tqdm
from Display_Subjects import show_slices
device = "cuda:6"
inp_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1_mini"
_, dataset = datasetLoader(inp_path)
print('Number of subjects in T1 dataset:', len(dataset))

#PATH = '../model_weights/BlurDetection_ModelWeights.pth'
PATH = '../model_weights/BlurDetection_ModelWeights_SinlgeGPU.pth' #model_U_Net.pth'
net = torch.load(PATH)
net.eval().to(device)

patch_size = 64,128,128
patch_overlap = 0
c_sample = dataset[18]
ic_sample = dataset[32]
L1 = torch.Tensor(c_sample['label']).long()
L2 = torch.Tensor(ic_sample['label']).long()
classes = ('Not Blur','Blur')


show_slices(c_sample['image'])
show_slices(ic_sample['image'])
#####################################################################
with torch.no_grad():
    batch_img = c_sample['image'][tio.DATA]
    batch_img = batch_img.permute(0, 3, 1, 2)
    batch_img = tio.Subject(image=tio.ScalarImage(tensor=batch_img))

    grid_sampler = tio.inference.GridSampler(batch_img, patch_size, patch_overlap,)
    patch_loader = torch.utils.data.DataLoader(grid_sampler,batch_size=1)
    #aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
    tot_out = 0.0
    counter = 0
    for patches_batch in tqdm(patch_loader):
        input_tensor = patches_batch['image'][tio.DATA].float().to(device)
        output = torch.sigmoid(net(input_tensor))
        print(output)
        tot_out = tot_out + output
        counter +=1
    output_tensor = tot_out/counter
    if(output_tensor>0.5): output_tensor = 1
    else: output_tensor = 0
    out_1 = output_tensor

    print("Another pic")
    #######################################################################

    batch_img = ic_sample
    grid_sampler = tio.inference.GridSampler(batch_img, patch_size, patch_overlap,)
    patch_loader = torch.utils.data.DataLoader(grid_sampler,batch_size=1)
    #aggregator = tio.inference.GridAggregator(grid_sampler, overlap_mode="average")
    tot_out = 0.0
    counter = 0
    for patches_batch in tqdm(patch_loader):
        input_tensor = patches_batch['image'][tio.DATA].float().to(device)
        output = torch.sigmoid(net(input_tensor))
        tot_out = tot_out + output
        counter +=1

    output_tensor = tot_out/counter
    if(output_tensor>0.5): output_tensor = 1
    else: output_tensor = 0
    out_2 = output_tensor


#######################################################################
print("Actual Label of Image 1 : ",classes[L1.item()])
print("Actual Label of Image 2 : ",classes[L2.item()])
print("Predicted Label of Image 1 : ",classes[out_1])
print("Predicted Label of Image 2 : ",classes[out_2])

print("END")
sys.exit(0)
#########################################################################