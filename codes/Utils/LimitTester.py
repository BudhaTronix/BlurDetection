import torchvision
import ModelTrainer_SingleGPU_ResNet
import torchio as tio
from datetime import datetime
import torch
from torch.utils.tensorboard import SummaryWriter

Test_Obj = ModelTrainer_SingleGPU_ResNet

def callFunc(CorruptionDegree,translation,num_transforms,corruptionProbability):
    data = Test_Obj.BlurDetection(corruptionDegree=CorruptionDegree,translation=translation,num_transforms=num_transforms,corruptionProbability=corruptionProbability).datasetCreation()
    #start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S") + "  Cd = " + str(CorruptionDegree)
    start_time = "CD=" + str(CorruptionDegree) + ", T=" + str(translation) + ", NT=" + str(num_transforms)
    TBLOGDIR="runs/BlurDetection/{}".format(start_time)
    writer = SummaryWriter(TBLOGDIR)

    batch_size = 4

    for batch in data[0]:
        batch_img = batch['image'][tio.DATA].float()

        batch_lbl = []
        for i in range(batch_size):
            if torch.equal(batch_img[i, 0], batch_img[i, 1]):
                batch_lbl.append([0])
            else:
                batch_lbl.append([1])
        batch_img = batch_img[:, 1, ...].unsqueeze(1).squeeze(-1)
        print("Test")

        img_grid = torchvision.utils.make_grid(batch_img, normalize=True)
        temp = img_grid[:1, :, :]
        text = str(batch_lbl)
        writer.add_image(text, temp)
        break


callFunc(CorruptionDegree=10,translation=10,num_transforms=4,corruptionProbability=1)
callFunc(CorruptionDegree=100,translation=10,num_transforms=2,corruptionProbability=1)
callFunc(CorruptionDegree=10,translation=100,num_transforms=2,corruptionProbability=1)
callFunc(CorruptionDegree=10,translation=10,num_transforms=4,corruptionProbability=1)

