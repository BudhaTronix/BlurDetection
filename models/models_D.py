import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from torch.cuda.amp import autocast

from VesselSeg_UNet3d import U_Net
from torchvision import transforms

class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.u = U_Net()
        self.gAvgPool = nn.AvgPool3d((1,128,128))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)

    def forward(self, x):
        # = x.to(device_1) #Added New
        x = self.u(x)
        #x = x.to(device_2)#Added New
        x = self.gAvgPool(x)
        x = x.squeeze()
        x = F.relu(self.fc1(x))
        x = self.fc2(x)

        return x


"""""
with autocast(enabled=True):
    t = torch.rand(4, 1, 64, 128, 128).to("cuda:5")
    net = D_Net().to("cuda:5")
    model = nn.DataParallel(net, device_ids=[5,6])
    out = model(t)
    print(out.size())
    batch_lbl = 1
    import numpy as np
    print(torch.Tensor(batch_lbl*np.ones([4,1])))
    #avgOut = nn.AvgPool3d((1,256,256))(out)
    #print(avgOut.size())
"""





