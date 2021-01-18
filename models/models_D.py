import torch.nn as nn
import torch.nn.functional as F
import torch.nn
from VesselSeg_UNet3d import U_Net
from torchvision import transforms


class D_Net(nn.Module):
    def __init__(self):
        super(D_Net, self).__init__()
        self.u = U_Net()
        #self.conv1 = nn.Conv2d(1, 6, 5)
        #self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5)
        self.gAvgPool = nn.AvgPool3d((1,128,128))
        self.fc1 = nn.Linear(64, 32)
        self.fc2 = nn.Linear(32, 1)
        #self.fc3 = nn.Linear(84, 1)

    def forward(self, x):
        x = self.u(x)
        #print(x.shape)
        x = self.gAvgPool(x)
        x = x.squeeze()
        #print(x.shape)
        #torch.flatten(t)
        #x = self.pool(F.relu(self.conv1(x)))
        #x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 61 * 61) #16 * 5 * 5
        x = F.relu(self.fc1(x))
        x = torch.sigmoid(self.fc2(x))
        #x = self.fc2(x)
        #x = self.fc3(x)
        return x
"""
t = torch.rand(1, 1, 64, 256, 256).to("cuda:6")
#t = torch.rand(1, 1, 256, 256, 64).to("cuda:6")
t = t.squeeze(0)
t = t.squeeze(0)
compose = transforms.Compose([transforms.Resize((128,128))])
t = compose(t)
t = t.unsqueeze(0)
t = t.unsqueeze(0)
print(t.size())
net = D_Net().to("cuda:6")
out = net(t)
print(out)
#avgOut = nn.AvgPool3d((1,256,256))(out)
#print(avgOut.size())
"""