import pytorch_ssim
import torch
from torch.autograd import Variable
import torchio as tio
import os
import time
os.environ["CUDA_VISIBLE_DEVICES"] = "5"

since = time.time()
img1 = Variable(torch.rand(1, 150, 256, 256))
img2 = Variable(torch.rand(1, 150, 256, 256))
i1 = tio.ScalarImage(tensor=torch.rand(1, 128, 128, 1))
i2 = tio.ScalarImage(tensor=torch.rand(1, 128, 128, 1))

s1 = tio.Subject(image=i1, label=1)
s2 = tio.Subject(image=i2, label=2)
if torch.cuda.is_available():
    img1 = img1.cuda()
    img2 = img2.cuda()
a = pytorch_ssim.ssim(img1, img2)
print(a.size())
time_elapsed = time.time() - since
print(time_elapsed)
#ssim_loss = pytorch_ssim.SSIM(window_size=11)
#b = ssim_loss(img1, img2)
#print(b.size())
