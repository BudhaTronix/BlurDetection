import torch
import torch.nn as nn

from models.models_D import D_Net
model = D_Net()
model = nn.DataParallel(model)#, device_ids=args.gpus).cuda()
if torch.cuda.device_count() > 1:
  print("Let's use", torch.cuda, "GPUs!")
device = "cuda:5"
model.to(device)

t = torch.rand(1, 1, 64, 128, 128)
out = model(t.to(device))