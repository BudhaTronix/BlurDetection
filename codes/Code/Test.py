import os
import numpy as np
import torch

from torch.cuda.amp import autocast
from tqdm import tqdm

precision = 4
flag = True
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda"


def testModel(model, dataloaders, modelweightPath):
    print("Model In Testing mode")
    model.load_state_dict(torch.load(modelweightPath))
    model.eval()

    running_corrects = 0
    # Iterate over data.
    for batch in tqdm(dataloaders):
        image_batch, labels_batch = batch
        image_batch = image_batch.unsqueeze(1)

        with torch.set_grad_enabled():
            with autocast(enabled=True):
                image_batch = (image_batch - image_batch.min()) / \
                              (image_batch.max() - image_batch.min())  # Min Max normalization
                outputs = model(image_batch.float().to(device))

            running_corrects += np.sum(np.around(outputs.detach().cpu().squeeze().numpy(),
                                                 decimals=precision) == np.around(labels_batch.numpy(),
                                                                                  decimals=precision))

    total_acc = running_corrects / len(dataloaders)
    print("Overall Accuracy :", total_acc)