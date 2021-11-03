from __future__ import division
from __future__ import print_function

import copy
import time
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# torch.autograd.set_detect_anomaly(True)

scaler = GradScaler()


def saveImage(images, labels, output):
    # create grid of images
    figure = plt.figure(figsize=(10, 10))
    for i in range(16):
        # Start next subplot.
        plt.subplot(4, 4, i + 1, title="Lbl:" + str(np.around(labels[i].item(), 2)) + " | Pred:"
                                       + str(np.around(output[i].item(), 2)))
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i].permute(1, 2, 0), cmap="gray")

    return figure


def trainModel(dataloaders, modelPath, modelPath_bestweight, num_epochs, model,
               criterion, optimizer, log=False, log_dir="runs/", device="cuda", isMultiGPU=False):
    model.to(device)
    criterion.to(device)
    precision = 1  # Sets the decimal value
    if log:
        start_time = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
        TBLOGDIR = log_dir.format(start_time)
        writer = SummaryWriter(TBLOGDIR)
    best_model_wts = ""
    best_acc = 0.0
    best_val_loss = 99999
    since = time.time()
    slice = 0
    for epoch in range(0, num_epochs):
        # torch.cuda.empty_cache()
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        for phase in [0, 1]:
            if phase == 0:
                print("\nModel In Training mode")
                model.train()  # Set model to training mode
            else:
                print("\nModel In Validation mode")
                model.eval()  # Set model to evaluate mode
                slice = np.random.randint(0, len(dataloaders[phase]))
            running_loss = 0.0
            running_corrects = 0
            itr = 0
            # Iterate over data.
            for batch in tqdm(dataloaders[phase]):
                image_batch, labels_batch = batch
                image_batch = image_batch.unsqueeze(1)
                image_batch = (image_batch - image_batch.min()) / \
                              (image_batch.max() - image_batch.min())  # Min Max normalization
                optimizer.zero_grad()
                # forward
                with torch.set_grad_enabled(phase == 0):
                    with autocast(enabled=True):
                        outputs = model(image_batch.float().to(device))
                        loss = criterion(outputs.squeeze(1).float(), labels_batch.float().to(device))

                    # backward + optimize only if in training phase
                    if phase == 0:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                    if log and phase == 1 and itr == slice and epoch % 50 == 0:
                        images = image_batch[0:16]
                        labels = labels_batch[0:16]
                        output = outputs[0:16].detach().cpu().squeeze().numpy()
                        figure = saveImage(images, labels, output)
                        text = "Epoch : " + str(epoch)
                        # write to tensorboard
                        writer.add_figure(text, figure, epoch)
                    # statistics
                    running_loss += loss.detach().cpu().item()
                    running_corrects += np.sum(np.around(outputs.detach().cpu().squeeze().numpy(),
                                                         decimals=precision) == np.around(labels_batch.numpy(),
                                                                                          decimals=precision))
                itr += 1
            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])

            if phase == 0:
                mode = "Train"
                if log:
                    writer.add_scalar("Loss/Train", epoch_loss, epoch)
                    writer.add_scalar("Acc/Train", epoch_acc, epoch)
            else:
                mode = "Val"
                if log:
                    writer.add_scalar("Loss/Validation", epoch_loss, epoch)
                    writer.add_scalar("Acc/Validation", epoch_acc, epoch)

            print('\n{} Loss: {:.4f} Acc: {:.4f}'.format(mode, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 1 and (epoch_acc >= best_acc or epoch_loss < best_val_loss):
                print("\nSaving the best model weights")
                best_acc = epoch_acc
                best_val_loss = epoch_loss
                # best_model_wts = copy.deepcopy(model.state_dict())
                if isMultiGPU:
                    torch.save(model.module.state_dict(), modelPath_bestweight)  # For multi GPU
                else:
                    torch.save(model.state_dict(), modelPath_bestweight)
                # torch.save(model, modelPath_bestweight)

    time_elapsed = time.time() - since
    print('\nTraining complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('\nBest val Acc: {:4f}'.format(best_acc))
    print("\nSaving the model")
    # save the model
    torch.save(model, modelPath)
    # load best model weights

    # print("\nSaving the best weights model")
    # model.load_state_dict(best_model_wts)
    # torch.save(model, modelPath_bestweight)
