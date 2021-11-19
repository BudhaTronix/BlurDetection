from pathlib import Path
import numpy as np
import torch
import torchio as tio
from motion import MotionCorrupter
import pytorch_ssim
import os
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import ImageFont
from PIL import ImageDraw
from torchvision import models
from torchvision import transforms


class Test:
    def __init__(self, model_selection, model_path, out_path, file_path):
        self.model_selection = model_selection
        self.transform_val = (1, 224, 224)
        self.output_path = out_path
        self.defaultGPU = "cuda:0"
        self.modelPath_bestweights = model_path
        self.file_path = file_path

    def saveImage(self, images, output):
        # create grid of images
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1, title=" | Pred:" + str(np.around(output[i], 2)))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images[i], cmap="gray")
        plt.show()

    @staticmethod
    def printLabel(img, pred, label, disp):
        trans = transforms.ToPILImage()
        image = trans(img.unsqueeze(0).float())
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        label = round(label, 2)
        pred = round(pred, 2)
        text = "L:" + str(label) + "|O:" + str(pred)
        draw.text((0, 0), str(text), 255, font=font)
        if disp:
            plt.imshow(image, cmap='gray')
            plt.show()
        trans1 = transforms.ToTensor()
        return trans1(image).unsqueeze(0)

    def getDevice(self):
        return torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')

    def defineModel(self):
        model = ""
        if self.model_selection == 1:
            model = models.resnet18(pretrained=True)
        elif self.model_selection == 2:
            model = models.resnet50(pretrained=True)
        elif self.model_selection == 3:
            model = models.resnet101(pretrained=True)
        model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_classes = 1
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())
        return model

    def testModel(self):
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        disp = False
        device = torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')
        transform_Images = True
        transform = tio.CropOrPad(self.transform_val)
        try:
            model = self.defineModel()
            model.load_state_dict(torch.load(self.modelPath_bestweights, map_location=device))
        except:
            print("Cannot load model weights, trying to load model")
            model = torch.load(self.modelPath_bestweights, map_location=device)
        model.to(device)
        n_threads = 5
        mu = 0.0
        inpPath = Path(self.file_path)
        print("\n Loading Files")
        for file_name in sorted(inpPath.glob("*T2*.nii.gz")):
            print("\nWorking on File             : ", file_name.name)
            fileName = file_name.name
            fileName = fileName.split(".nii.gz")[0]
            subject = tio.Subject(image=tio.ScalarImage(file_name))
            # Motion Corruption
            # sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))
            sigma = 0.1
            print("Corrupting with Sigma Value :", sigma)
            moco = MotionCorrupter(mode=2, n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=False)
            corruptions = [tio.Lambda(moco.perform, p=1)]
            corruption = tio.Compose(corruptions)
            imgOrig = subject["image"][tio.DATA]
            imgReg = corruption(imgOrig)[1:2, :, :, :]
            # Normalization
            imgOrig = (imgOrig - imgOrig.min()) / (imgOrig.max() - imgOrig.min())
            imgReg = (imgReg - imgReg.min()) / (imgReg.max() - imgReg.min())
            # Copy data
            imgReg_op = imgReg
            imgOrig_op = imgOrig
            SSIM_lowest = 1
            slice_number = 0
            SSIM_ctr = []
            for axis in range(0, 3):
                if axis == 1:
                    imgReg_op = imgReg.permute(0, 2, 3, 1)
                    imgOrig_op = imgOrig.permute(0, 2, 3, 1)
                if axis == 2:
                    imgReg_op = imgReg.permute(0, 3, 2, 1)
                    imgOrig_op = imgOrig.permute(0, 3, 2, 1)
                if torch.cuda.is_available():
                    imgReg_ssim = imgReg_op.cuda()
                    imgOrig_ssim = imgOrig_op.cuda()
                else:
                    imgReg_ssim = imgReg_op
                    imgOrig_ssim = imgOrig_op
                # Calculate the SSIM
                ssim = pytorch_ssim.ssim(imgOrig_ssim.double(), imgReg_ssim.double()).mean(0).mean(1).mean(
                    1).detach().cpu()
                for slice in range(0, len(imgReg_op.squeeze())):
                    # Get the slice image of the subject
                    image = imgReg_op[:, slice:(slice + 1), :, :]
                    if transform_Images:
                        inputs = transform(image).squeeze()
                    else:
                        inputs = image.squeeze()
                    with torch.no_grad():
                        output = model(inputs.unsqueeze(0).unsqueeze(0).float().to(device))
                        output = output.detach().cpu().squeeze().tolist()
                        if not np.isnan(output):
                            SSIM_ctr.append(output)
                        if slice == 0:
                            store = self.printLabel(inputs, output, ssim[slice].item(),  disp)
                        else:
                            temp = self.printLabel(inputs, output, ssim[slice].item(), disp)
                            store = torch.cat((store, temp), 0)
                        if output < SSIM_lowest and not np.isnan(output):
                            SSIM_lowest = output
                            slice_number = slice
                    # print("Axis:", axis, "  Slice Number:", slice, "  Prediction:", output, "  SSIM:",ssim[slice].item())

                print("\nLowest SSIM val in Axis  ", axis, " ->   Slice Number : ", slice_number, " --  SSIM Val : ",
                      SSIM_lowest)
                print("Average Predicted SSIM val in Axis ", axis, "  : ", sum(SSIM_ctr) / len(SSIM_ctr))
                print("Average Actual SSIM val in Axis    ", axis, "  : ", ssim.mean().item())
                print("Saved :", self.output_path + str(fileName) + "-" + str(axis) + '.nii.gz')
                temp = tio.ScalarImage(tensor=store.permute(0, 3, 2, 1))
                temp.save(self.output_path + str(fileName) + "-" + str(axis) + '.nii.gz', squeeze=True)

            break
