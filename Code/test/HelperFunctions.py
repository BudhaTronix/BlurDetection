import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from PIL import ImageFont
from PIL import ImageDraw
from torchvision import models
from torchvision import transforms


class Test:
    def __init__(self, model_selection, model_path, testFile, out_path):
        self.model_selection = model_selection
        self.transform_val = (1, 224, 224)
        self.testFile = testFile
        self.output_path = out_path
        self.defaultGPU = "cuda:0"
        self.modelPath_bestweights = model_path

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
    def printLabel(img, label, disp):
        trans = transforms.ToPILImage()
        image = trans(img.unsqueeze(0))
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default()
        label = round(label, 2)
        draw.text((0, 0), str(label), 255, font=font)
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

    def testModel_singleImage(self, niftyFilePath=None, model=None, transform=None, output_path="", device="cuda"):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        model.eval()
        model.to(device)
        store_images = False
        fileName = niftyFilePath.split("/")[-1].split(".nii.gz")[0]
        disp = False
        Subject = tio.ScalarImage(niftyFilePath)[tio.DATA].squeeze()
        SSIM_lowest = 1
        slice_number = 0
        SSIM_ctr = []
        # Traverse through the dataset to get the ssim values
        for axis in range(0, 3):
            if axis == 0:
                Subject = Subject  # Coronal
            elif axis == 1:
                Subject = Subject.permute(2, 1, 0)  # Transverse
            elif axis == 2:
                Subject = Subject.permute(1, 2, 0)  # Axial
            itr = 1
            store = None
            for i in range(0, len(Subject)):
                img = Subject[i:(i + 1), :, :]
                if transform is not None:
                    img_transformed = transform(img.unsqueeze(0)).squeeze()
                else:
                    img_transformed = img.squeeze()
                inputs = (img_transformed - img_transformed.min()) / (
                        img_transformed.max() - img_transformed.min())  # Min Max normalization
                with torch.no_grad():
                    output = model(inputs.unsqueeze(0).unsqueeze(0).float().to(device))
                    output = output.detach().cpu().squeeze().tolist()
                    if not np.isnan(output):
                        SSIM_ctr.append(output)
                        # print("Axis : ", axis, " Slice Number: ", i, " --  Model OP-> ", output)
                    if i == 0:
                        store = self.printLabel(inputs, output, disp)
                    else:
                        temp = self.printLabel(inputs, output, disp)
                        store = torch.cat((store, temp), 0)
                    if output < SSIM_lowest and not np.isnan(output):
                        SSIM_lowest = output
                        slice_number = i
                    if store_images:
                        store_output.append(output)
                        store_img.append(inputs.unsqueeze(2))
                        if itr % 16 == 0:
                            images = store_img
                            output = store_output
                            self.saveImage(images, output)
                            store_output = []
                            store_img = []
                        itr += 1
            temp = tio.ScalarImage(tensor=store.permute(0, 3, 2, 1))
            print("\nLowest SSIM val in Axis  ", axis, " ->   Slice Number : ", slice_number, " --  SSIM Val : ", SSIM_lowest)
            print("Average SSIM val in Axis ", axis, "  : ", sum(SSIM_ctr) / len(SSIM_ctr))
            print("Saved :", output_path + str(fileName) + "-" + str(axis) + '.nii.gz')
            temp.save(output_path + str(fileName) + "-" + str(axis) + '.nii.gz', squeeze=True)

    def test_singleFile(self, transform_Images, ):
        transform = None
        device = torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')
        if transform_Images:
            transform = tio.CropOrPad(self.transform_val)
        try:
            model = self.defineModel()
            model.load_state_dict(torch.load(self.modelPath_bestweights, map_location=device))
        except:
            print("Cannot load model weights, trying to load model")
            model = torch.load(self.modelPath_bestweights)
        print("Testing model with saved weights")
        self.testModel_singleImage(niftyFilePath=self.testFile, model=model,
                                   transform=transform, output_path=self.output_path)
