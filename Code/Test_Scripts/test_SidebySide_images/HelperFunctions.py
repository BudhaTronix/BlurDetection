import os
import pydicom
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torchio as tio
from PIL import ImageFont
from PIL import ImageDraw
from torchvision import models
from torchvision import transforms
# from skimage.metrics import structural_similarity as ssim
from Code.Utils.pytorch_ssim import ssim


class Test:
    def __init__(self, model_selection, model_path, transform, out_path):
        self.model_selection = model_selection
        self.transform_val = (1, 224, 224)
        self.transform = transform
        self.output_path = out_path
        self.defaultGPU = "cuda:0"
        self.modelPath_bestweights = model_path
        self.save = True


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

    def testModel_singleImage(self, niftyFilePath=None, model=None, transform=None, output_path="", device="cuda"):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        model.eval()
        model.to(device)
        fileName = niftyFilePath.split("/")[-1].split(".nii.gz")[0]
        print(fileName)
        disp = False
        Subject = tio.ScalarImage(niftyFilePath)[tio.DATA].squeeze()
        print(Subject.shape)
        size = len(Subject)
        image = Subject[0:int(size / 2), :, :]
        orig_image = Subject[int(size / 2):size, :, :]
        SSIM_lowest = 1
        slice_number = 0
        SSIM_ctr = []
        SSIM_pred = []
        SSIM_actual = []
        # Traverse through the dataset to get the ssim values

        store = None
        for i in range(0, len(Subject.permute(2, 0, 1))):
            orig = orig_image[:, :, i:(i + 1)].permute(2, 1, 0)
            img = image[:, :, i:(i + 1)].permute(2, 1, 0)
            if transform is not None:
                img = transform(img.unsqueeze(0)).squeeze()
                orig = transform(orig.unsqueeze(0)).squeeze()
            else:
                img = img.squeeze()
                orig = orig.squeeze()
            # ssim_none = ssim(img.detach().cpu().numpy(), orig.detach().cpu().numpy(), data_range=1)
            ssim_none = ssim(img.detach().cpu().unsqueeze(0).unsqueeze(0), orig.detach().cpu().unsqueeze(0).unsqueeze(0)).squeeze().mean().item()
            SSIM_actual.append(ssim_none)
            with torch.no_grad():
                output = model(img.unsqueeze(0).unsqueeze(0).float().to(device))
                output = output.detach().cpu().squeeze().tolist()
                SSIM_pred.append(output)
                if not np.isnan(output):
                    SSIM_ctr.append(output)
                    # print("Axis : ", axis, " Slice Number: ", i, " --  Model OP-> ", output)
                if i == 0:
                    store = self.printLabel(Subject[:, :, i:(i + 1)].squeeze(), output, ssim_none, disp)
                else:
                    temp = self.printLabel(Subject[:, :, i:(i + 1)].squeeze(), output, ssim_none, disp)
                    store = torch.cat((store, temp), 0)
                if output < SSIM_lowest and not np.isnan(output):
                    SSIM_lowest = output
                    slice_number = i

        print("\nLowest SSIM val in Slice Number : ", slice_number, " --  SSIM Val : ", SSIM_lowest)
        print("Average SSIM val : ", sum(SSIM_ctr) / len(SSIM_ctr))
        if self.save:
            temp = tio.ScalarImage(tensor=store.permute(0, 3, 2, 1))
            print("Saved :", output_path + str(fileName) + '.nii.gz')
            temp.save(output_path + str(fileName) + '.nii.gz', squeeze=True)

        return SSIM_actual, SSIM_pred

    def test_singleFile(self, image_file):
        transform = None
        device = torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')
        if self.transform:
            transform = tio.CropOrPad(self.transform_val)
        try:
            model = self.defineModel()
            model.load_state_dict(torch.load(self.modelPath_bestweights, map_location=device))
        except:
            print("Cannot load model weights, trying to load model")
            model = torch.load(self.modelPath_bestweights, map_location=device)
        print("Testing model with saved weights")

        label, preds = self.testModel_singleImage(niftyFilePath=image_file, model=model,
                                   transform=transform, output_path=self.output_path)

        return label, preds
