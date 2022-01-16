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
from sklearn.metrics import f1_score, mean_squared_error


class Test:
    def __init__(self):
        self.transform_val = (1, 224, 224)
        self.output_path = "Outputs/"
        self.defaultGPU = "cuda:0"
        self.modelPath_bestweights = "/home/budha/PycharmProjects/BlurDetection/Code/Test_Scripts/model_weights/"
        self.file_path = "/media/hdd_storage/Budha/Dataset/Otherdataset/Isotropic/"

    def saveImage(self, images, output):
        # create grid of images
        plt.figure(figsize=(10, 10))
        for i in range(16):
            plt.subplot(4, 4, i + 1, title=" | Pred:" + str(np.around(output[i], 2)))
            plt.xticks([])
            plt.yticks([])
            plt.imshow(images[i], cmap="gray")
        plt.show()

    def returnClass(self, no_of_class, array):
        class_intervals = 1 / no_of_class
        array[array <= 0] = 0
        array[array >= 1] = no_of_class - 1
        for i in range(no_of_class - 1, -1, -1):
            array[(array > (class_intervals * i)) & (array <= (class_intervals * (i + 1)))] = i
        return array

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
        num_classes = 1

        model_18 = models.resnet18(pretrained=True)
        model_50 = models.resnet50(pretrained=True)
        model_101 = models.resnet101(pretrained=True)

        model_18.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_18.fc.in_features
        model_18.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        model_50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_50.fc.in_features
        model_50.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        model_50.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        num_ftrs = model_50.fc.in_features
        model_50.fc = nn.Sequential(nn.Linear(num_ftrs, num_classes), nn.Sigmoid())

        return model_18, model_50, model_101

    def testModel(self):
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        device = torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')
        transform_Images = True
        transform = tio.CropOrPad(self.transform_val)
        try:
            model_18, model_50, model_101 = self.defineModel()
            model_18.load_state_dict(torch.load(self.modelPath_bestweights + "RESNET18.pth", map_location=device))
            model_50.load_state_dict(torch.load(self.modelPath_bestweights + "RESNET50.pth", map_location=device))
            model_101.load_state_dict(torch.load(self.modelPath_bestweights + "RESNET101.pth", map_location=device))
        except:
            print("Cannot load model weights, trying to load model")
            model_18 = torch.load(self.modelPath_bestweights + "RESNET18.pth", map_location=device)
            model_50 = torch.load(self.modelPath_bestweights + "RESNET50.pth", map_location=device)
            model_101 = torch.load(self.modelPath_bestweights + "RESNET101.pth", map_location=device)
        model_18.to(device)
        model_50.to(device)
        model_101.to(device)
        n_threads = 5
        mu = 0.0

        f1_micro_avg_sub_18 = 0
        f1_micro_avg_sub_50 = 0
        f1_micro_avg_sub_101 = 0

        f1_macro_avg_sub_18 = 0
        f1_macro_avg_sub_50 = 0
        f1_macro_avg_sub_101 = 0

        rmse_avg_sub_18 = 0
        rmse_avg_sub_50 = 0
        rmse_avg_sub_101 = 0

        c = 0
        tc = 0

        inpPath = Path(self.file_path)
        print("\n Loading Files")
        for file_name in sorted(inpPath.glob("*T2*.nii.gz")):
            if c == 100:
                break
            print("\nWorking on File             : ", file_name.name)
            subject = tio.Subject(image=tio.ScalarImage(file_name))
            # Motion Corruption
            sigma = 0.1  # sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))
            moco = MotionCorrupter(mode=2, n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=False)
            corruptions = [tio.Lambda(moco.perform, p=1)]
            corruption = tio.Compose(corruptions)
            imgOrig = subject["image"][tio.DATA]
            imgReg = corruption(imgOrig)[1:2, :, :, :]

            # Normalization
            imgOrig = (imgOrig - imgOrig.min()) / (imgOrig.max() - imgOrig.min())
            imgReg = (imgReg - imgReg.min()) / (imgReg.max() - imgReg.min())

            # Initialize data
            imgReg_op = imgReg
            imgOrig_op = imgOrig
            SSIM_pred_18 = []
            SSIM_pred_50 = []
            SSIM_pred_101 = []
            SSIM_lbl = []
            f1_micro_val_18 = 0
            f1_micro_val_50 = 0
            f1_micro_val_101 = 0

            f1_macro_val_18 = 0
            f1_macro_val_50 = 0
            f1_macro_val_101 = 0

            rmse_18 = 0
            rmse_50 = 0
            rmse_101 = 0

            count = 0
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

                # Traverse each direction of the image
                for slice in range(0, len(imgReg_op.squeeze())):
                    # Get the slice image of the subject
                    image = imgReg_op[:, slice:(slice + 1), :, :]
                    if transform_Images:
                        inputs = transform(image).squeeze()
                    else:
                        inputs = image.squeeze()
                    with torch.no_grad():
                        output_18 = model_18(
                            inputs.unsqueeze(0).unsqueeze(0).float().to(device)).detach().cpu().squeeze().tolist()

                        torch.cuda.empty_cache()
                        output_50 = model_50(
                            inputs.unsqueeze(0).unsqueeze(0).float().to(device)).detach().cpu().squeeze().tolist()

                        torch.cuda.empty_cache()
                        output_101 = model_101(
                            inputs.unsqueeze(0).unsqueeze(0).float().to(device)).detach().cpu().squeeze().tolist()

                        torch.cuda.empty_cache()

                        if not np.isnan(output_18) and not np.isnan(output_50) and not np.isnan(output_101):
                            SSIM_pred_18.append(output_18)
                            SSIM_pred_50.append(output_50)
                            SSIM_pred_101.append(output_101)
                            SSIM_lbl.append(ssim[slice].item())
                            tc += 1

                # Converting to Class
                no_class = 9
                lbl_class = self.returnClass(no_class, np.asarray(SSIM_lbl))
                pred_class_18 = self.returnClass(no_class, np.asarray(SSIM_pred_18))
                pred_class_50 = self.returnClass(no_class, np.asarray(SSIM_pred_50))
                pred_class_101 = self.returnClass(no_class, np.asarray(SSIM_pred_101))

                # F1 Cal
                f1_micro_val_18 += f1_score(lbl_class, pred_class_18, average='micro')
                f1_macro_val_18 += f1_score(lbl_class, pred_class_18, average='macro')

                f1_micro_val_50 += f1_score(lbl_class, pred_class_50, average='micro')
                f1_macro_val_50 += f1_score(lbl_class, pred_class_50, average='macro')

                f1_micro_val_101 += f1_score(lbl_class, pred_class_101, average='micro')
                f1_macro_val_101 += f1_score(lbl_class, pred_class_101, average='macro')

                # RMSE Cal
                rmse_18 += mean_squared_error(lbl_class, pred_class_18, squared=False)
                rmse_50 += mean_squared_error(lbl_class, pred_class_50, squared=False)
                rmse_101 += mean_squared_error(lbl_class, pred_class_101, squared=False)

                count += 1


            f1_micro_avg_sub_18 += f1_micro_val_18 / count
            f1_macro_avg_sub_18 += f1_macro_val_18 / count

            f1_micro_avg_sub_50 += f1_micro_val_50 / count
            f1_macro_avg_sub_50 += f1_macro_val_50 / count

            f1_micro_avg_sub_101 += f1_micro_val_101 / count
            f1_macro_avg_sub_101 += f1_macro_val_101 / count

            rmse_avg_sub_18 += rmse_18 / count
            rmse_avg_sub_50 += rmse_50 / count
            rmse_avg_sub_101 += rmse_101 / count

            c += 1

        print("Total number of Subjects         : ", c)
        print("Total number of images           : ", tc)

        print("##" * 10)
        print("Avg F1 micro score for RESNET 18 : ", f1_micro_avg_sub_18 / c)
        print("Avg F1 macro score for RESNET 18 : ", f1_macro_avg_sub_18 / c)
        print("Avg RMSE score for RESNET 18     : ", rmse_avg_sub_18 / c)

        print("##" * 10)
        print("Avg F1 micro score for RESNET 50 : ", f1_micro_avg_sub_50 / c)
        print("Avg F1 macro score for RESNET 50 : ", f1_macro_avg_sub_50 / c)
        print("Avg RMSE score for RESNET 50     : ", rmse_avg_sub_50 / c)

        print("##" * 10)
        print("Avg F1 micro score for RESNET 101 : ", f1_micro_avg_sub_101 / c)
        print("Avg F1 macro score for RESNET 101 : ", f1_macro_avg_sub_101 / c)
        print("Avg RMSE score for RESNET 101     : ", rmse_avg_sub_101 / c)


obj = Test()
obj.testModel()
