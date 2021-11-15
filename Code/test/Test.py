import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torchio as tio
from PIL import ImageFont
from PIL import ImageDraw
from torchvision import models
from torchvision import transforms
from tqdm import tqdm
from Code.Utils.utils import returnClass


class Test:
    def __init__(self):
        self.path = ""
        self.getTransformation()
        self.model_selection = 1
        self.transform_val = (1, 224, 224)
        self.testFile = ""
        self.output_path = ""
        self.defaultGPU = "cuda"

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
    def visualize(pred, label, no_class):
        sns.set_theme(color_codes=True)
        df = pd.DataFrame(
            {'pred': pred,
             'label': label,
             })
        sns.regplot(x="label", y="pred", data=df)
        plt.show()
        sns.lmplot(x="label", y="pred", data=df)
        plt.show()
        sns.lmplot(x="label", y="pred", data=df, x_estimator=np.mean)
        plt.show()
        df = pd.DataFrame(
            {'pred': returnClass(no_class, pred),
             'label': returnClass(no_class, label),
             })
        sns.boxplot(x="label", y="pred", data=df)
        plt.show()

    @staticmethod
    def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Displaying Normalized confusion matrix")
        else:
            print('Displaying Confusion matrix, without normalization')

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        classes = list(range(0, classes))
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.show()

    @staticmethod
    def calMSE(predicted, expected):
        errors = list()
        for i in range(len(expected)):
            err = (expected[i] - predicted[i]) ** 2
            errors.append(err)
        print("Average MSE            : ", np.mean(errors))
        print("Standard deviation     : ", np.std(errors))
        print("Variance               : ", np.std(errors), "\n")

    @staticmethod
    def calMeanAbsError(expected, predicted):
        errors = list()
        for i in range(len(expected)):
            err = abs((expected[i] - predicted[i]))
            errors.append(err)
        print("Average Absolute Error : ", np.mean(errors))
        print("Standard deviation     : ", np.std(errors))
        print("Variance               : ", np.std(errors), "\n")

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

    def testModel_multipleImages(self, dataloaders, no_class, model, debug=False, device="cuda"):
        model.eval()
        model.to(device)
        running_corrects = 0
        batch_c = 0
        lbl = pred = []
        nb_classes = no_class
        cf = torch.zeros(nb_classes, nb_classes)
        with torch.no_grad():
            for i, (inputs, classes) in tqdm(enumerate(dataloaders)):
                inputs = inputs.unsqueeze(1).float()
                inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())  # Min Max normalization
                classes = classes  # .to(device)
                outputs = model(inputs.to(device))
                if len(lbl) == 0:
                    lbl = classes.cpu().tolist()
                    pred = outputs.detach().cpu().squeeze().tolist()
                else:
                    lbl.extend(classes.cpu().tolist())
                    pred.extend(outputs.detach().cpu().squeeze().tolist())

                op_class = returnClass(no_class, outputs.squeeze().detach().cpu().numpy()).astype(int)
                lbl_class = returnClass(no_class, classes).numpy().astype(int)
                for t, p in zip(lbl_class, op_class):
                    cf[t, p] += 1

                out = np.sum(op_class == lbl_class)
                batch_c += len(inputs)
                running_corrects += out
                if debug:
                    print("Raw Output    : Model OP-> ", outputs.detach().cpu().squeeze().tolist(), ", Label-> ",
                          classes.cpu().tolist())
                    print("Class Output  : Model OP-> ", op_class, ", \nLabel-> ", lbl_class, "\n")
                    print("Accuracy per batch :", float(out / len(inputs) * 100), "%")

        cf = cf.detach().cpu().numpy()
        cf = cf.astype(int)
        self.plot_confusion_matrix(cm=cf, classes=no_class)

        total_acc = running_corrects
        print("\nTotal Correct :", total_acc, ", out of :", batch_c)
        print("Accuracy      :", float(total_acc / batch_c * 100), "%\n")

        lbl = np.float32(lbl)
        pred = np.float32(pred)
        self.calMeanAbsError(predicted=pred, expected=lbl)
        self.calMSE(predicted=pred, expected=lbl)

    def testModel_singleImage(self, niftyFilePath=None, model=None, transform=None, output_path="", device="cuda"):
        if not os.path.isdir(output_path):
            os.mkdir(output_path)
        model.eval()
        model.to(device)
        store_images = False
        fileName = niftyFilePath.split("/")[-1].split(".nii.gz")[0]
        disp = False
        Subject = tio.ScalarImage(niftyFilePath)[tio.DATA].squeeze()
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
                    # print("Axis : ", axis, " Slice Number: ", i, " --  Model OP-> ", output)
                    if i == 0:
                        store = self.printLabel(inputs, output, disp)
                    else:
                        temp = self.printLabel(inputs, output, disp)
                        store = torch.cat((store, temp), 0)
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
            print("Saved :", output_path + str(fileName) + "-" + str(axis) + '.nii.gz')
            temp.save(output_path + str(fileName) + "-" + str(axis) + '.nii.gz', squeeze=True)

    def test_singleFile(self, transform_Images, custom_model_path=None):
        transform = None
        device = torch.device(self.defaultGPU if torch.cuda.is_available() else 'cpu')
        if custom_model_path is not None:
            self.modelPath_bestweights = custom_model_path
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
