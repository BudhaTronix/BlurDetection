import os
import itertools
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torchio as tio
from PIL import ImageFont
from PIL import ImageDraw
from torchvision import transforms
from tqdm import tqdm
from Code.Utils.utils import returnClass
from sklearn.metrics import f1_score, mean_squared_error


def saveImage(images, output):
    # create grid of images
    plt.figure(figsize=(10, 10))
    for i in range(16):
        # Start next subplot.
        plt.subplot(4, 4, i + 1, title=" | Pred:" + str(np.around(output[i], 2)))
        plt.xticks([])
        plt.yticks([])
        # plt.grid(False)
        plt.imshow(images[i], cmap="gray")
    plt.show()


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
    """sns.lmplot(x="label", y="pred", data=df, x_estimator=np.mean)
    plt.show()
    df = pd.DataFrame(
        {'pred': returnClass(no_class, pred),
         'label': returnClass(no_class, label),
         })
    sns.boxplot(x="label", y="pred", data=df)
    plt.show()"""


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


def calMSE(predicted, expected):
    errors = list()
    for i in range(len(expected)):
        err = (expected[i] - predicted[i]) ** 2
        errors.append(err)
    print("Average MSE            : ", np.mean(errors))
    print("Standard deviation     : ", np.std(errors))
    print("Variance               : ", np.std(errors), "\n")


def calMeanAbsError(expected, predicted):
    errors = list()
    for i in range(len(expected)):
        err = abs((expected[i] - predicted[i]))
        errors.append(err)
    print("Average Absolute Error : ", np.mean(errors))
    print("Standard deviation     : ", np.std(errors))
    print("Variance               : ", np.std(errors), "\n")


def getCF(dataloaders, no_class, model, debug=False, device="cuda"):
    print("Generating CF In Testing mode")
    model.eval()
    model.to(device)
    nb_classes = no_class
    cf = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in tqdm(enumerate(dataloaders)):
            inputs = inputs.unsqueeze(1).float()
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())  # Min Max normalization
            classes = classes
            outputs = model(inputs.to(device))
            op_class = returnClass(no_class, outputs.squeeze().detach().cpu().numpy()).astype(int)
            lbl_class = returnClass(no_class, classes).numpy().astype(int)
            for t, p in zip(lbl_class, op_class):
                cf[t, p] += 1
    cf = cf.detach().cpu().numpy()
    cf = cf.astype(int)
    plot_confusion_matrix(cm=cf, classes=no_class)


def testModel(dataloaders, no_class, model, debug=False, device="cuda"):
    print("Model In Testing mode")
    model.eval()
    model.to(device)
    lbl = []
    pred = []
    with torch.no_grad():
        for i, (inputs, classes) in tqdm(enumerate(dataloaders)):
            inputs = inputs.unsqueeze(1).float()
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())  # Min Max normalization
            classes = classes
            outputs = model(inputs.to(device))
            if len(lbl) == 0:
                lbl = classes.cpu().tolist()
                pred = outputs.detach().cpu().squeeze().tolist()
            else:
                lbl.extend(classes.cpu().tolist())
                pred.extend(outputs.detach().cpu().squeeze().tolist())

    lbl_class = returnClass(no_class, np.asarray(lbl))
    pred_class = returnClass(no_class, np.asarray(pred))

    # Accuracy Cal
    total_acc = np.sum(pred_class == lbl_class)

    # F1 Cal
    f1_micro_val = f1_score(lbl_class, pred_class, average='micro')
    f1_macro_val = f1_score(lbl_class, pred_class, average='macro')

    # RMSE Cal
    rmse = mean_squared_error(lbl, pred, squared=False)

    print("\nTotal Correct :", total_acc, ", out of :", len(lbl))
    print("\nAccuracy      :", float(total_acc / len(lbl) * 100), "%\n")
    print("\nF1 micro score: ", f1_micro_val)
    print("F1 macro score: ", f1_macro_val)
    print("\nRegression based RMSE: ", rmse)

    visualize(pred, lbl, no_class)



def getModelOP(dataloaders, modelPath, debug=False, device="cuda"):
    print("Model In Testing mode")
    model = torch.load(modelPath)
    model.eval()
    model.to(device)
    with torch.no_grad():
        for batch in dataloaders:
            image_batch, labels_batch = batch
            inputs = image_batch.unsqueeze(1).float().to(device)
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())  # Min Max normalization
            outputs = model(inputs)
            outputs = outputs.detach().cpu().squeeze().tolist()
            if debug:
                for i in range(len(outputs)):
                    print("FileName: ", labels_batch[i] + " --  Model OP-> ", outputs[i])


def testModel_Image(niftyFilePath=None, model=None, transform=None, output_path="", device="cuda"):
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
        SSIM_lowest = 1
        slice_number = 0
        SSIM_ctr = []
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
                SSIM_ctr.append(output)
                if output < SSIM_lowest:
                    SSIM_lowest = output
                    slice_number = i
                if i == 0:
                    store = printLabel(inputs, output, disp)
                else:
                    temp = printLabel(inputs, output, disp)
                    store = torch.cat((store, temp), 0)
                if store_images:
                    store_output.append(output)
                    store_img.append(inputs.unsqueeze(2))
                    if itr % 16 == 0:
                        images = store_img
                        output = store_output
                        saveImage(images, output)
                        store_output = []
                        store_img = []
                    itr += 1

        print("Lowest SSIM val in Axis  ", axis, " ->   Slice Number : ", slice_number, " --  SSIM Val : ", output)
        print("Average SSIM val in Axis ", axis, "  : ", sum(SSIM_ctr) / len(SSIM_ctr))
        temp = tio.ScalarImage(tensor=store.permute(0, 3, 2, 1))
        print("Saved :", output_path + str(fileName) + "-" + str(axis) + '.nii.gz')
        temp.save(output_path + str(fileName) + "-" + str(axis) + '.nii.gz', squeeze=True)


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
