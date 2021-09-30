import itertools
import os
import sys
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torch

try:
    from utils import returnClass
except ImportError:
    sys.path.insert(1, '/project/mukhopad/tmp/BlurDetection_tmp/codes/Utils/')
    from utils import returnClass

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = "cuda"


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


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Displaying Normalized confusion matrix")
    else:
        print('Displaying Confusion matrix, without normalization')

    # print(cm)
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


def testModel(dataloaders, no_class, modelPath, debug=False):
    print("Model In Testing mode")
    model = torch.load(modelPath)
    model.eval()
    model.to(device)
    running_corrects = 0
    batch_c = 0
    lbl = pred = []
    nb_classes = no_class
    cf = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders):
            inputs = inputs.unsqueeze(1).float().to(device)
            inputs = (inputs - inputs.min()) / (inputs.max() - inputs.min())  # Min Max normalization
            classes = classes.to(device)
            outputs = model(inputs)
            if len(lbl) == 0:
                lbl = classes.cpu().tolist()
                pred = outputs.detach().cpu().squeeze().tolist()
            else:
                lbl.extend(classes.cpu().tolist())
                pred.extend(outputs.detach().cpu().squeeze().tolist())
            op_class = returnClass(no_class, outputs.squeeze())
            lbl_class = returnClass(no_class, classes)
            for t, p in zip(lbl_class, op_class):
                cf[t.long(), p.long()] += 1

            out = np.sum(op_class.cpu().detach().numpy() == lbl_class.cpu().numpy())
            batch_c += len(inputs)
            running_corrects += out
            if debug:
                print("Raw Output    : Model OP-> ", outputs.detach().cpu().squeeze().tolist(), ", Label-> ",
                      classes.cpu().tolist())
                print("Class Output  : Model OP-> ", op_class, ", Label-> ", lbl_class, "\n")
                print("Accuracy per batch :", float(out / len(inputs) * 100), "%")

    cf = cf.detach().cpu().numpy()
    cf = cf.astype(int)
    plot_confusion_matrix(cm=cf, classes=no_class)

    total_acc = running_corrects
    print("\nTotal Correct :", total_acc, ", out of :", batch_c)
    print("Accuracy      :", float(total_acc / batch_c * 100), "%\n")

    lbl = np.float32(lbl)
    pred = np.float32(pred)
    calMeanAbsError(predicted=pred, expected=lbl)
    calMSE(predicted=pred, expected=lbl)
    visualize(pred, lbl, no_class)


def getModelOP(dataloaders, modelPath, debug=False):
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