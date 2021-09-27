import itertools
import os
import time
import pandas as pd
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
import numpy as np
import torch
from tqdm import tqdm
from sklearn.metrics import confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "4"
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


def returnClass(no_of_class, array):
    class_intervals = 1 / no_of_class
    array[array == 0] = 0
    array[array >= 1] = no_of_class - 1
    for i in range(no_of_class):
        array[(array >= (class_intervals * i)) & (array < (class_intervals * (i + 1)))] = i
    return array


def plot_confusion_matrix(cm, classes, normalize=True, title='Confusion matrix', cmap=plt.cm.Blues):
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

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
    print("Variance               : ", np.std(errors))


def calMeanAbsError(expected, predicted):
    errors = list()
    for i in range(len(expected)):
        err = abs((expected[i] - predicted[i]))
        errors.append(err)

    print("Average Absolute Error : ", np.mean(errors))
    print("Standard deviation     : ", np.std(errors))
    print("Variance               : ", np.std(errors))


def testModel(dataloaders, no_class, modelPath, debug=False):
    print("Model In Testing mode")
    model = torch.load(modelPath)
    model.eval()
    model.to(device)
    running_corrects = 0
    batch_c = 0

    """since = time.time()
    nb_classes = no_class
    cf = torch.zeros(nb_classes, nb_classes)
    with torch.no_grad():
        for i, (inputs, classes) in enumerate(dataloaders):
            inputs = inputs.unsqueeze(1).float().to(device)
            classes = classes.to(device)
            outputs = model(inputs)
            op_class = returnClass(no_class, outputs)
            lbl_class = returnClass(no_class, classes)
            for t, p in zip(lbl_class, op_class):
                cf[t.long(), p.long()] += 1
    cf = cf.detach().cpu().numpy()
    cf = cf.astype(int)
    #print(cf)
    plot_confusion_matrix(cm=cf, classes=no_class)
    time_elapsed = time.time() - since
    print("Time taken :", time_elapsed)"""

    # Iterate over data.
    lbl = pred = []
    cm = 0
    with torch.no_grad():
        for batch in tqdm(dataloaders):
            image_batch, labels_batch = batch
            image_batch = image_batch.unsqueeze(1)
            image_batch = (image_batch - image_batch.min()) / \
                          (image_batch.max() - image_batch.min())  # Min Max normalization
            outputs = model(image_batch.float().to(device))
            output = outputs.detach().cpu().squeeze().tolist()
            label = labels_batch.tolist()
            if len(lbl) == 0:
                lbl = label
                pred = output
            else:
                lbl.extend(label)
                pred.extend(output)
            op_class = returnClass(no_class, np.array(output))
            lbl_class = returnClass(no_class, np.array(label))
            cm += confusion_matrix(lbl_class, op_class)
            out = np.sum(op_class == lbl_class)
            batch_c += len(batch[0])
            running_corrects += out
            if debug:
                print("Raw Output    : Model OP-> ", output, ", Label-> ", label)
                print("Class Output  : Model OP-> ", op_class, ", Label-> ", lbl_class, "\n")
                print("Accuracy per batch :", float(out / len(batch[0]) * 100), "%")

    # Plot Confusion Matrix of class
    plot_confusion_matrix(cm=cm, classes=no_class)

    # Print Accuracy on class basis
    total_acc = running_corrects
    print("\nTotal Correct :", total_acc, ", out of :", batch_c)
    print("Accuracy      :", float(total_acc / batch_c * 100), "%\n")

    lbl = np.float32(lbl)
    pred = np.float32(pred)
    calMeanAbsError(predicted=pred, expected=lbl)
    print()
    calMSE(predicted=pred, expected=lbl)
    visualize(pred, lbl, no_class)
