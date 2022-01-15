import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions import Test
from pathlib import Path
import seaborn as sns
import pandas as pd
from sklearn.metrics import f1_score, mean_squared_error


def set_axis_style(ax, labels):
    ax.xaxis.set_tick_params(direction='out')
    ax.xaxis.set_ticks_position('bottom')
    ax.set_xticks(np.arange(1, len(labels) + 1), labels=labels)
    ax.set_xlim(0.25, len(labels) + 0.75)
    # ax.set_xlabel('Sample name')


def plotGraph(label, preds):
    # Create a plot
    fig, ax = plt.subplots()
    ax.violinplot([label, preds])
    ax.set_title('Violin Plot')
    labels = ['label', 'preds']
    # set_axis_style(ax, labels)
    plt.show()

    # Scatter plots
    plt.figure(figsize=(10, 10))
    plt.scatter(label, preds)
    p1 = max(max(preds), max(label))
    p2 = min(min(preds), min(label))
    plt.plot([p1, p2], [p1, p2], 'b-')
    plt.xlabel('True Values', fontsize=15)
    plt.ylabel('Predictions', fontsize=15)
    plt.axis('equal')
    plt.show()


def showPredictions(file_path, obj):
    label_store = []
    pred_store = []
    itr = 0
    for file_name in sorted(file_path.glob("*.nii.gz")):
        label, preds = obj.test_singleFile(str(file_name))
        print(len(label), len(preds))
        if itr == 0:
            label_store = label
            pred_store = preds
        else:
            label_store.extend(label)
            pred_store.extend(preds)
        itr += 1
    print(len(label_store), len(pred_store))
    # plotGraph(label_store, pred_store)

    return label_store, pred_store


def m1(file_path):
    model_selection = 1
    model_path = "../model_weights/RESNET18.pth"
    output_path = "Outputs/ResNet18/"
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True, out_path=output_path)
    return showPredictions(file_path, obj)


def m2(file_path):
    model_selection = 2
    model_path = "../model_weights/RESNET50.pth"
    output_path = "Outputs/ResNet50/"
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True, out_path=output_path)
    return showPredictions(file_path, obj)


def m3(file_path):
    model_selection = 3
    model_path = "../model_weights/RESNET101.pth"
    output_path = "Outputs/ResNet101/"
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True, out_path=output_path)
    return showPredictions(file_path, obj)


def getDF(l1, p1, p2, p3):
    # Calling DataFrame after zipping both lists, with columns specified
    subject = range(len(l1))
    df = pd.DataFrame(list(zip(subject, l1, p1, p2, p3)), columns=['Sub', 'Label', 'ResNet18', 'ResNet50', 'ResNet101'])
    print(df)

    return df


def returnClass(no_of_class, array):
    class_intervals = 1 / no_of_class
    array[array <= 0] = 0
    array[array >= 1] = no_of_class - 1
    for i in range(no_of_class - 1, -1, -1):
        array[(array > (class_intervals * i)) & (array <= (class_intervals * (i + 1)))] = i
    return array


def main():
    no_class = 9
    file_path = Path("/home/budha/Desktop/test_files/")  # sys.argv[1]
    l1, p1 = m1(file_path)
    _, p2 = m2(file_path)
    _, p3 = m3(file_path)

    lbl_class = returnClass(no_class, np.asarray(l1))
    p1_class = returnClass(no_class, np.asarray(p1))
    p2_class = returnClass(no_class, np.asarray(p2))
    p3_class = returnClass(no_class, np.asarray(p3))

    # Accuracy Cal
    total_acc_p1 = np.sum(p1_class == lbl_class)
    total_acc_p2 = np.sum(p2_class == lbl_class)
    total_acc_p3 = np.sum(p3_class == lbl_class)

    # F1 Cal
    f1_micro_val_p1 = f1_score(lbl_class, p1_class, average='micro')
    f1_macro_val_p1 = f1_score(lbl_class, p1_class, average='macro')

    f1_micro_val_p2 = f1_score(lbl_class, p2_class, average='micro')
    f1_macro_val_p2 = f1_score(lbl_class, p2_class, average='macro')

    f1_micro_val_p3 = f1_score(lbl_class, p3_class, average='micro')
    f1_macro_val_p3 = f1_score(lbl_class, p3_class, average='macro')

    # RMSE Cal
    rmse_p1 = mean_squared_error(l1, p1, squared=False)
    rmse_p2 = mean_squared_error(l1, p2, squared=False)
    rmse_p3 = mean_squared_error(l1, p3, squared=False)

    print("\nAccuracy ResNet18     :", float(total_acc_p1 / len(l1) * 100), "%\n")
    print("Accuracy ResNet50     :", float(total_acc_p2 / len(l1) * 100), "%\n")
    print("Accuracy ResNet101    :", float(total_acc_p3 / len(l1) * 100), "%\n")

    print("\nF1 micro score ResNet18: ", f1_micro_val_p1)
    print("F1 micro score ResNet50: ", f1_micro_val_p2)
    print("F1 micro score ResNet101: ", f1_micro_val_p3)

    print("\nF1 macro score ResNet18: ", f1_macro_val_p1)
    print("F1 macro score ResNet50: ", f1_macro_val_p2)
    print("F1 macro score ResNet101: ", f1_macro_val_p3)

    print("\nRegression based RMSE ResNet18: ", rmse_p1)
    print("Regression based RMSE ResNet50: ", rmse_p2)
    print("Regression based RMSE ResNet101: ", rmse_p3)

    data = getDF(l1, p1, p2, p3)
    sns.set_theme(style="whitegrid")
    subjects = range(len(l1))
    data = pd.DataFrame(data, subjects, columns=['Label', 'ResNet18', 'ResNet50', 'ResNet101'])
    data = data.rolling(7).mean()
    sns.lineplot(data=data, palette="tab10", linewidth=2)
    plt.xlabel("Subject Number")
    plt.ylabel("SSIM Value")
    plt.show()


main()
