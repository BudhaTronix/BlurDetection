import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from HelperFunctions import Test
from pathlib import Path
import seaborn as sns
import pandas as pd


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
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True,out_path=output_path)
    return showPredictions(file_path, obj)


def m2(file_path):
    model_selection = 2
    model_path = "../model_weights/RESNET50.pth"
    output_path = "Outputs/ResNet50/"
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True,out_path=output_path)
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
    df = pd.DataFrame(list(zip(subject, l1, p1, p2, p3)), columns=['Sub', 'Label', 'M1', 'M2', 'M3'])
    print(df)

    return df


def main():
    file_path = Path("/home/budha/Desktop/test_files/")  # sys.argv[1]
    l1, p1 = m1(file_path)
    _, p2 = m2(file_path)
    _, p3 = m3(file_path)

    data = getDF(l1, p1, p2, p3)
    sns.set_theme(style="whitegrid")
    subjects = range(len(l1))
    data = pd.DataFrame(data, subjects, columns=['Label', 'M1', 'M2', 'M3'])
    data = data.rolling(7).mean()

    sns.lineplot(data=data, palette="tab10", linewidth=2.5)
    plt.show()


main()
