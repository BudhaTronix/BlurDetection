import os
import sys
import matplotlib.pyplot as plt
from HelperFunctions import Test
from pathlib import Path


def plotGraph(label, preds):
    plt.scatter(label, preds)
    plt.show()
    fig, ax = plt.subplots()

    # Create a plot
    ax.violinplot([label, preds])

    # Add title
    ax.set_title('Violin Plot')
    plt.show()


def m1():
    model_selection = 1
    model_path = "../model_weights/RESNET18.pth"
    output_path = "Outputs/ResNet18/"

    file_path = Path("/home/budha/Desktop/test_files/")  # sys.argv[1]
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True,
               out_path=output_path)
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
    plotGraph(label_store, pred_store)


def m2():
    model_selection = 2
    model_path = "../model_weights/RESNET50.pth"
    output_path = "Outputs/ResNet50/"

    file_path = Path("/home/budha/Desktop/test_files/")  # sys.argv[1]
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True,
               out_path=output_path)
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
    plotGraph(label_store, pred_store)


def m3():
    model_selection = 3
    model_path = "../model_weights/RESNET101.pth"
    output_path = "Outputs/ResNet101/"

    file_path = Path("/home/budha/Desktop/test_files/")  # sys.argv[1]
    obj = Test(model_selection=model_selection, model_path=model_path, transform=True,
               out_path=output_path)
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
    plotGraph(label_store, pred_store)


def main():
    m1()
    m2()
    m3()


main()
