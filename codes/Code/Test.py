import os
import torch

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = "cuda"


def getClass(value):
    if 0 <= value <= .25:
        return 0
    elif 0.25 < value <= .5:
        return 1
    elif 0.5 < value <= .75:
        return 2
    elif 0.75 < value <= 1.0:
        return 3


def getClass_tol(value_model, value_label, tol=0.05):
    if value_label * (1 - tol) <= value_model <= value_label * (1 + tol):
        value_model = value_label
    return getClass(value_model), getClass(value_label)


def testModel(dataloaders, modelPath, tol=0.5, debug=False):
    print("Model In Testing mode")
    print("Tolerance factor : ", tol*100, "%")
    model = torch.load(modelPath)
    model.eval()
    running_corrects = 0
    # Iterate over data.
    for batch in dataloaders:
        image_batch, labels_batch = batch
        image_batch = image_batch.unsqueeze(1)
        image_batch = (image_batch - image_batch.min()) / \
                      (image_batch.max() - image_batch.min())  # Min Max normalization
        outputs = model(image_batch.float().to(device))[0]
        output_raw = outputs.detach().cpu().squeeze().numpy()
        label_raw = labels_batch.numpy().item()
        if debug:
            print("Raw Output    : Model OP-> ", output_raw, ", Label-> ", label_raw)
            print("Class Output  : Model OP-> ", getClass(output_raw), ", Label-> ", getClass(label_raw), "\n")
        output, label = getClass_tol(output_raw, label_raw, tol=tol)
        if output == label:
            running_corrects += 1

    total_acc = running_corrects
    print("Total Correct :", total_acc, ", out of :", len(dataloaders))
    print("Accuracy      :", float(total_acc / len(dataloaders) * 100), "%")