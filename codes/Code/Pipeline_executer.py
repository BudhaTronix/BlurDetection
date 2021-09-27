from Pipeline import BlurDetection


def main():
    dataset_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Dataset/"
    test_dataset_Path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/TestDataset/"
    model_Path = '../../model_weights/RESNET101.pth'
    model_bestweight_Path = '../../model_weights/RESNET101_bestWeights.pth'
    obj1 = BlurDetection(dataset_path, test_dataset_Path, model_Path, model_bestweight_Path)
    print(" Tensorboard Logging : ", obj1.log)
    print(" Validation Split    : ", obj1.val_split * 100, "%")

    # Define Transformation
    transform_val = obj1.getTransformation()

    # Training and Validating Model
    obj1.trainModel(useSaveWeights=False, transform_val=transform_val)

    # Testing Model
    obj1.test(transform_val)

main()