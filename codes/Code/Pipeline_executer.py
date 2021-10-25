from Pipeline import BlurDetection
import json


def main():
    # Define Paths
    with open('Config.json') as f:
        data = json.load(f)
        # Configuration
        system_to_run = "StudentPC"  # Select between - StudentPC, FCM, GPU18
        model_selection = 2  # Set 1 for Resnet18, 2 for ResNet50, 3 for ResNet101

        obj1 = BlurDetection(data, system_to_run, model_selection)

        print(" Tensorboard Logging : ", obj1.log)
        print(" Validation Split    : ", obj1.val_split * 100, "%")

        obj1.trainModel()
        obj1.test()


main()
