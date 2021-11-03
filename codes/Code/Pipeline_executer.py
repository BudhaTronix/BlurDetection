from Pipeline import BlurDetection
import json
import os

os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


def main():
    # Define Paths
    with open('Config.json') as f:
        data = json.load(f)
        # Configuration
        system_to_run = "FCM"  # Select between - StudentPC, FCM, GPU18
        model_selection = 2  # Set 1 for Resnet18, 2 for ResNet50, 3 for ResNet101
        deviceIds = [3,4]

        obj1 = BlurDetection(data, system_to_run, model_selection, deviceIds)

        print(" Tensorboard Logging : ", obj1.log)
        print(" Validation Split    : ", obj1.val_split * 100, "%")

        obj1.trainModel()
        # obj1.test()


main()
