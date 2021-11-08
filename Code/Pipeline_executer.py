import json
from Code.src.Pipeline import BlurDetection

# os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
# os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


def main():
    with open('Config.json') as f:
        data = json.load(f)

        # Configuration - System
        system_to_run = "FCM"  # Select between - StudentPC, FCM, GPU18, BRAIN
        model_selection = 3  # Set 1 for Resnet18, 2 for ResNet50, 3 for ResNet101

        # Configuration - GPU
        enableMultiGPU = False
        deviceIds = [3, 4]
        defaultGPUID = "cuda:2"

        # Configuration - Logging
        Tensorboard = False

        # Configuration - Training and Validation
        epochs = 1000
        batch_size = 64
        validation_split = 0.3

        # Configuration - Testing
        num_class_confusionMatrix = 4

        obj = BlurDetection(data, system_to_run, model_selection, deviceIds, enableMultiGPU, defaultGPUID,
                            epochs, Tensorboard, batch_size, validation_split, num_class_confusionMatrix)

        print(" Tensorboard Logging : ", obj.log)
        print(" Validation Split    : ", obj.val_split * 100, "%")

        # obj.train()
        obj.test()


main()
