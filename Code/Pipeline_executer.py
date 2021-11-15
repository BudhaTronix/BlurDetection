import json
from Code.src.Pipeline import BlurDetection

# os.environ['HTTP_PROXY'] = 'http://proxy:3128/'
# os.environ['HTTPS_PROXY'] = 'http://proxy:3128/'


def main():
    with open('Config.json') as f:
        data = json.load(f)

        # Configuration - System
        system_to_run = "StudentPC"                                     # Select between - StudentPC, FCM, GPU18, BRAIN
        model_selection = 3                                       # Set 1 for Resnet18, 2 for ResNet50, 3 for ResNet101

        # Configuration - GPU
        enableMultiGPU = False
        deviceIds = [3, 4]
        defaultGPUID = "cuda:0"

        # Configuration - Logging
        Tensorboard = False

        # Configuration - Training and Validation
        epochs = 1000
        batch_size = 64
        validation_split = 0.3

        # Configuration - Testing
        num_class_confusionMatrix = 9
        path_single_file = "/home/budha/Desktop/T2W_TSE.nii.gz"                # Single file for testing
        output_path = "/home/budha/Desktop/output/M1/"                         # Output of model in Testing
        custom_model_path = ""                                                 # Use this path to load a different model
        Transform_Images = True

        # Train Test setting
        Train = False
        Test = True

        obj = BlurDetection(data, system_to_run, model_selection, deviceIds, enableMultiGPU, defaultGPUID,
                            epochs, Tensorboard, batch_size, validation_split,
                            num_class_confusionMatrix, path_single_file, output_path)

        if Train:
            print(" Tensorboard Logging : ", obj.log)
            print(" Validation Split    : ", obj.val_split * 100, "%")
            obj.train()
        if Test:
            # obj.test()
            if path_single_file != "":
                if custom_model_path != "":
                    obj.test_singleFile(Transform_Images, custom_model_path)
                else:
                    obj.test_singleFile(Transform_Images)


main()
