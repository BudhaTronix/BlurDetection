import Test

def main():
    with open('Config.json') as f:
        model_selection = 3  # Set 1 for Resnet18, 2 for ResNet50, 3 for ResNet101

        # Configuration - Testing
        num_class_confusionMatrix = 9
        model_path = ""
        path_single_file = "/home/budha/Desktop/T2W_TSE.nii.gz"  # Single file for testing
        output_path = "/home/budha/Desktop/output/M1/"  # Output of model in Testing
        custom_model_path = ""  # Use this path to load a different model
        Transform_Images = True

        obj = Test()
        if path_single_file != "":
            if custom_model_path != "":
                obj.test_singleFile(Transform_Images, custom_model_path)
            else:
                obj.test_singleFile(Transform_Images)

main()