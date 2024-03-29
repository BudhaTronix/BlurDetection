import os
import sys
from HelperFunctions import Test


def main():
    model_selection = 1
    model_path = "../model_weights/RESNET18_bestWeights.pth"
    output_path = "Outputs/ResNet18/"
    try:
        # Arguments passed
        print("\nName of Python script:", sys.argv[0])
        print("\nName of File         :", sys.argv[1])
        path_single_file = sys.argv[1]

        if os.path.isfile(path_single_file):
            print("File exists")
        else:
            print("File not accessible")
            exit()
        Transform_Images = True
        obj = Test(model_selection=model_selection, model_path=model_path, testFile=path_single_file,
                   out_path=output_path)
        obj.test_singleFile(Transform_Images)
    except:
        print("Error in passing arguments")
        exit()

main()
