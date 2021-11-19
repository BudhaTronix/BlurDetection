import sys
from HelperFunctions import Test


def main():
    model_selection = 3
    model_path = "../model_weights/RESNET101_bestWeights.pth"
    output_path = "Outputs/ResNet101/"
    try:
        # Arguments passed
        print("\nName of Python script:", sys.argv[0])
        print("\nName of File         :", sys.argv[1])
        path_single_file = sys.argv[1]
        Transform_Images = True
        obj = Test(model_selection=model_selection, model_path=model_path, testFile=path_single_file,
                   out_path=output_path)
        obj.test_singleFile(Transform_Images)
    except:
        print("Error in passing arguments")
        exit()

main()
