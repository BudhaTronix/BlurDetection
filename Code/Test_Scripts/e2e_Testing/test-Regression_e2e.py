from HelperFunctions import Test

path = "/media/hdd_storage/Budha/Dataset/Otherdataset/Isotropic/"

model_selection = 1
model_path = "/home/budha/PycharmProjects/BlurDetection/Code/Test_Scripts/model_weights/RESNET18.pth"
output_path = "Outputs/ResNet18/"
obj = Test(model_selection=model_selection, model_path=model_path,out_path=output_path, file_path=path)
obj.testModel()

model_selection = 2
model_path = "/home/budha/PycharmProjects/BlurDetection/Code/Test_Scripts/model_weights/RESNET50.pth"
output_path = "Outputs/ResNet50/"
obj = Test(model_selection=model_selection, model_path=model_path,out_path=output_path, file_path=path)
obj.testModel()

model_selection = 3
model_path = "/home/budha/PycharmProjects/BlurDetection/Code/Test_Scripts/model_weights/RESNET101.pth"
output_path = "Outputs/ResNet101/"
obj = Test(model_selection=model_selection, model_path=model_path,out_path=output_path, file_path=path)
obj.testModel()
