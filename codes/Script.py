import ModelTrainer_SingleGPU_ResNet

temp = ModelTrainer_SingleGPU_ResNet

model_name = "resnet"
num_classes = 1
batch_size = 4
num_epochs = 10
corruptionDegree = 10
device = "cuda:4"

functionCall = temp.BlurDetection(model_name=model_name, num_classes=num_classes, batch_size=batch_size,
                                  num_epochs=num_epochs, corruptionDegree=corruptionDegree,
                                  device=device)
functionCall.callFunction()

"""
functionCall = temp.BlurDetection(model_name=model_name, num_classes=num_classes, batch_size=batch_size,
                                  num_epochs=num_epochs, corruptionDegree=2*corruptionDegree,
                                  device=device)
functionCall.callFunction()



functionCall = temp.BlurDetection(model_name=model_name, num_classes=num_classes, batch_size=batch_size,
                                  num_epochs=num_epochs, corruptionDegree=3*corruptionDegree,
                                  device=device)
functionCall.callFunction()

"""
