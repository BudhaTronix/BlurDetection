import os
import random
from ImageTransformer import transform_file
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1/"

device = "cuda:6"

def train_model(subject):
    return 0


print("Training Started")
for epoch in range(2):
    print("Epoch :",(epoch+1))
    arr = os.listdir(path)
    random.shuffle(arr)
    running_loss = 0.0
    for file_name in arr:

        file_path = os.path.join(path, file_name)
        select = random.randint(0, 4)
        subject = transform_file(select,file_path)
        """
        Train the model here are get the output of each file. 
        If required can be converted to batches
        """

        loss_subject = train_model(subject)

        running_loss = running_loss + (loss_subject)

    print("Loss :", running_loss/len(arr))



