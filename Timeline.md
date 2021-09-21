
# Meeting :
###Tasks of previous week - 
1. Check the images saved in tensorboard by (one channel has to be taken while visualizing) - Done
2. Try to print the class in the tensorboard with the images - Done
3. Find the lower and upper bounds - Need input
4. Create the dataset then train the model - Hybrid structure need to ask 
5. Extend the torch utils Data - dataset class use get items - Did not use this but used another method - Need to use this to introduce randomization while training + no. of workers 
6. PIQ package (BRISQUE) - where do we need this? 
7. sewar package - implemented but where to do we need this?

### Changes made -
1. Previously the entire dataset was corrupted and the corruption was based on the corruption probability
2. Currently multiclass has been introduced and the loss function has been changed to CrossEntropy
3. Currently each Image of the T1 dataset is used as a subject and the corruption is made in the subject. Range [0-4]. 0 -means no corruption

### Questions asked - 
1. Can we have a file based approach?(Explain) - NO
2. What is the requirement of the PIQ package? 
3. What is the requirement of sewar package? - To be used for motion corruption analysis
4. Is there any need to have any speed improvements when it comes to training the model? - No 

### Current Stats -
1. Creating dataset(transforming each subject) of 581 subjects - 35mins 10 secs
2. Training 523 subjects takes 40mins 44secs x 3 every epoch
3. Validation of 58 subjects - 55 secs


### Tasks for next meet- 
1. Train the model with two more sides of data - Coronal and Axial  - Done
2. Use the Isotropic data - Done
3. Dataset using the get items and training the model - Done
4. SSIM vs NT - Single subject - 5 times 
5. Prepare a document - with different parameters - images 
Getting from Alex - Isotropic Images

Next meeting - 14:00 hours

###########################################################################################
# Meeting : 
### Tasks done - 
1. Introduce batch while training
2. Check with number_of_workers 
3. Create dataset and train the mode - store class id with image name
4. Use reality motion
5. Tested the motion corruptor with changing of Sigma value

### Questions - 
1. Issue with the IXI dataset - Issue with viewing. Image is Okay
2. Effect of batch size on training - 
3. Is cropping the images a good idea? Should i crop from both the sides? - Use Patches 
4. Normailization of two types and its significance. - Use Gaussian Normalization
5. Check with Adam optimizer
6. Put value of mu as 0

### Issues faced - 
1. Cropping for batch creation 
2. Use of nibabel save option - used torchio functionality

###########################################################################################
Meeting : 
### Things done - 
1. Introduced patching in dataset creation
2. Ran the corruption tester - 
	a. Fixed sigma values [0.01, 0.05, 0.1, 0.2] x5
	b. Increasing sigma values [0.01 - 2.0]
3. Ran regression model - 100 epochs
4. Ran classification model - 100 epochs

### Need input - 
1. How to do Gaussian Normailization 
2. What should be the patch size - currently it is the minimum size(of an image) in the entire dataset

### Creates 
1. Irregularrity seen in the corruption tester
2. Regression - Make another dataset 5 different sigma values 
###########################################################################################
# Meeting : 18/06/2021
Tasks completed - 
1. Creating the regression dataset with 5 values of sigma(random) per subject - code still running
2. Check the issue with corruption tester(Tensorboard) - issue fixed
3. Applying gaussian normalization during training - done
4. Train the regression model with entire dataset - done on partial dataset
5. Train the classification model with entire dataset - done on partial dataset


### Questions/Issues -
1. Sigma 0 still produces motion in the image. Should I manually change that? - No issue
2. There is a anomaly seen for sigma value - 0.18 
3. Issue faced while doing torch.sum(preds == labels)
4. Need to run the dataset creator code in other server

### Stats -
1. Classification model - best val accuracy - 24.67, epochs - 50, No of subjects - 762
2. Regression model - best val accuracy - 0, epochs - 20, No of subjects - 618

### Tasks -
1. Share the 3 images(share in telegram) 
2. Train/Val split - 50/50
3. Add test set and plot
4. Focus on the regression
5. Harmonization code incorporation 
6. Text to Soumick regarding dataset creation
7. Corrupt T1, T2, PD
8. Start working on T1 only

### Short update over Telegram -

Number of isotropic T1 subjects used for corruption - 581
Time taken for corrupting 581 subjects(per round)- 22 hour 48 mins
Current status - 83% completed of 2nd time corrupting the dataset 

I am working on efficiently splitting corrupted patient data based on patient ID and training. 
############################################################################################
# Meeting: 30/06/21
### Work done till now -- 
1. Corrupting subject 5 times with regression 
2. Splitting the dataset - training/validation based on filenames 
	*after getting input about using TorchIO's corruption*
3. Checked in corruption tester and found increasing intensity increases corruption level
4. Currently increasing the intensity value at an interval of 0.4
5. Training the model in calssification manner
6. Training datset on the fly - need input on this
7. Created a multithreading approach towards dataset creation

There are 4 parameters to tweek - num_ghosts, intensity, axis, restore
### Questions - 
1. Do we need to increase the num_ghosts(10-40 no effect, 50 had reduced the SSIM)
2. Is traing on every axis required? 
3. Can there be a motion in two axis? 
4. Is only Ghosting requrie or Motion can be added too? 
5. Should I take the same approach of regression? if yes then what parameters to alter?

### TASKS - 
1. Continue with the regression 
2. Complete the dataset of regression and train the model 
3. Early stopping in validation 
4. Create a set for testing 
################################################################################################
Meeting: 09/07/21
Work done till now -- 
1. Training the regression model with 1000 epochs(ResNet18)
2. Training the regression model with 1000 epochs(ResNet101) - Better performance 
3. Tested the TorchIO's motion corruption - seen difference by changing the parameters 
4. Coded for TorchIO's motio corruption - Classification model 
5. Coded for TorchIO's motio corruption - Regression model
6. Test set implementation after every 5 epoch

### Question - 
1. Does Ghosting has to have same values and only Motion corruption values are changed? 
2. Is the TEST set check acceptable method? 
3. How should i train or should i test more in corruption tester? 

### TASKS-
1. Put dropout - stick to ResNet18 as of now then follwed by ResNet101 - Need help
2. Increase dataset by 5 more corruptions - Started
3. [Low Prio] - Testing with other models(Densenet) 
4. Share the code for regression 
5. Reality motion corruption to be used
##################################################################################################
### TASKS -
1. Change the normalization and check
2. AMP - loss part needs to be checked - grad scaler - Done
3. Refer: https://github.com/Po-Hsun-Su/pytorch-ssim/blob/master/pytorch_ssim/__init__.py
4. Need to have different slices while training - Done 
5. Stratifed batches - Not Done

Joblib
Multiprocessing - python 
multiprocessing.dummy 

### Currently what is done -
1. Random slice selection in a batch for different subject during training.
2. Using the grad scaler while backpropagating the loss

### What i am doing -
1. Stratifed batches based on the SSIM values
2. Converting the torchIO subjects from 3D to 2D to store the SSIM value as labels and then train the model - Done
3. Calculation of SSIM strategy - 
	a. Approach 1 - calculate the SSIM value during subject creation and then check the distribution - Done
	b. Approach 2 - calculate the SSIM value during the training and then check the distribution

### TASKS- 
1. 3D to 2D slice conversion
2. Stop using TorchIO's Queue
3. Calculate the SSIM valuee for every slice
4. Distribute the SSIM valuee
5. Shuffling the main dataset everytime the data creation is called

# Meeting: 06/09/21
### TASKS- 
1. Reduce the number of slies when training - Done - added buffer which can be changed as per requirement
2. Reduce the number of classes to 4 - Done
3. Check the direction of the slice vs ssim value - can be checked from the csv created 
4. Remove the extra lines - Done 
5. Check the vloumes for inconsistent data - Done 
6. Custom dataset classes(check from Covid code) - Done
7. Implement using Pytorch lightning - Done (Need to be verfied)
8. Check the entire normaization applied on the dataset should be min max on everything - Done
9. Read the volume normalize then save in pickle - Not needed

### Currently what is done -
1. Splitting code into train-validate, test, dataloader
2. Custom dataloader
3. new dataset creator code


# Meeting: 21/09/21

### TASKS :
1. Read about evaluation of regression data - different metrics
2. Confusion matrix
3. Check about axis
4. put back the subject selection during train/val
5. Update about subject selection - train/val/test
6. Write a script: input nifty files -> output of the model
7. Add sample snashots in the tensorboard

### Plans:
1. Run code with with Densnet model
2. Change the optimizer and check use adabound - 
   https://pytorch-optimizer.readthedocs.io/en/latest/_modules/torch_optimizer/adabound.html 
   - Not needed
3. Increase the dataset 
4. Finish with lightning - can be done later 
5. Include the T2 images as well in the training and test - Not needed now. Finish with T1 first

