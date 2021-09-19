from Display_Subjects import show_slices, show_slices_2
from ImageTransformer import transform_subject_reality,transform_subject
from motion import MotionCorrupter
from pathlib import Path
import torchio as tio
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import sewar
import numpy as np
from torch.utils.tensorboard import SummaryWriter
#path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1_mini/"
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsoMini/"
################################################################
subjects = []
inpPath = Path(path)
disp = False
number_of_times = 6
for file_name in inpPath.glob("*.nii.gz"):
    print("######################  REGRESSION  ######################\n")
    for i in range (1,number_of_times):
        print("Iteration : ", i)
        TBLOGDIR = "runs/BlurDetection/CorruptionTesting_Regression/Iteration_{}".format(i)
        writer = SummaryWriter(TBLOGDIR)
        subject = tio.Subject(image=tio.ScalarImage(file_name))

        for j in range(0,21):
            sigma = j*0.01
            moco = MotionCorrupter(mode=2,n_threads=48, mu=0.0, sigma=sigma, random_sigma=False)
            transforms = [tio.Lambda(moco.perform, p=1)]
            transform = tio.Compose(transforms)
            s = transform(subject["image"][tio.DATA])
            if disp == True and i==1:
                text = "Iteration: " + str(i) + "; Corruption Level: " + str(j)
                GT = s[1,:,:,:]
                show_slices_2((GT-GT.min())/(GT.max()-GT.min()), text)
            P = s[1:2, :, :, :].squeeze(0).permute(2, 0, 1).numpy()
            GT = s[0:1, :, :, :].squeeze(0).permute(2, 0, 1).numpy()
            GT = (GT-GT.min())/(GT.max()-GT.min())
            P = (P-P.min())/(P.max()-P.min())
            ssim = sewar.full_ref.ssim(GT, P, ws=11, K1=0.01, K2=0.03, MAX=1.0, fltr_specs=None, mode='valid')
            MSE = sewar.full_ref.mse(GT, P)
            UQI = sewar.full_ref.uqi(GT, P, ws=8)
            print("\nCorruption Level: ", j)
            print("SSIM - SIGMA : ", (sigma) ," --> ",ssim)
            print("MSE  - SIGMA : ", (sigma) ," --> ", MSE)
            print("UQI  - SIGMA : ", (sigma) ," --> ",UQI)
            writer.add_scalar("SSIM", ssim[0], j)
            writer.add_scalar("MSE", MSE, j)
            writer.add_scalar("UQI", UQI, j)

        print("\n######################  CLASSIFICATION  ######################\n")

        TBLOGDIR = "runs/BlurDetection/CorruptionTesting_Classification/Iteration_{}".format(i)
        writer = SummaryWriter(TBLOGDIR)
        #subject = tio.Subject(image=tio.ScalarImage(file_name))

        for j in range(0,5):
            s = transform_subject_reality(j, subject["image"][tio.DATA].squeeze(1))
            if disp == True and i==1:
                text = "Iteration: " + str(i) + "; Corruption Level: " + str(j)
                GT = s[1,:,:,:]
                show_slices_2((GT-GT.min())/(GT.max()-GT.min()), text)
            P = s[1:2, :, :, :].squeeze(0).permute(2, 0, 1).numpy()
            GT = s[0:1, :, :, :].squeeze(0).permute(2, 0, 1).numpy()
            GT = (GT-GT.min())/(GT.max()-GT.min())
            P = (P-P.min())/(P.max()-P.min())
            ssim = sewar.full_ref.ssim(GT, P, ws=11, K1=0.01, K2=0.03, MAX=1.0, fltr_specs=None, mode='valid')
            MSE = sewar.full_ref.mse(GT, P)
            UQI = sewar.full_ref.uqi(GT, P, ws=8)
            sigma = 0
            if(j==1):sigma=1
            elif(j==2):sigma=5
            elif (j == 3):sigma = 10
            elif (j == 4):sigma = 20
            print("\nCorruption Level: ", j)
            print("SSIM - SIGMA : ", (sigma) ," --> ",ssim)
            print("MSE  - SIGMA : ", (sigma) ," --> ", MSE)
            print("UQI  - SIGMA : ", (sigma) ," --> ",UQI)
            writer.add_scalar("SSIM", ssim[0], sigma)
            writer.add_scalar("MSE", MSE, sigma)
            writer.add_scalar("UQI", UQI, sigma)
    break