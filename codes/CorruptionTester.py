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
TBLOGDIR = "runs/BlurDetection/SSIM"
writer = SummaryWriter(TBLOGDIR)
subjects = []
inpPath = Path(path)
disp = False
for file_name in inpPath.glob("*.nii.gz"):
    for i in range (1,5):
        print("Corruption Level : ", i)
        subject = tio.Subject(image=tio.ScalarImage(file_name))
        text = "Corruption Level : " + str(i)
        for j in range(1,6):
            s = transform_subject_reality(i, subject["image"][tio.DATA].squeeze(1))
            if disp == True and j==1:
                #show_slices_2(s[1, :, :, :], text)
                GT = s[1,:,:,:]
                show_slices_2((GT-GT.min())/(GT.max()-GT.min()), text)
            P = s[1, :, :, :].permute(2, 0, 1).numpy()
            GT = s[0, :, :, :].permute(2, 0, 1).numpy()
            GT = (GT-GT.min())/(GT.max()-GT.min())
            P = (P-P.min())/(P.max()-P.min())
            ssim = sewar.full_ref.ssim(GT, P, ws=11, K1=0.01, K2=0.03, MAX=1.0, fltr_specs=None, mode='valid')
            MSE = sewar.full_ref.mse(GT, P)
            UQI = sewar.full_ref.uqi(GT, P, ws=8)
            print("SSIM - Iteration : ", (j) ," --> ",ssim)
            print("MSE  - Iteration : ", (j) ," --> ", MSE)
            print("UQI  - Iteration : ", (j) ," --> ",UQI)
            writer.add_scalar("SSIM/{}".format(i), scalar_value=ssim[0], global_step=j)
            writer.add_scalar("MSE/{}".format(i), MSE, j)
            writer.add_scalar("UQI/{}".format(i), UQI, j)
            #print("########################################")
            """
            GT_temp = subject['image'][tio.DATA].permute(3,0,1,2).float()
            P_temp = t = s['image'][tio.DATA][0:1, :, :, :].permute(3,0,1,2)
            GT_temp = (GT_temp + 1) / 2  # [-1, 1] => [0, 1]
            P_temp = (P_temp + 1) / 2
            ms_ssim_val = ms_ssim(GT_temp, P_temp, data_range=255, size_average=False)  # return (N,)
            print(ms_ssim_val)
            #ms_ssim_val = ms_ssim(X, Y, data_range=255, size_average=False)  # (N,)
            """
    break
