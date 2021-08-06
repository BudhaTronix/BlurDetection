from Display_Subjects import show_slices, show_slices_2
from ImageTransformer import transform_subject_reality, transform_subject
from motion import MotionCorrupter
from pathlib import Path
import torchio as tio
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import sewar
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1_mini/"
path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
################################################################
log = False
subjects = []
inpPath = Path(path)
disp = False
number_of_times = 1
calc = True
for file_name in inpPath.glob("*T1*.nii.gz"):
    for i in range(0, number_of_times):
        print("Iteration : ", i, " Number of Ghosts: ", 10)
        if log:
            TBLOGDIR = "runs/BlurDetection/CorruptionTestinng_TorchIO_3/Iteration_4{}".format(i)
            writer = SummaryWriter(TBLOGDIR)
        subject = tio.Subject(image=tio.ScalarImage(file_name))

        for j in range(0, 6):
            moco = tio.transforms.Ghosting(num_ghosts=10, intensity=j*0.2, axis=0, restore=0)
            # moco = tio.transforms.Motion(degrees=(10.0,2.0,3.0),translation=(10.0,2.0,3.0),times=[1],image_interpolation='linear')
            # transforms = [tio.transforms.Ghosting(num_ghosts=10, intensity=0.4, axis=0, restore=0),
            #              tio.transforms.RandomMotion(p=1,degrees=10,translation=1.0,image_interpolation='linear',num_transforms=j*2)]
            # moco = tio.Compose(transforms)
            s = moco(subject)

            if disp == True and i == 0:
                text = "Itr: " + str(i) + "; Degrees: " + str(10) + "; num_transforms: " + str(j)
                P = s["image"][tio.DATA]
                show_slices_2(P, text)
            if calc:
                P = s["image"][tio.DATA].squeeze(0).permute(2, 0, 1).numpy()
                GT = subject["image"][tio.DATA].squeeze(0).permute(2, 0, 1).numpy()

                GT = (GT - GT.min()) / (GT.max() - GT.min())
                P = (P - P.min()) / (P.max() - P.min())
                ssim = sewar.full_ref.ssim(GT, P, ws=11, K1=0.01, K2=0.03, MAX=1.0, fltr_specs=None, mode='valid')
                MSE = sewar.full_ref.mse(GT, P)
                UQI = sewar.full_ref.uqi(GT, P, ws=8)

                print("\nCorruption Level: ", j)
                print("SSIM - SIGMA : ", (j), " --> ", ssim)
                print("MSE  - SIGMA : ", (j), " --> ", MSE)
                print("UQI  - SIGMA : ", (j), " --> ", UQI)
                if log:
                    writer.add_scalar("SSIM", ssim[0], j)
                    writer.add_scalar("MSE", MSE, j)
                    writer.add_scalar("UQI", UQI, j)
        # break
    break
