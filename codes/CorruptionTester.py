from Display_Subjects import show_slices, show_slices_2
from motion import MotionCorrupter
from pathlib import Path
import torchio as tio
import random
import torch
from piq import ssim, SSIMLoss
import sewar

path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/ixi_root/T1_mini/"
################################################################
moco0 = MotionCorrupter(degrees=0, translation=0, num_transforms=1)
moco1 = MotionCorrupter(degrees=10, translation=10, num_transforms=4)
moco2 = MotionCorrupter(degrees=10, translation=10, num_transforms=4)
moco3 = MotionCorrupter(degrees=10, translation=10, num_transforms=4)
moco4 = MotionCorrupter(degrees=10, translation=10, num_transforms=4)
prob_corrupt = 1
moco_transforms_0 = [tio.Lambda(moco0.perform, p=prob_corrupt), tio.Lambda(moco0.prepare, p=0)]
moco_transforms_1 = [tio.Lambda(moco1.perform, p=prob_corrupt), tio.Lambda(moco1.prepare, p=1)]
moco_transforms_2 = [tio.Lambda(moco2.perform, p=prob_corrupt), tio.Lambda(moco2.prepare, p=1)]
moco_transforms_3 = [tio.Lambda(moco3.perform, p=prob_corrupt), tio.Lambda(moco3.prepare, p=1)]
moco_transforms_4 = [tio.Lambda(moco4.perform, p=prob_corrupt), tio.Lambda(moco4.prepare, p=1)]

moco_transform_0 = tio.Compose(moco_transforms_0)
moco_transform_1 = tio.Compose(moco_transforms_1)
moco_transform_2 = tio.Compose(moco_transforms_2)
moco_transform_3 = tio.Compose(moco_transforms_3)
moco_transform_4 = tio.Compose(moco_transforms_4)

subjects = []
inpPath = Path(path)
for file_name in sorted(inpPath.glob("*.nii.gz")):
    select = random.randint(0, 4)
    if select == 0:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=[0], )
        s1 = moco_transform_0(subject)
    elif select == 1:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=[1])
        s1 = moco_transform_1(subject)
    elif select == 2:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=[2])
        s1 = moco_transform_2(subject)
    elif select == 3:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=[3])
        s1 = moco_transform_3(subject)
    elif select == 4:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=[4])
        s1 = moco_transform_4(subject)

    print(subject['label'])
    show_slices(subject['image'])
    show_slices_2(s1['image'][tio.DATA][1, :, :, :])
    subjects.append(s1)


    #ssim_index: torch.Tensor = ssim(s1, subject[tio.DATA], data_range=1.)
    #print(ssim_index)
    GT = subject['image'][tio.DATA].squeeze().permute(2,0,1).numpy()
    P = s1['image'][tio.DATA][1, :, :, :].permute(2,0,1).numpy()
    print(sewar.full_ref.ergas(GT, P, r=4, ws=8))
    print(sewar.full_ref.mse(GT, P))
    print(sewar.full_ref.msssim(GT, P, weights=[0.0448, 0.2856, 0.3001, 0.2363, 0.1333], ws=11, K1=0.01, K2=0.03, MAX=None))

    break

