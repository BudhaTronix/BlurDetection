import platform
from typing import Optional

from warnings import warn
from pathlib import Path
import torch
from torch.utils.data import DataLoader
import random
import torchio as tio
from tqdm import tqdm
from pytorch_lightning import LightningDataModule
import numpy as np

from codes.Utils import pytorch_ssim


class BDDataModule(LightningDataModule):
    """Standard MNIST, train, val, test splits and transforms.
    >>> BDDataModule()  # doctest: +ELLIPSIS
    <...mnist_datamodule.MNISTDataModule object at ...>
    """

    name = "BlurDetection"

    def __init__(
            self,
            data_dir: str = "",
            val_split: int = 5000,
            num_workers: int = 16,
            normalize: bool = False,
            seed: int = 42,
            batch_size: int = 32,
            *args,
            **kwargs,
    ):
        """
        Args:
            data_dir: where to save/load the data
            val_split: how many of the training images to use for the validation split
            num_workers: how many workers to use for loading data
            normalize: If true applies image normalize
            seed: starting seed for RNG.
            batch_size: desired batch size.
        """
        super().__init__(*args, **kwargs)
        if num_workers and platform.system() == "Windows":
            # see: https://stackoverflow.com/a/59680818
            warn(
                f"You have requested num_workers={num_workers} on Windows,"
                " but currently recommended is 0, so we set it for you"
            )
            num_workers = 0

        self.dims = (1, 28, 28)
        self.data_dir = data_dir
        self.val_split = val_split
        self.num_workers = num_workers
        self.normalize = normalize
        self.seed = seed
        self.batch_size = batch_size
        self.dataset_train = ...
        self.dataset_val = ...

    @property
    def num_classes(self):
        return 6

    def prepare_data(self):
        """Saves MNIST files to `data_dir`"""
        # MNIST(self.data_dir, train=True, download=True)
        # MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):
        """Split the train and valid dataset."""
        main_Path = str("/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/")
        inp_Path = str("/project/mukhopad/tmp/BlurDetection_tmp/Dataset/SSIM/")
        val_split = 0.5
        mem_batch = 0.01
        batch_size = 128
        inpPath = Path(inp_Path)
        main_Path = Path(main_Path)
        output = []
        random_seed = 42
        random.seed(random_seed)

        for file_name in sorted(inpPath.glob("*.nii.gz")):
            temp = str(file_name.name)
            ssim = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
            fileName = temp.replace(ssim, "")
            if fileName not in output:
                output.append(fileName)

        val_subjects = output[0:int(len(output) * val_split)]
        train_subjects = output[int(len(output) * val_split):len(output)]

        train_files = []
        t_len = 0
        random.shuffle(train_subjects)
        train_batch = train_subjects[0:int(len(train_subjects) * mem_batch)]
        for subject_id in train_batch:
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                t = tio.ScalarImage(file_name)
                t_len = t_len + len(t.data[0]) + len(t.data[0][0]) + len(t.data[0][0][0])
                break
        t_len = t_len * 10
        print("Training batch files - ", t_len)
        flag = ctr_1 = ctr_2 = ctr_3 = ctr_4 = ctr_5 = 0
        l = w = []
        for subject_id in tqdm(train_subjects):
            if flag == 1:
                break
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                imgReg = tio.ScalarImage(file_name)[tio.DATA]
                imgOrig = tio.ScalarImage(main_Path / str(subject_id + ".nii.gz"))[tio.DATA]
                for j in range(3):
                    if j == 0:
                        imgReg_op = imgReg
                        imgOrig_op = imgOrig
                    if j == 1:
                        imgReg_op = imgReg.permute(0, 2, 3, 1)
                        imgOrig_op = imgOrig.permute(0, 2, 3, 1)
                    if j == 2:
                        imgReg_op = imgReg.permute(0, 3, 2, 1)
                        imgOrig_op = imgOrig.permute(0, 3, 2, 1)
                    if torch.cuda.is_available():
                        imgReg_ssim = imgReg_op.cuda()
                        imgOrig_ssim = imgOrig_op.cuda()
                    ssim = pytorch_ssim.ssim(imgOrig_ssim.double(), imgReg_ssim.double()).mean(0).mean(1).mean(
                        1).detach().cpu()
                    c = 0
                    for i in range(0, len(imgReg_op.squeeze())):
                        subject = imgReg_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)
                        if 0 <= ssim[i] <= .2 and ctr_1 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_1 += 1
                        elif 0.2 < ssim[i] <= .4 and ctr_2 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_2 += 1
                        elif 0.4 < ssim[i] <= .6 and ctr_3 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_3 += 1
                        elif 0.6 < ssim[i] <= .8 and ctr_4 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_4 += 1
                        elif 0.8 < ssim[i] <= 1 and ctr_5 < int(t_len / 5):
                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_5 += 1
                        if (ctr_5 < int(t_len / 5) and c < 3):
                            slice = random.randint(0, len(imgReg_op.squeeze()))
                            ssim[0] = 1.0
                            subject = imgOrig_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)

                            train_files.append([subject, ssim[i]])
                            l.append(len(subject))
                            w.append(len(subject[0]))
                            ctr_5 += 1
                            c += 1
                        if ctr_1 == ctr_2 == ctr_3 == ctr_4 == ctr_5 == int(t_len / 5):
                            flag = 1
                            break
                flag = 1

        del imgReg_ssim
        del imgOrig_ssim
        torch.cuda.empty_cache()
        print("Total files in Training: ", len(train_files))
        transform = tio.CropOrPad((1, int(np.median(np.sort(np.array(l)))), int(np.median(np.sort(np.array(w))))))
        for i in range(len(train_files)):
            train_files[i] = [transform(train_files[i].__getitem__(0).unsqueeze(0).unsqueeze(0)).double().squeeze(0),
                              train_files[i].__getitem__(1)]
        #################

        random.shuffle(val_subjects)
        val_batch = val_subjects[0:int(len(val_subjects) * mem_batch)]
        v_len = 0
        for subject_id in val_batch:
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                t = tio.ScalarImage(file_name)
                v_len = v_len + len(t.data[0]) + len(t.data[0][0]) + len(t.data[0][0][0])
                break
        print("Validation batch files - ", v_len * 10)
        v_len = v_len * 10
        val_files = []
        flag = ctr_1 = ctr_2 = ctr_3 = ctr_4 = ctr_5 = 0
        for subject_id in tqdm(val_subjects):
            if flag == 1:
                break
            for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
                imgReg = tio.ScalarImage(file_name)[tio.DATA]
                imgOrig = tio.ScalarImage(main_Path / str(subject_id + ".nii.gz"))[tio.DATA]
                for j in range(3):
                    if j == 0:
                        imgReg_op = imgReg
                        imgOrig_op = imgOrig
                    if j == 1:
                        imgReg_op = imgReg.permute(0, 2, 3, 1)
                        imgOrig_op = imgOrig.permute(0, 2, 3, 1)
                    if j == 2:
                        imgReg_op = imgReg.permute(0, 3, 2, 1)
                        imgOrig_op = imgOrig.permute(0, 3, 2, 1)
                    if torch.cuda.is_available():
                        imgReg_ssim = imgReg_op.cuda()
                        imgOrig_ssim = imgOrig_op.cuda()
                    ssim = pytorch_ssim.ssim(imgOrig_ssim.double(), imgReg_ssim.double()).mean(0).mean(1).mean(
                        1).detach().cpu()

                    for i in range(0, len(imgReg_op.squeeze())):
                        subject = imgReg_op[:, i:(i + 1), :, :].squeeze(0).squeeze(0)
                        if 0 <= ssim[i] <= .2 and ctr_1 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_1 += 1
                        elif 0.2 < ssim[i] <= .4 and ctr_2 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_2 += 1
                        elif 0.4 < ssim[i] <= .6 and ctr_3 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_3 += 1
                        elif 0.6 < ssim[i] <= .8 and ctr_4 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_4 += 1
                        elif 0.8 < ssim[i] <= 1 and ctr_5 < int(v_len / 5):
                            val_files.append([subject, ssim[i]])
                            ctr_5 += 1
                        if ctr_5 < int(v_len / 5) and c < 3:
                            slice = random.randint(0, len(imgReg_op.squeeze()))
                            ssim[0] = 1.0
                            subject = imgOrig_op[:, slice:(slice + 1), :, :].squeeze(0).squeeze(0)
                            val_files.append([subject, ssim[i]])
                            ctr_5 += 1
                            c += 1
                        if ctr_1 == ctr_2 == ctr_3 == ctr_4 == ctr_5 == int(v_len / 5):
                            flag = 1
                            break
                flag = 1
        for i in range(len(val_files)):
            val_files[i] = [transform(val_files[i].__getitem__(0).unsqueeze(0).unsqueeze(0)).double().squeeze(0),
                            val_files[i].__getitem__(1)]
        del imgReg_ssim
        del imgOrig_ssim
        torch.cuda.empty_cache()

        self.dataset_train = train_files
        self.dataset_val = val_files

        torch.cuda.empty_cache()

    def train_dataloader(self):
        """MNIST train set removes a subset to use for validation."""
        loader = DataLoader(
            self.dataset_train,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def val_dataloader(self):
        """MNIST val set uses a subset of the training set for validation."""
        loader = DataLoader(
            self.dataset_val,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader

    def test_dataloader(self):
        """MNIST test set uses the test split."""
        extra = dict(transform=self.test_transforms) if self.test_transforms else {}
        dataset = MNIST(self.data_dir, train=False, download=False, **extra)
        loader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
            pin_memory=True,
        )
        return loader
