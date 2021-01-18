import numpy as np
import torchio as tio
import matplotlib.pyplot as plt
import nibabel as nib

#img_path = 'IXI002-Guys-0828-T1.nii.gz'
#inp_path = "F:/Datasets/ixi_root"

def show_slices_path(inp_path):
    epi_img = nib.load(inp_path)
    epi_img_data = epi_img.get_fdata()
    slice_0 = epi_img_data[128, :, :]
    slice_1 = epi_img_data[:, 128, :]
    slice_2 = epi_img_data[:, :, 75]
    slices = [slice_0, slice_1, slice_2]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Center slices for EPI image")
    plt.show()

def show_slices(subject):
    epi_img_data = subject[tio.DATA]
    epi_img_data = np.squeeze(epi_img_data)
    slice_0 = epi_img_data[128, :, :]
    slice_1 = epi_img_data[:, 128, :]
    slice_2 = epi_img_data[:, :, 75]
    slices = [slice_0, slice_1, slice_2]
    fig, axes = plt.subplots(1, len(slices))
    for i, slice in enumerate(slices):
        axes[i].imshow(slice.T, cmap="gray", origin="lower")
    plt.suptitle("Center slices for EPI image")
    plt.show()