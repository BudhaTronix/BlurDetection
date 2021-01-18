import torchio as tio
from torchio.transforms import RandomBlur
from torchio.transforms import Blur,RandomNoise
from Display_Subjects import show_slices


def transform_subject(trns_type,subject):

    #print("Before Transformation: ")
    #show_slices(subject)
    if (trns_type == 1):
        affine_transform = tio.RandomAffine()
        transformed_subject = affine_transform(subject)
        #show_slices(transformed_subject)
    if (trns_type == 2):
        spatial_transforms = {tio.RandomElasticDeformation(): 0.2, tio.RandomAffine(): 0.8, }
        transform = tio.Compose([tio.OneOf(spatial_transforms, p=0.5), tio.RescaleIntensity((0, 1)),])
        transformed_subject = transform(subject['image'])
        #show_slices(transformed_subject)
    if (trns_type == 3):
        Blur_transform = RandomBlur(p=1)
        transform = tio.Compose([tio.RandomGhosting((4,10),(0,1,2),(.5,1)),tio.RandomBlur((1,2))])
        transformed_subject = transform(subject)
        #show_slices(transformed_subject)

    return transformed_subject


#img_path = 'IXI002-Guys-0828-T1.nii.gz'
#inp_path = "F:/Datasets/ixi_root/T1"
#subs, dataset = datasetLoader(inp_path)
