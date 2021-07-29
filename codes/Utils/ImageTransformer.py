import torchio as tio
from motion import MotionCorrupter
import torch

def transform_subject(trns_type,subject):
    normalization_mode = 1
    moco0 = MotionCorrupter(degrees=0, translation=0, num_transforms=1,norm_mode=normalization_mode)
    moco1 = MotionCorrupter(degrees=10, translation=10, num_transforms=3,norm_mode=normalization_mode)
    moco2 = MotionCorrupter(degrees=10, translation=10, num_transforms=5,norm_mode=normalization_mode)
    moco3 = MotionCorrupter(degrees=10, translation=10, num_transforms=7,norm_mode=normalization_mode)
    moco4 = MotionCorrupter(degrees=10, translation=10, num_transforms=10,norm_mode=normalization_mode)
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

    if trns_type == 0:
        #transformed_subject = moco_transform_0(subject)
        transformed_subject = torch.cat([subject,subject],0)
    elif trns_type == 1:
        transformed_subject = moco_transform_1(subject)
    elif trns_type == 2:
        transformed_subject = moco_transform_2(subject)
    elif trns_type == 3:
        transformed_subject = moco_transform_3(subject)
    elif trns_type == 4:
        transformed_subject = moco_transform_4(subject)

    return transformed_subject

def transform_subject_reality(trns_type,subject):
    n_threads = 48
    mu = 0.0  ## 0.0 - 1.0

    normalization_mode = 0
    moco1 = MotionCorrupter(mode=2,n_threads=n_threads, mu=mu, sigma=0.01, random_sigma=False)
    moco2 = MotionCorrupter(mode=2,n_threads=n_threads, mu=mu, sigma=0.05, random_sigma=False)
    moco3 = MotionCorrupter(mode=2,n_threads=n_threads, mu=mu, sigma=0.1, random_sigma=False)
    moco4 = MotionCorrupter(mode=2,n_threads=n_threads, mu=mu, sigma=0.2, random_sigma=False)

    """
    prob_corrupt = 1
    #moco_transforms_0 = [tio.Lambda(moco0.perform, p=prob_corrupt), tio.Lambda(moco0.prepare, p=0)]
    moco_transforms_1 = [tio.Lambda(moco1.perform, p=prob_corrupt), tio.Lambda(moco1.prepare, p=1)]
    moco_transforms_2 = [tio.Lambda(moco2.perform, p=prob_corrupt), tio.Lambda(moco2.prepare, p=1)]
    moco_transforms_3 = [tio.Lambda(moco3.perform, p=prob_corrupt), tio.Lambda(moco3.prepare, p=1)]
    moco_transforms_4 = [tio.Lambda(moco4.perform, p=prob_corrupt), tio.Lambda(moco4.prepare, p=1)]
    """

    moco_transforms_1 = [tio.Lambda(moco1.perform, p = 1)]
    moco_transforms_2 = [tio.Lambda(moco2.perform, p = 1)]
    moco_transforms_3 = [tio.Lambda(moco3.perform, p = 1)]
    moco_transforms_4 = [tio.Lambda(moco4.perform, p = 1)]


    moco_transform_1 = tio.Compose(moco_transforms_1)
    moco_transform_2 = tio.Compose(moco_transforms_2)
    moco_transform_3 = tio.Compose(moco_transforms_3)
    moco_transform_4 = tio.Compose(moco_transforms_4)

    if trns_type == 0:
        transformed_subject = torch.cat((subject,subject),0)
    if trns_type == 1:
        transformed_subject = moco_transform_1(subject)
    elif trns_type == 2:
        transformed_subject = moco_transform_2(subject)
    elif trns_type == 3:
        transformed_subject = moco_transform_3(subject)
    elif trns_type == 4:
        transformed_subject = moco_transform_4(subject)

    return transformed_subject


def transform_file_reality(trns_type,file_name):
    n_threads = 48
    mu = 0.0  ## 0.0 - 1.0
    #sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))  ## 0.05 - 0.1 np.random.uniform(low=0.01, high=0.2, size=(50,))

    normalization_mode = 1
    moco0 = MotionCorrupter(n_threads=n_threads, mu=mu, sigma=0.01, random_sigma=False,norm_mode=normalization_mode)
    moco1 = MotionCorrupter(n_threads=n_threads, mu=mu, sigma=0.01, random_sigma=False,norm_mode=normalization_mode)
    moco2 = MotionCorrupter(n_threads=n_threads, mu=mu, sigma=0.05, random_sigma=False,norm_mode=normalization_mode)
    moco3 = MotionCorrupter(n_threads=n_threads, mu=mu, sigma=0.1, random_sigma=False,norm_mode=normalization_mode)
    moco4 = MotionCorrupter(n_threads=n_threads, mu=mu, sigma=0.2, random_sigma=False,norm_mode=normalization_mode)
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

    if trns_type == 0:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=0)
        transformed_subject = moco_transform_0(subject)
    elif trns_type == 1:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=1)
        transformed_subject = moco_transform_1(subject)
    elif trns_type == 2:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=2)
        transformed_subject = moco_transform_2(subject)
    elif trns_type == 3:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=3)
        transformed_subject = moco_transform_3(subject)
    elif trns_type == 4:
        subject = tio.Subject(image=tio.ScalarImage(file_name), label=4)
        transformed_subject = moco_transform_4(subject)

    return transformed_subject
