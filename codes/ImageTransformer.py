import torchio as tio
from motion import MotionCorrupter

def transform_subject(trns_type,subject):
    moco0 = MotionCorrupter(degrees=0, translation=0, num_transforms=1)
    moco1 = MotionCorrupter(degrees=10, translation=10, num_transforms=2)
    moco2 = MotionCorrupter(degrees=20, translation=10, num_transforms=2)
    moco3 = MotionCorrupter(degrees=30, translation=10, num_transforms=3)
    moco4 = MotionCorrupter(degrees=40, translation=10, num_transforms=4)
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
        transformed_subject = moco_transform_0(subject)
    elif trns_type == 1:
        transformed_subject = moco_transform_1(subject)
    elif trns_type == 2:
        transformed_subject = moco_transform_2(subject)
    elif trns_type == 3:
        transformed_subject = moco_transform_3(subject)
    elif trns_type == 4:
        transformed_subject = moco_transform_4(subject)

    return transformed_subject

def transform_file(trns_type,file_name):
    moco0 = MotionCorrupter(degrees=0, translation=0, num_transforms=1)
    moco1 = MotionCorrupter(degrees=10, translation=10, num_transforms=2)
    moco2 = MotionCorrupter(degrees=20, translation=10, num_transforms=2)
    moco3 = MotionCorrupter(degrees=30, translation=10, num_transforms=3)
    moco4 = MotionCorrupter(degrees=40, translation=10, num_transforms=4)
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
