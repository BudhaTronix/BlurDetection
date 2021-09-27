import torchio as tio
from motion import MotionCorrupter
import numpy as np
from pathlib import Path
from tqdm import tqdm


def create_subjectlist(inpPath):
    subjects = []
    inpPath = Path(inpPath)
    print("\n Loading Dataset")
    for file_name in sorted(inpPath.glob("*T1*.nii.gz")):
        subject = tio.Subject(
            image=tio.ScalarImage(file_name),
            filename=str(file_name.name),
        )
        subjects.append(subject)
    return subjects


def datasetCreator_regression(in_path, out_path, no_of_corruption, add_base_image):
    subjects_dataset = create_subjectlist(in_path)
    print("\n Corrupting Dataset....")
    n_threads = 5
    mu = 0.0
    for s in tqdm(subjects_dataset):
        for i in range(0, no_of_corruption):
            name = s["filename"]
            sigma = np.random.uniform(low=0.01, high=0.2, size=(1,))
            moco = MotionCorrupter(mode=2, n_threads=n_threads, mu=mu, sigma=sigma, random_sigma=False)
            transforms = [tio.Lambda(moco.perform, p=1)]
            transform = tio.Compose(transforms)
            img = transform(s["image"][tio.DATA])[1:2, :, :, :]
            temp = tio.ScalarImage(tensor=img)
            temp.save(out_path + str(name.split(".nii.gz")[0]) + "-" + str(sigma[0]) + '.nii.gz', squeeze=True)