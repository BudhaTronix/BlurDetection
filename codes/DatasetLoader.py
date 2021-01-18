import torchio as tio
from pathlib import Path
from ImageTransformer import transform_subject

def datasetLoader(inpPath):
    #os.chdir(inpPath)
    inpPath = Path(inpPath)
    subjects = []
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        subject = tio.Subject(
            image = tio.ScalarImage(file_name),
            label = [0],
        )
        subjects.append(subject)

    for file_name in sorted(inpPath.glob("*.nii.gz")):
        subject = tio.Subject(
            image = transform_subject(3,tio.ScalarImage(file_name)),
            label = [1],
        )
        
        subjects.append(subject)

    dataset = tio.SubjectsDataset(subjects)

    return subjects,dataset

