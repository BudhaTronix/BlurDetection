import torchio as tio
from pathlib import Path
def func():
    #path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/Iso_Transformed_Regression_T1/"
    path = "/media/hdd_storage/Budha/Dataset"
    inpPath = Path(path)
    output = []
    for file_name in sorted(inpPath.glob("*.nii.gz")):
        temp = str(file_name.name)
        sigma = str("-" + file_name.name.split(".nii.gz")[0].split("-")[-1] + ".nii.gz")
        fileName = temp.replace(sigma, "")
        if fileName not in output:
            output.append(fileName)
    print(len(output))
    new_subs = []
    for subject_id in output:
        count = 0
        for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
            count += 1
        if(count < 5):
            print(subject_id)
            new_subs.append(subject_id)
    print(len(new_subs))
    """
    in_path = "/project/mukhopad/tmp/BlurDetection_tmp/Dataset/IsotropicDataset/"
    inpPath = Path(in_path)
    subjects = []
    for subject_id in new_subs:
        for file_name in sorted(inpPath.glob(str("*" + subject_id + "*"))):
            print(file_name)
            subject = tio.Subject(
                image=tio.ScalarImage(file_name),
                filename=str(file_name.name),
            )
            subjects.append(subject)
    print(len(subjects))
    return(subjects)
    
    """