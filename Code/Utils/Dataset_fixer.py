for file_name in sorted(main_Path.glob("*T1*.nii.gz")):
    output.append(file_name.name.replace(".nii.gz", ""))