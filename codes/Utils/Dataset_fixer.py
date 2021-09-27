from utils import returnClass
import numpy as np
from pathlib import Path

for file_name in sorted(main_Path.glob("*T1*.nii.gz")):
    output.append(file_name.name.replace(".nii.gz", ""))