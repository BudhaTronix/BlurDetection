import torchio as tio
import matplotlib.pyplot as plt
from pathlib import Path


ssim = []
s1 = s2 = s3 = s4 = []
c = 0
main_Path = Path("/project/mukhopad/tmp/BlurDetection_tmp/Dataset/SSIM/")
for file_name in sorted(main_Path.glob(str("*.nii.gz"))):
    imgReg = tio.ScalarImage(file_name)[tio.DATA]
    s1.append(len(imgReg))
    s2.append(len(imgReg[0]))
    s3.append(len(imgReg[0][0]))
    s4.append(len(imgReg[0][0][0]))
    ssim.append(float(str(file_name).split(".nii.gz")[0].split("-")[-1]))
    c += 1
    #if c == 100: break

fig, ax = plt.subplots()
ax.set(title="SSIM distribution",
       xlabel="SSIM",
       ylabel="Count")
ax.hist(ssim, bins=5)
plt.show()
print(c)