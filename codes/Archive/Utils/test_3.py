import matplotlib.pyplot as plt
import numpy as np

"""
a = [.1, .2, .3, .4, .5, .6, .7, .30, .10, .12, .23]
fig, ax = plt.subplots()
ax.set(title="SSIM distribution",
       xlabel="SSIM",
       ylabel="Count")
ax.hist(a, bins=6)
plt.show()
print()
# a = [0.2, 0.4, 0.6, 0.7, 0.8, 1.0]
print([i for i in a if 0 <= i <= .2])
print([i for i in a if 0.2 < i <= .4])
print([i for i in a if 0.4 < i <= .6])
print([i for i in a if 0.6 < i <= .8])
print([i for i in a if 0.8 < i <= 1.0])

print(len([i for i in a if 0 <= i <= .2]))
print(len([i for i in a if 0.2 < i <= .4]))
print(len([i for i in a if 0.4 < i <= .6]))
print(len([i for i in a if 0.6 < i <= .8]))
print(len([i for i in a if 0.8 < i <= 1.0]))
val_files = []
ssim = 0
batch_size = 255
ctr_1, ctr_2, ctr_3, ctr_4, ctr_5 = 0
if 0 <= ssim <= .2 and ctr_1 <= int(batch_size / 5):
    val_files.append(0)
    ctr_1 += 1
elif 0.2 < ssim <= .4 and ctr_2 <= int(batch_size / 5):
    val_files.append(0)
    ctr_2 += 1
elif 0.4 < ssim <= .6 and ctr_3 <= int(batch_size / 5):
    val_files.append(0)
    ctr_3 += 1
elif 0.6 < ssim <= .8 and ctr_4 <= int(batch_size / 5):
    val_files.append(0)
    ctr_4 += 1
elif 0.8 < ssim <= 1 and ctr_5 <= int(batch_size / 5):
    val_files.append(0)
    ctr_4 += 1
else:
    print("Subject Skipped")

if ctr_1 == ctr_2 == ctr_3 == ctr_4 == ctr_5 == int(batch_size / 5):
    print("Break")


"""
a = b = c = d = 1
print(a, b, c, d)
