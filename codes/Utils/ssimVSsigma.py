import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import iqr

data = pd.read_csv("data.csv")
# Preview the first 5 lines of the loaded data
fig, ax = plt.subplots()
ax.set(title="SSIM vs Sigma value graph",
       xlabel="Sigma",
       ylabel="SSIM")
ax.scatter(data["Sigma"][:100], data["SSIM"][:100])
plt.show()

fig, ax = plt.subplots()
ax.set(title="SSIM distribution",
       xlabel="SSIM",
       ylabel="Count")
ax.hist(data["SSIM"], bins=1000)
plt.show()


print(iqr(data["SSIM"], axis=0))
"""
for x in ['SSIM']:
       print(x)
       q75, q25 = np.percentile(data.loc[:, x], [75, 25])
       intr_qr = q75 - q25

       max = q75 + (1.5 * intr_qr)
       min = q25 - (1.5 * intr_qr)

       data.loc[data[x] < min, x] = "" #np.nan
       data.loc[data[x] > max, x] = "" #np.nan

"""
