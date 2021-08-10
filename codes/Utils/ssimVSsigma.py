import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv("data_2.csv")
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
ax.hist(data["SSIM"], bins=100)
plt.show()
