# 研究方差（MSD)，标准差/均方差(RMSD)，以及均方误差MSE
import matplotlib.pyplot as plt
import numpy as np

a = np.linspace(0, 100, 100).reshape(-1, 2)
print(a)

rms = np.std(a)
print("RMS of all is", rms)

rms = np.std(a, axis=0)
print("RMS of column are", rms)

# rms = np.std(a, axis=1)
# print("RMS of row are",rms)


a = np.random.randn(1000)
print(a)
plt.hist(a * 100, bins=100, range=(-300, 300))
plt.show()

b = np.random.random(1000)
print(b)
plt.hist(b * 100, bins=100, range=(0, 100))
plt.show()

rms = np.std(a)
print("rms= ", rms)
