from scipy.stats import norm
import numpy as np
import matplotlib.pyplot as plt

N = 10000

pcorrect = np.arange(0.1, 1.0001, 0.0001)
# voi = np.array([-2, 0, 1, 2, 3, 4, 5, 6])
voi = np.array([0.1, 0.6, 1.1, 1.6, 2.1, 2.6, 3.1])

d1 = np.full(len(pcorrect), np.nan)
for i, p in enumerate(pcorrect):
    d1[i] = norm.ppf(p) - norm.ppf(1 - p)

for v in voi:
    idx = (np.abs(d1[~np.isnan(d1)] - v)).argmin()
    print(f"d'={v:.1f}: p={pcorrect[~np.isnan(d1)][idx]:.2f}")

plt.figure()
plt.plot(pcorrect, d1)
plt.xlim(0.1, 1)
plt.xlabel('Proportion correct')
plt.ylabel("d'")
