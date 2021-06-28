import numpy as np

niter = 100000

perf = 0.99

n = 2351

acc = np.random.rand(niter, n) < perf
print(np.mean(acc), np.mean(np.abs(np.mean(acc, axis=1) - perf) < 0.02))