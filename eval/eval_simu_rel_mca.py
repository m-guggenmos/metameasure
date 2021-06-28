import os

import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm
import pandas as pd

pd.set_option('display.expand_frame_repr', False)
pd.set_option('display.max_colwidth', 300)
pd.set_option('display.max_rows', 800)

HOME = os.path.expanduser('~')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

noise_model = 'beta'

df = pd.read_pickle(os.path.join(HOME, f'Dropbox/projects/confidence/metameasure/data/simu_rel_mca_{noise_model}.pkl'))
df = df[(df.nsamples <= 1000)].reset_index(drop=True)

compare = ['mratio_pearson', 'logmratio_pearson']
df1, df2 = df.copy().reset_index(level=0), df.copy().reset_index(level=0)
df1['measure_id'], df2['measure_id'] = 0, 1
df1['rho'], df2['rho'] = np.arctanh(np.minimum(0.99, df1[compare[0]])), np.arctanh(np.minimum(0.99, df2[compare[1]]))
df1['nmae'], df2['nmae'] = np.arctanh(np.minimum(0.99, df1[compare[0].replace('pearson', 'nmae')])), np.arctanh(np.minimum(0.99, df2[compare[1].replace('pearson', 'nmae')]))
data = pd.concat([df1, df2]).reset_index(drop=True)

# for col in data.columns:
data -= data.mean()
data /= data.std()

print(sm.MixedLM.from_formula('rho ~ measure_id * performance * nsamples', groups='index', data=data).fit().summary())
print(sm.MixedLM.from_formula('nmae ~ measure_id * performance * nsamples', groups='index', data=data).fit().summary())