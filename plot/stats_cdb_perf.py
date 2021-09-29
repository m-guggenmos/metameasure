import os

import numpy as np
import pandas as pd
import statsmodels.api as sm

HOME = os.path.expanduser('~')

z_orig = pd.read_pickle('../data/mca_test_retest.pkl')
xlim = np.array([0.5, 3.1])

N = 400

z = z_orig[(z_orig.nsamples >= N) & (z_orig.d1 >= xlim[0]) & (z_orig.d1 <= xlim[1])].copy()
z['logmratio_test'] = np.log(np.maximum(0.1, z['mratio_test']))
z['mratio_bounded_test'] = np.minimum(1.6, np.maximum(0, z['mratio_test']))
z['mratio_logistic_test'] = 1 / (1 + np.exp(-(z.mratio_test-z.mratio_test.median())))
z['mratio_exclude_test'] = z['mratio_test']
z.loc[z.mratio_exclude_test < 0, 'mratio_exclude_test'] = np.nan
z.loc[z.mratio_exclude_test > 1.6, 'mratio_exclude_test'] = np.nan

for i, m in enumerate(('mdiff', 'mratio', 'mratio_bounded', 'mratio_exclude', 'mratio_hmeta', 'logmratio')):

    patsy_string = f'd1_retest ~ {m}_test + C(category_id)'
    model = sm.MixedLM.from_formula(patsy_string, groups='study_id', data=z[~z[f'{m}_test'].isna()]).fit()

    print(model.summary())
