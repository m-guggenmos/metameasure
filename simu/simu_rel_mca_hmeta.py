#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Matthias Guggenmos <mg.corresponding@gmail.com>
# 2021

# Simulation: test-retest reliability of Mratio based on hmeta-d'

import os
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import pandas as pd
from scipy.stats import uniform, pearsonr

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.type2SDT_hmeta import type2_SDT_MLE_group  # noqa
from simu_confidence import gen_data  # noqa

# run each iteration separately, as JAGS failed with a multiprocessing scheme
run = 199  # 0 .. 199

# Seed
np.random.seed(run)

# Parameters
mu = 0.5
# sigma_sens_list = [2, 1, 0.64, 0.48, 0.3, 0.24, 0.19, 0.155]
sigma_sens_list = [2, 1, 0.64, 0.48, 0.41, 0.3, 0.24, 0.195, 0.155, 0]
noise_model = 'beta'
sigma_meta_range = [0, 0.5]

# Ns
nsamples_list = np.hstack((np.arange(25, 1001, 25), 2500, 5000, 10000))
nsamples_list_len = len(nsamples_list)
nsubjects = 100
nsigmasens = len(sigma_sens_list)


# Create storage data frame
df = pd.DataFrame(index=range(nsamples_list_len * nsigmasens))
df['sigma_sens_id'] = np.tile(range(nsigmasens), nsamples_list_len)
df['nsamples_id'] = np.repeat(range(nsamples_list_len), nsigmasens)

# generate uniform samples of sigma_meta
sigma_meta_subject = uniform(loc=sigma_meta_range[0], scale=sigma_meta_range[1]).rvs(nsubjects)

print(f'run={run}')

for i, nsamp in enumerate(nsamples_list):
    t0 = default_timer()

    for j, sigma_sens in enumerate(sigma_sens_list):

        # generate test and retest data
        stimulus_test, choices_test, correct_test, confidence_test, confidence_disc_test, nratings_test = \
            gen_data(nsubjects, nsamp, sigmas_sens=sigma_sens, sigma_meta=sigma_meta_subject, mu=mu, noise_model='beta')
        stimulus_retest, choices_retest, correct_retest, confidence_retest, confidence_disc_retest, nratings_retest = \
            gen_data(nsubjects, nsamp, sigmas_sens=sigma_sens, sigma_meta=sigma_meta_subject, mu=mu, noise_model='beta')

        # compute hmeta-d'-based Mratio for both test and retest
        mratio_hmeta_test = np.array(type2_SDT_MLE_group(stimulus_test, choices_test, confidence_disc_test,
                                                         nratings_test).M_ratio)[0]
        mratio_hmeta_retest = np.array(type2_SDT_MLE_group(stimulus_retest, choices_retest, confidence_disc_retest,
                                                           nratings_retest).M_ratio)[0]

        # subtract minimum for NMAE
        mratio_hmeta_test_zb = mratio_hmeta_test - min(mratio_hmeta_test.min(), mratio_hmeta_retest.min())
        mratio_hmeta_retest_zb = mratio_hmeta_retest - min(mratio_hmeta_test.min(), mratio_hmeta_retest.min())

        # store metadata
        cond = (df.nsamples_id == i) & (df.sigma_sens_id == j)
        df.loc[cond, 'nsamp'] = nsamp
        df.loc[cond, 'seed'] = run
        df.loc[cond, 'sigma_sens'] = sigma_sens
        df.loc[cond, 'sigma_meta_min'] = sigma_meta_range[0]
        df.loc[cond, 'sigma_meta_max'] = sigma_meta_range[1]

        # compute and store test-retest reliabilities
        df.loc[cond, 'mratio_hmeta'] = (np.mean(mratio_hmeta_test) + np.mean(mratio_hmeta_retest)) / 2
        df.loc[cond, 'mratio_hmeta_pearson'] = pearsonr(mratio_hmeta_test, mratio_hmeta_retest)[0]
        df.loc[cond, 'mratio_hmeta_mae'] = np.mean(np.abs(mratio_hmeta_test - mratio_hmeta_retest))
        df.loc[cond, 'mratio_hmeta_nmae'] = \
            np.mean(np.abs(mratio_hmeta_test_zb - mratio_hmeta_retest_zb)) / \
            (0.5*(np.mean(np.abs(mratio_hmeta_test_zb - np.mean(mratio_hmeta_retest_zb))) +
                  np.mean(np.abs(mratio_hmeta_retest_zb - np.mean(mratio_hmeta_test_zb)))))

    print(f'\t[run={run}] Finished nsamp {i + 1} / {nsamples_list_len}: {default_timer() - t0:.1f} secs')

df.to_pickle(f'../data/hmeta_rel/{Path(__file__).stem}_{noise_model}_{run:03g}.pkl', protocol=4)
