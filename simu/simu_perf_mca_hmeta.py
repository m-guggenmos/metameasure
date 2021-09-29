#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Matthias Guggenmos <mg.corresponding@gmail.com>
# 2021

# Simulation: type 1 performance dependency of Mratio based on hmeta-d'

import os
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import pandas as pd

from simu_confidence import gen_data

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.type2SDT_hmeta import type2_SDT_MLE_group  # noqa


np.random.seed(1)

# Ns
####

nsamples = 10000
nsubjects = 1000
# nsamples = 300
# nsubjects = 10

mu = 0.5
sigma_sens_list = [0, 0.0005]
for sigma_sens_id in range(98):
    sigma_sens_list += [sigma_sens_list[-1] + 1.0681355555555*(sigma_sens_list[-1] - sigma_sens_list[-2])]
nsigmasens = len(sigma_sens_list)
noise_model = 'beta'
# noise_model = 'lognorm'
# noise_model = 'censored_norm'

sigma_meta_list = dict(
    beta=np.arange(0, 0.51, 0.1),
    lognorm=np.exp(np.arange(0, 4, 0.2))-1,
    censored_norm=np.arange(0, 2, 0.1)
)[noise_model]
nsigmameta = len(sigma_meta_list)

sigma_sens_ids = range(47, 49)
for i, sigma_sens_id in enumerate(sigma_sens_ids):
    print(f'({i + 1} / {len(sigma_sens_ids)}) sigma_sens_id: {sigma_sens_id}')

    np.random.seed(sigma_sens_id)
    sigma_sens = sigma_sens_list[sigma_sens_id]

    t0 = default_timer()

    df = pd.DataFrame(index=range(nsubjects*nsigmameta))
    df['subject'] = np.tile(range(nsubjects), nsigmameta)
    df['sigma_meta_id'] = np.repeat(range(nsigmameta), nsubjects)
    df['sigma_sens_id'] = sigma_sens_id
    df['sigma_sens'] = sigma_sens

    for j, sigma_meta in enumerate(sigma_meta_list):

        t1 = default_timer()

        print(f'\tSigma meta {j + 1} / {len(sigma_meta_list)} [{sigma_meta:.3f}]')
        print(f'\tsigma_sens_id: {sigma_sens_id}')

        stimulus, choices, correct, confidence, confidence_disc, nratings = \
            gen_data(nsubjects, nsamples, sigmas_sens=sigma_sens, sigma_meta=sigma_meta, mu=mu, noise_model=noise_model)

        mratio_hmeta = np.array(type2_SDT_MLE_group(stimulus, choices, confidence_disc, nratings).M_ratio)[0]

        cond = df.sigma_meta_id == j
        df.loc[cond, 'sigma_meta'] = sigma_meta
        df.loc[cond, 'mratio_hmeta'] = mratio_hmeta
        print(f'\t\tTook {default_timer() - t1:.1f} secs')

    print(f'\t\tFinished parameter {sigma_sens_id + 1} / {nsigmasens}: {default_timer() - t0:.1f} secs')

    df.to_pickle(f'../data/hmeta/{Path(__file__).stem}_{noise_model}_{sigma_sens_id:02g}.pkl', protocol=4)
