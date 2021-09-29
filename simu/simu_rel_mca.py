#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Matthias Guggenmos <mg.corresponding@gmail.com>
# 2021

# Simulation: test-retest reliability of metacognitive performance measures (except Mratio based on hmeta-d')

import os
import sys
from pathlib import Path
from timeit import default_timer

import numpy as np
import pandas as pd
from multiprocessing_on_dill.pool import Pool
from scipy.stats import norm, uniform, pearsonr

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.type2SDT import type2_SDT_MLE, type2roc  # noqa
from simu_confidence import gen_data  # noqa


HOME = os.path.expanduser('~')

# Parameters
mu = 0.5
sigma_sens_list = [2, 1, 0.64, 0.48, 0.41, 0.3, 0.24, 0.195, 0.155, 0]
noise_model = 'beta'
sigma_meta_range = [0, 0.5]

# Ns
niter = 20
nsamples_list = np.hstack((np.arange(25, 1001, 25), 2500, 5000, 10000))
nsamples_list_len = len(nsamples_list)
nsubjects = 100
nsigmasens = len(sigma_sens_list)


def loop(n):
    seed = n
    np.random.seed(seed+220)

    t0 = default_timer()

    # Create storage data frame
    df = pd.DataFrame(index=range(nsamples_list_len * nsigmasens))
    df['sigma_sens_id'] = np.tile(range(nsigmasens), nsamples_list_len)
    df['nsamples_id'] = np.repeat(range(nsamples_list_len), nsigmasens)

    # generate uniform samples of sigma_meta
    sigma_meta_subject = uniform(loc=sigma_meta_range[0], scale=sigma_meta_range[1]).rvs(nsubjects)

    for i, nsamp in enumerate(nsamples_list):
        d1_test, d1_retest = np.full((nsigmasens, nsubjects), np.nan), np.full((nsigmasens, nsubjects), np.nan)
        d1_fit_test, d1_fit_retest = np.full((nsigmasens, nsubjects), np.nan), np.full((nsigmasens, nsubjects), np.nan)
        performance_test, performance_retest = np.full((nsigmasens, nsubjects), np.nan), \
            np.full((nsigmasens, nsubjects), np.nan)

        for j, sigma_sens in enumerate(sigma_sens_list):

            # generate test and retest data
            stimulus_test, choices_test, correct_test, confidence_test, confidence_disc_test, nratings_test = \
                gen_data(nsubjects, nsamp, sigmas_sens=sigma_sens, sigma_meta=sigma_meta_subject, mu=mu,
                         noise_model=noise_model)
            stimulus_retest, choices_retest, correct_retest, confidence_retest, confidence_disc_retest, \
                nratings_retest = gen_data(nsubjects, nsamp, sigmas_sens=sigma_sens, sigma_meta=sigma_meta_subject,
                                           mu=mu, noise_model=noise_model)

            conf_test, conf_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            metad1_test, metad1_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            auc_test, auc_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            mdiff_test, mdiff_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            mratio, mratio_test, mratio_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan), \
                np.full(nsubjects, np.nan)
            logmratio_test, logmratio_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            logmratio001_test, logmratio001_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            mratio_bounded_test, mratio_bounded_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            mratio_bounded1_test, mratio_bounded1_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            mratio_logistic_test, mratio_logistic_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            mratio_logistic2_test, mratio_logistic2_retest = np.full(nsubjects, np.nan), np.full(nsubjects, np.nan)
            for s in range(nsubjects):
                performance_test[j, s] = np.mean(choices_test[s] == stimulus_test[s])
                conf_test[s] = np.mean(confidence_test[s])
                d1_test[j, s] = norm.ppf(np.mean(choices_test[s, stimulus_test[s] == 1])) - \
                    norm.ppf(np.mean(choices_test[s, stimulus_test[s] == 0]))
                fit_test = type2_SDT_MLE(stimulus_test[s], choices_test[s], confidence_disc_test[s], nratings_test)
                mratio_test[s] = fit_test.M_ratio
                logmratio_test[s] = np.log(np.maximum(0.1, fit_test.M_ratio))
                logmratio001_test[s] = np.log(np.maximum(0.01, fit_test.M_ratio))
                mratio_bounded_test[s] = np.minimum(1.6, np.maximum(0, fit_test.M_ratio))
                mratio_bounded1_test[s] = np.minimum(1, np.maximum(0, fit_test.M_ratio))
                mratio_logistic_test[s] = (1 / (1 + np.exp(-max(-1000, fit_test.M_ratio - 0.8),
                                                           dtype=np.float128))).astype(np.float64)
                mratio_logistic2_test[s] = (1 / (1 + np.exp(-2.5*max(-1000, fit_test.M_ratio - 0.8),
                                                            dtype=np.float128))).astype(np.float64)
                mdiff_test[s] = fit_test.M_diff
                metad1_test[s] = fit_test.meta_da
                d1_fit_test[j, s] = fit_test.d1
                auc_test[s] = type2roc((choices_test[s] == stimulus_test[s]).astype(int), confidence_test[s])  # noqa

                performance_retest[j, s] = np.mean(choices_retest[s] == stimulus_retest[s])
                conf_retest[s] = np.mean(confidence_retest[s])
                d1_retest[j, s] = norm.ppf(np.mean(choices_retest[s, stimulus_retest[s] == 1])) - \
                    norm.ppf(np.mean(choices_retest[s, stimulus_retest[s] == 0]))
                fit_retest = type2_SDT_MLE(stimulus_retest[s], choices_retest[s], confidence_disc_retest[s],
                                           nratings_retest)
                mratio_retest[s] = fit_retest.M_ratio
                logmratio_retest[s] = np.log(np.maximum(0.1, fit_retest.M_ratio))
                logmratio001_retest[s] = np.log(np.maximum(0.01, fit_retest.M_ratio))
                mratio_bounded_retest[s] = np.minimum(1.6, np.maximum(0, fit_retest.M_ratio))
                mratio_bounded1_retest[s] = np.minimum(1, np.maximum(0, fit_retest.M_ratio))
                mratio_logistic_retest[s] = (1 / (1 + np.exp(-np.maximum(-1000, fit_retest.M_ratio - 0.8),
                                                             dtype=np.float128))).astype(np.float64)
                mratio_logistic2_retest[s] = (1 / (1 + np.exp(-2.5*np.maximum(-1000, fit_retest.M_ratio - 0.8),
                                                              dtype=np.float128))).astype(np.float64)
                mdiff_retest[s] = fit_retest.M_diff
                metad1_retest[s] = fit_retest.meta_da
                d1_fit_retest[j, s] = fit_retest.d1
                auc_retest[s] = type2roc((choices_retest[s] == stimulus_retest[s]).astype(int), confidence_retest[s])  # noqa
                
                mratio[s] = type2_SDT_MLE(np.hstack((stimulus_test[s], stimulus_retest[s])),
                                          np.hstack((choices_test[s], choices_retest[s])),
                                          np.hstack((confidence_test[s], confidence_retest[s])), nratings_test).M_ratio

            mratio_exclude_test = mratio_test[(mratio_test >= 0) & (mratio_test <= 1.6) & (mratio_retest >= 0) &
                                              (mratio_retest <= 1.6)]
            mratio_exclude_retest = mratio_retest[(mratio_test >= 0) & (mratio_test <= 1.6) & (mratio_retest >= 0) &
                                                  (mratio_retest <= 1.6)]
            mratio_exclude2_test = mratio_test[(mratio >= 0) & (mratio <= 1.6)]
            mratio_exclude2_retest = mratio_retest[(mratio >= 0) & (mratio <= 1.6)]
            mratio_bound10_test = np.maximum(-10, np.minimum(10, mratio_test))
            mratio_bound10_retest = np.maximum(-10, np.minimum(10, mratio_retest))

            # subtract minimum for NMAE
            mratio_exclude_test_zb = mratio_exclude_test - \
                min(np.min(mratio_exclude_test), np.min(mratio_exclude_retest))
            mratio_exclude_retest_zb = mratio_exclude_retest - \
                min(np.min(mratio_exclude_test), np.min(mratio_exclude_retest))
            mratio_exclude2_test_zb = mratio_exclude2_test - \
                min(np.min(mratio_exclude2_test), np.min(mratio_exclude2_retest))
            mratio_exclude2_retest_zb = mratio_exclude2_retest - \
                min(np.min(mratio_exclude2_test), np.min(mratio_exclude2_retest))
            mratio_test_zb = mratio_test - min(np.min(mratio_test), np.min(mratio_retest))
            mratio_retest_zb = mratio_retest - min(np.min(mratio_test), np.min(mratio_retest))
            mratio_bound10_test_zb = mratio_bound10_test - \
                min(np.min(mratio_bound10_test), np.min(mratio_bound10_retest))
            mratio_bound10_retest_zb = mratio_bound10_retest - \
                min(np.min(mratio_bound10_test), np.min(mratio_bound10_retest))
            logmratio_test_zb = logmratio_test - min(np.min(logmratio_test), np.min(logmratio_retest))
            logmratio_retest_zb = logmratio_retest - min(np.min(logmratio_test), np.min(logmratio_retest))
            mratio_bounded_test_zb = mratio_bounded_test - \
                min(np.min(mratio_bounded_test), np.min(mratio_bounded_retest))
            mratio_bounded_retest_zb = mratio_bounded_retest - \
                min(np.min(mratio_bounded_test), np.min(mratio_bounded_retest))
            logmratio001_test_zb = logmratio001_test - min(np.min(logmratio001_test), np.min(logmratio001_retest))
            logmratio001_retest_zb = logmratio001_retest - min(np.min(logmratio001_test), np.min(logmratio001_retest))
            mratio_bounded1_test_zb = mratio_bounded1_test - \
                min(np.min(mratio_bounded1_test), np.min(mratio_bounded1_retest))
            mratio_bounded1_retest_zb = mratio_bounded1_retest - \
                min(np.min(mratio_bounded1_test), np.min(mratio_bounded1_retest))
            mratio_logistic_test_zb = mratio_logistic_test - \
                min(np.min(mratio_logistic_test), np.min(mratio_logistic_retest))
            mratio_logistic_retest_zb = mratio_logistic_retest - \
                min(np.min(mratio_logistic_test), np.min(mratio_logistic_retest))
            mratio_logistic2_test_zb = mratio_logistic2_test - \
                min(np.min(mratio_logistic2_test), np.min(mratio_logistic2_retest))
            mratio_logistic2_retest_zb = mratio_logistic2_retest - \
                min(np.min(mratio_logistic2_test), np.min(mratio_logistic2_retest))
            mdiff_test_zb = mdiff_test - min(np.min(mdiff_test), np.min(mdiff_retest))
            mdiff_retest_zb = mdiff_retest - min(np.min(mdiff_test), np.min(mdiff_retest))
            metad1_test_zb = metad1_test - min(np.min(metad1_test), np.min(metad1_retest))
            metad1_retest_zb = metad1_retest - min(np.min(metad1_test), np.min(metad1_retest))
            auc_test_zb = auc_test - min(np.min(auc_test), np.min(auc_retest))
            auc_retest_zb = auc_retest - min(np.min(auc_test), np.min(auc_retest))

            cond = (df.nsamples_id == i) & (df.sigma_sens_id == j)

            # store metadata and averages
            df.loc[cond, 'nsamp'] = nsamp
            df.loc[cond, 'seed'] = seed
            df.loc[cond, 'sigma_sens'] = sigma_sens
            df.loc[cond, 'sigma_meta_min'] = sigma_meta_range[0]
            df.loc[cond, 'sigma_meta_max'] = sigma_meta_range[1]
            df.loc[cond, 'performance'] = (np.mean(performance_test[j]) + np.mean(performance_retest[j])) / 2
            df.loc[cond, 'confidence'] = (np.mean(conf_test) + np.mean(conf_retest)) / 2
            df.loc[cond, 'd1'] = (np.mean(d1_test[j]) + np.mean(d1_retest[j])) / 2
            df.loc[cond, 'mratio'] = (np.mean(mratio_test) + np.mean(mratio_retest)) / 2
            df.loc[cond, 'logmratio'] = (np.mean(logmratio_test) + np.mean(logmratio_retest)) / 2
            df.loc[cond, 'logmratio001'] = (np.mean(logmratio001_test) + np.mean(logmratio001_retest)) / 2
            df.loc[cond, 'mratio_logistic'] = (np.mean(mratio_logistic_test) + np.mean(mratio_logistic_retest)) / 2
            df.loc[cond, 'mratio_logistic2'] = (np.mean(mratio_logistic2_test) + np.mean(mratio_logistic2_retest)) / 2
            df.loc[cond, 'mratio_bounded'] = (np.mean(mratio_bounded_test) + np.mean(mratio_bounded_retest)) / 2
            df.loc[cond, 'mratio_bounded1'] = (np.mean(mratio_bounded1_test) + np.mean(mratio_bounded1_retest)) / 2
            df.loc[cond, 'mratio_exclude'] = (np.mean(mratio_exclude_test) + np.mean(mratio_exclude_retest)) / 2
            df.loc[cond, 'mratio_exclude2'] = (np.mean(mratio_exclude2_test) + np.mean(mratio_exclude2_retest)) / 2
            df.loc[cond, 'mdiff'] = (np.mean(mdiff_test) + np.mean(mdiff_retest)) / 2
            df.loc[cond, 'metad1'] = (np.mean(metad1_test) + np.mean(metad1_retest)) / 2
            df.loc[cond, 'd1_fit'] = (np.mean(d1_fit_test[j]) + np.mean(d1_fit_retest[j])) / 2
            df.loc[cond, 'auc'] = (np.mean(auc_test) + np.mean(auc_retest)) / 2

            # compute and store test-retest reliabilities
            df.loc[cond, 'mratio_pearson'] = pearsonr(mratio_test, mratio_retest)[0]
            df.loc[cond, 'mratio_mae'] = np.mean(np.abs(mratio_test - mratio_retest))
            df.loc[cond, 'mratio_nmae'] = \
                np.mean(np.abs(mratio_test_zb - mratio_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_test_zb - np.mean(mratio_retest_zb))) +
                      np.mean(np.abs(mratio_retest_zb - np.mean(mratio_test_zb)))))
            df.loc[cond, 'mratio_bound10_pearson'] = pearsonr(mratio_bound10_test, mratio_bound10_retest)[0]
            df.loc[cond, 'mratio_bound10_mae'] = np.mean(np.abs(mratio_bound10_test - mratio_bound10_retest))
            df.loc[cond, 'mratio_bound10_nmae'] = \
                np.mean(np.abs(mratio_bound10_test_zb - mratio_bound10_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_bound10_test_zb - np.mean(mratio_bound10_retest_zb))) +
                      np.mean(np.abs(mratio_bound10_retest_zb - np.mean(mratio_bound10_test_zb)))))
            df.loc[cond, 'logmratio_pearson'] = pearsonr(logmratio_test, logmratio_retest)[0]
            df.loc[cond, 'logmratio_mae'] = np.mean(np.abs(logmratio_test - logmratio_retest))
            df.loc[cond, 'logmratio_nmae'] = \
                np.mean(np.abs(logmratio_test_zb - logmratio_retest_zb)) / \
                (0.5*(np.mean(np.abs(logmratio_test_zb - np.mean(logmratio_retest_zb))) +
                      np.mean(np.abs(logmratio_retest_zb - np.mean(logmratio_test_zb)))))
            df.loc[cond, 'logmratio001_pearson'] = pearsonr(logmratio001_test, logmratio001_retest)[0]
            df.loc[cond, 'logmratio001_mae'] = np.mean(np.abs(logmratio001_test - logmratio001_retest))
            df.loc[cond, 'logmratio001_nmae'] = \
                np.mean(np.abs(logmratio001_test_zb - logmratio001_retest_zb)) / \
                (0.5*(np.mean(np.abs(logmratio001_test_zb - np.mean(logmratio001_retest_zb))) +
                      np.mean(np.abs(logmratio001_retest_zb - np.mean(logmratio001_test_zb)))))
            df.loc[cond, 'mratio_bounded_pearson'] = pearsonr(mratio_bounded_test, mratio_bounded_retest)[0]
            df.loc[cond, 'mratio_bounded_mae'] = np.mean(np.abs(mratio_bounded_test - mratio_bounded_retest))
            df.loc[cond, 'mratio_bounded_nmae'] = \
                np.mean(np.abs(mratio_bounded_test_zb - mratio_bounded_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_bounded_test_zb - np.mean(mratio_bounded_retest_zb))) +
                      np.mean(np.abs(mratio_bounded_retest_zb - np.mean(mratio_bounded_test_zb)))))
            df.loc[cond, 'mratio_bounded1_pearson'] = pearsonr(mratio_bounded1_test, mratio_bounded1_retest)[0]
            df.loc[cond, 'mratio_bounded1_mae'] = np.mean(np.abs(mratio_bounded1_test - mratio_bounded1_retest))
            df.loc[cond, 'mratio_bounded1_nmae'] = \
                np.mean(np.abs(mratio_bounded1_test_zb - mratio_bounded1_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_bounded1_test_zb - np.mean(mratio_bounded1_retest_zb))) +
                      np.mean(np.abs(mratio_bounded1_retest_zb - np.mean(mratio_bounded1_test_zb)))))
            df.loc[cond, 'mratio_exclude_pearson'] = pearsonr(mratio_exclude_test, mratio_exclude_retest)[0]
            df.loc[cond, 'mratio_exclude_mae'] = np.mean(np.abs(mratio_exclude_test - mratio_exclude_retest))
            df.loc[cond, 'mratio_exclude_nmae'] = \
                np.mean(np.abs(mratio_exclude_test_zb - mratio_exclude_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_exclude_test_zb - np.mean(mratio_exclude_retest_zb))) +
                      np.mean(np.abs(mratio_exclude_retest_zb - np.mean(mratio_exclude_test_zb)))))
            df.loc[cond, 'mratio_exclude2_pearson'] = pearsonr(mratio_exclude2_test, mratio_exclude2_retest)[0]
            df.loc[cond, 'mratio_exclude2_mae'] = np.mean(np.abs(mratio_exclude2_test - mratio_exclude2_retest))
            df.loc[cond, 'mratio_exclude2_nmae'] = \
                np.mean(np.abs(mratio_exclude2_test_zb - mratio_exclude2_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_exclude2_test_zb - np.mean(mratio_exclude2_retest_zb))) +
                      np.mean(np.abs(mratio_exclude2_retest_zb - np.mean(mratio_exclude2_test_zb)))))
            df.loc[cond, 'mratio_logistic_pearson'] = pearsonr(mratio_logistic_test, mratio_logistic_retest)[0]
            df.loc[cond, 'mratio_logistic_mae'] = np.mean(np.abs(mratio_logistic_test - mratio_logistic_retest))
            df.loc[cond, 'mratio_logistic_nmae'] = \
                np.mean(np.abs(mratio_logistic_test_zb - mratio_logistic_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_logistic_test_zb - np.mean(mratio_logistic_retest_zb))) +
                      np.mean(np.abs(mratio_logistic_retest_zb - np.mean(mratio_logistic_test_zb)))))
            df.loc[cond, 'mratio_logistic2_pearson'] = pearsonr(mratio_logistic2_test, mratio_logistic2_retest)[0]
            df.loc[cond, 'mratio_logistic2_mae'] = np.mean(np.abs(mratio_logistic2_test - mratio_logistic2_retest))
            df.loc[cond, 'mratio_logistic2_nmae'] = \
                np.mean(np.abs(mratio_logistic2_test_zb - mratio_logistic2_retest_zb)) / \
                (0.5*(np.mean(np.abs(mratio_logistic2_test_zb - np.mean(mratio_logistic2_retest_zb))) +
                      np.mean(np.abs(mratio_logistic2_retest_zb - np.mean(mratio_logistic2_test_zb)))))
            df.loc[cond, 'mdiff_pearson'] = pearsonr(mdiff_test, mdiff_retest)[0]
            df.loc[cond, 'mdiff_mae'] = np.mean(np.abs(mdiff_test - mdiff_retest))
            df.loc[cond, 'mdiff_nmae'] = np.mean(np.abs(mdiff_test_zb - mdiff_retest_zb)) / \
                (0.5*(np.mean(np.abs(mdiff_test_zb - np.mean(mdiff_retest_zb))) +
                      np.mean(np.abs(mdiff_retest_zb - np.mean(mdiff_test_zb)))))
            df.loc[cond, 'metad1_pearson'] = pearsonr(metad1_test, metad1_retest)[0]
            df.loc[cond, 'metad1_mae'] = np.mean(np.abs(metad1_test - metad1_retest))
            df.loc[cond, 'metad1_nmae'] = np.mean(np.abs(metad1_test_zb - metad1_retest_zb)) / \
                (0.5*(np.mean(np.abs(metad1_test_zb - np.mean(metad1_retest_zb))) +
                      np.mean(np.abs(metad1_retest_zb - np.mean(metad1_test_zb)))))
            df.loc[cond, 'auc_pearson'] = pearsonr(auc_test, auc_retest)[0]
            df.loc[cond, 'auc_mae'] = np.mean(np.abs(auc_test - auc_retest))
            df.loc[cond, 'auc_nmae'] = np.mean(np.abs(auc_test_zb - auc_retest_zb)) / \
                (0.5*(np.mean(np.abs(auc_test_zb - np.mean(auc_retest_zb))) +
                      np.mean(np.abs(auc_retest_zb - np.mean(auc_test_zb)))))

        ind = [np.random.choice(range(100), 10, replace=False) for _ in range(nsigmasens)]
        performance_test = np.array([performance_test[j, ind[j]] for j in range(nsigmasens)]).flatten()
        performance_retest = np.array([performance_retest[j, ind[j]] for j in range(nsigmasens)]).flatten()
        d1_test = np.array([d1_test[j, ind[j]] for j in range(nsigmasens)]).flatten()
        d1_retest = np.array([d1_retest[j, ind[j]] for j in range(nsigmasens)]).flatten()
        exclude = ~np.isnan(d1_test) & ~np.isnan(d1_retest) & ~np.isinf(d1_test) & ~np.isinf(d1_retest)
        d1_test = d1_test[exclude]
        d1_retest = d1_retest[exclude]
        d1_fit_test = np.array([d1_fit_test[j, ind[j]] for j in range(nsigmasens)]).flatten()
        d1_fit_retest = np.array([d1_fit_retest[j, ind[j]] for j in range(nsigmasens)]).flatten()

        performance_test_zb = performance_test - min(np.min(performance_test), np.min(performance_retest))
        performance_retest_zb = performance_retest - min(np.min(performance_test), np.min(performance_retest))
        if len(d1_test) >= 3:
            d1_test_zb = d1_test - min(np.min(d1_test), np.min(d1_retest))
            d1_retest_zb = d1_retest - min(np.min(d1_test), np.min(d1_retest))
        d1_fit_test_zb = d1_fit_test - min(np.min(d1_fit_test), np.min(d1_fit_retest))
        d1_fit_retest_zb = d1_fit_retest - min(np.min(d1_fit_test), np.min(d1_fit_retest))

        cond = (df.nsamples_id == i) & (df.sigma_sens_id == 0)
        if len(d1_test) >= 3:
            df.loc[cond, 'd1_pearson'] = pearsonr(d1_test, d1_retest)[0]
            df.loc[cond, 'd1_mae'] = np.mean(np.abs(d1_test - d1_retest))
            df.loc[cond, 'd1_nmae'] = \
                np.mean(np.abs(d1_test_zb - d1_retest_zb)) / (0.5*(np.mean(np.abs(d1_test_zb - np.mean(d1_retest_zb))) +  # noqa
                                                                   np.mean(np.abs(d1_retest_zb - np.mean(d1_test_zb)))))
        df.loc[cond, 'd1_fit_pearson'] = pearsonr(d1_fit_test, d1_fit_retest)[0]
        df.loc[cond, 'd1_fit_mae'] = np.mean(np.abs(d1_fit_test - d1_fit_retest))
        df.loc[cond, 'd1_fit_nmae'] = np.mean(np.abs(d1_fit_test_zb - d1_fit_retest_zb)) / \
            (0.5*(np.mean(np.abs(d1_fit_test_zb - np.mean(d1_fit_retest_zb))) +
                  np.mean(np.abs(d1_fit_retest_zb - np.mean(d1_fit_test_zb)))))
        df.loc[cond, 'perf_pearson'] = pearsonr(performance_test, performance_retest)[0]
        df.loc[cond, 'perf_mae'] = np.mean(np.abs(performance_test - performance_retest))
        df.loc[cond, 'perf_nmae'] = np.mean(np.abs(performance_test_zb - performance_retest_zb)) / \
            (0.5*(np.mean(np.abs(performance_test_zb - np.mean(performance_retest_zb))) +
                  np.mean(np.abs(performance_retest_zb - np.mean(performance_test_zb)))))

    print(f'\tFinished iteration {n + 1} / {niter}: {default_timer() - t0:.1f} secs')

    return df


with Pool(14) as pool:
    results = list(pool.map(loop, range(niter)))
# results = [None] * niter
# for n in range(niter):
#     results[n] = loop(n)

# combine multiprocessing results in single dataframe
result = pd.concat(results, keys=range(niter)).reset_index().drop(columns='level_1').rename(columns=dict(level_0='iter'))  # noqa

# save results
result.to_pickle(f'../data/{Path(__file__).stem}_{noise_model}.pkl', protocol=4)
