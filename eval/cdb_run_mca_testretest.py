import os
from datetime import timedelta
from glob import glob
from timeit import default_timer as timer

import numpy as np
import pandas as pd
from scipy.stats import entropy

from cdb_helper import preprocess, meta_info, preprocess2
from type2SDT_hmeta import type2_SDT_MLE_group, type2_SDT_MLE_groupCorr
from type2SDT import type2_SDT_MLE, type2roc_disc2

files = sorted(glob('../data/CDB/*.csv'))
nfiles = len(files)

columns = dict(
    study_id='int',
    study_name='string',
    zsubject='int',
    subject='int',
    perf='float',
    conf='float',
    category='string',
    category_id='int',
    feedback_given='float',
    any_feedback='float',
    perception='int',
    memory='int',
    cognitive='int',
    motor='int',
    mixed='int',
    nsubjects='int',
    min_trials_per_subject='int',
    nratings='float',
    continuous='float',
    conf_dec_simu='float',
    trial_feedback='float',
    block_feedback='float'
)


for var in ('nsamples', 'perf', 'conf', 'mratio', 'mratio_hmeta', 'mratio_hmetacorr', 'mratio_reg0', 'mratio_reg', 'mratio_reg2', 'mratio_con0', 'mratio_con', 'mratio_con2', 'mdiff', 'metad1', 'd1', 'auc', 'entropy'):
    for suffix in ('', '_test', '_retest'):
        columns.update({f'{var}{suffix}': 'int' if var in ('nsamples', ) else 'float'})
del columns['mratio_hmetacorr']

def loop(f):
    start = timer()
    file = files[f]
    print(f'File {f + 1} / {nfiles} [{file}]')
    d = preprocess(file)

    nsubjects = len(d[~d.Subj_idx.isna()].Subj_idx.unique())
    nratings = len(d[~d.confidence_meta.isna()].confidence_meta.unique())

    df = pd.DataFrame(index=range(nsubjects), columns=columns.keys())

    df['study_id'] = f
    df['study_name'] = os.path.split(file)[1].split('.')[0][5:]

    stimulus_all, choice_all, confidence_all = [None] * nsubjects, [None] * nsubjects, [None] * nsubjects
    stimulus_all_test, choice_all_test, confidence_all_test = [None]*nsubjects, [None]*nsubjects, [None]*nsubjects
    stimulus_all_retest, choice_all_retest, confidence_all_retest = [None] * nsubjects, [None] * nsubjects, [None] * nsubjects
    subjects_all_valid, subjects_all_testretest_valid = [], []

    stim = sorted(list(d[~d.Stimulus.isna()].Stimulus.unique()))

    for i, s in enumerate(sorted(d[~d.Subj_idx.isna()].Subj_idx.unique())):
        # print(f'\tSubject {i + 1} / {nsubjects}')
        df.loc[i, 'subject'] = i
        if 'feedback_given' in d:
            df.loc[i, 'feedback_given'] = d[d.Subj_idx == s].feedback_given.astype(int).mean()
        stimulus_ = np.array(d[d.Subj_idx == s].Stimulus)
        stimulus = np.full(len(stimulus_), np.nan)
        stimulus[stimulus_ == stim[0]] = 0
        stimulus[stimulus_ == stim[1]] = 1
        choice_ = np.array(d[d.Subj_idx == s].Response)
        choice = np.full(len(choice_), np.nan)
        choice[choice_ == stim[0]] = 0
        choice[choice_ == stim[1]] = 1
        confidence = np.array(d[d.Subj_idx == s].confidence_meta)
        confidence_norm = np.array(d[d.Subj_idx == s].confidence_norm)
        correct = np.array(stimulus == choice, float)
        correct[np.isnan(choice)] = np.nan

        stimulus_test = stimulus[::2]
        choice_test = choice[::2]
        confidence_test = confidence[::2]
        confidence_norm_test = confidence_norm[::2]
        correct_test = correct[::2]
        stimulus_retest = stimulus[1::2]
        choice_retest = choice[1::2]
        confidence_retest = confidence[1::2]
        confidence_norm_retest = confidence_norm[1::2]
        correct_retest = correct[1::2]


        stimulus_ = stimulus[~np.isnan(choice) & ~np.isnan(confidence)]
        choice_ = choice[~np.isnan(choice) & ~np.isnan(confidence)]
        confidence_ = confidence[~np.isnan(choice) & ~np.isnan(confidence)]
        confidence_norm_ = confidence_norm[~np.isnan(choice) & ~np.isnan(confidence_norm)]
        correct_ = correct[~np.isnan(choice) & ~np.isnan(confidence)]
        if len(stimulus_):
            stimulus_all[i] = stimulus_
            choice_all[i] = choice_
            confidence_all[i] = confidence_
            subjects_all_valid += [i]
            fit = type2_SDT_MLE(stimulus_, choice_, confidence_, nratings)
            df.loc[i, 'mratio'] = fit.M_ratio
            df.loc[i, 'nsamples'] = len(confidence_)
            df.loc[i, 'mdiff'] = fit.M_diff
            df.loc[i, 'metad1'] = fit.meta_da
            df.loc[i, 'd1'] = fit.da
            df.loc[i, 'auc'] = type2roc_disc2(stimulus_, choice_, confidence_-1, nRatings=nratings)
            df.loc[i, 'entropy'] = entropy(np.bincount(confidence_.astype(int)-1, minlength=nratings)) / np.log(nratings)
            df.loc[i, 'conf'] = np.nanmean(confidence_norm_)
            df.loc[i, 'perf'] = np.nanmean(correct_)
        else:
            df.loc[i, 'nsamples'] = 0

        stimulus_test_ = stimulus_test[~np.isnan(choice_test) & ~np.isnan(confidence_test)]
        choice_test_ = choice_test[~np.isnan(choice_test) & ~np.isnan(confidence_test)]
        confidence_test_ = confidence_test[~np.isnan(choice_test) & ~np.isnan(confidence_test)]
        confidence_norm_test_ = confidence_norm_test[~np.isnan(choice_test) & ~np.isnan(confidence_norm_test)]
        correct_test_ = correct_test[~np.isnan(choice_test) & ~np.isnan(confidence_test)]
        if len(stimulus_test_):
            fit_test = type2_SDT_MLE(stimulus_test_, choice_test_, confidence_test_, nratings)
            df.loc[i, 'mratio_test'] = fit_test.M_ratio
            df.loc[i, 'nsamples_test'] = len(confidence_test_)
            df.loc[i, 'mdiff_test'] = fit_test.M_diff
            df.loc[i, 'metad1_test'] = fit_test.meta_da
            df.loc[i, 'd1_test'] = fit_test.da
            df.loc[i, 'perf_test'] = np.nanmean(correct_test_)
            df.loc[i, 'auc_test'] = type2roc_disc2(stimulus_test_, choice_test_, confidence_test_-1, nRatings=nratings)
            df.loc[i, 'entropy_test'] = entropy(np.bincount(confidence_test_.astype(int) - 1, minlength=nratings)) / np.log(nratings)
            df.loc[i, 'conf_test'] = np.nanmean(confidence_norm_test_)
        else:
            df.loc[i, 'nsamples_test'] = 0
        stimulus_retest_ = stimulus_retest[~np.isnan(choice_retest) & ~np.isnan(confidence_retest)]
        choice_retest_ = choice_retest[~np.isnan(choice_retest) & ~np.isnan(confidence_retest)]
        confidence_retest_ = confidence_retest[~np.isnan(choice_retest) & ~np.isnan(confidence_retest)]
        confidence_norm_retest_ = confidence_norm_retest[~np.isnan(choice_retest) & ~np.isnan(confidence_norm_retest)]
        correct_retest_ = correct_retest[~np.isnan(choice_retest) & ~np.isnan(confidence_retest)]
        if (len(stimulus_test_) > 0) and (len(stimulus_retest_) > 0):
            stimulus_all_retest[i] = stimulus_retest_
            choice_all_retest[i] = choice_retest_
            confidence_all_retest[i] = confidence_retest_
            stimulus_all_test[i] = stimulus_test_
            choice_all_test[i] = choice_test_
            confidence_all_test[i] = confidence_test_
            subjects_all_testretest_valid += [i]
        if len(stimulus_retest_):
            fit_retest = type2_SDT_MLE(stimulus_retest_, choice_retest_, confidence_retest_, nratings)
            df.loc[i, 'mratio_retest'] = fit_retest.M_ratio
            df.loc[i, 'nsamples_retest'] = len(confidence_retest_)
            df.loc[i, 'mdiff_retest'] = fit_retest.M_diff
            df.loc[i, 'metad1_retest'] = fit_retest.meta_da
            df.loc[i, 'd1_retest'] = fit_retest.da
            df.loc[i, 'perf_retest'] = np.nanmean(correct_retest_)
            df.loc[i, 'auc_retest'] = type2roc_disc2(stimulus_retest_, choice_retest_, confidence_retest_-1, nRatings=nratings)
            df.loc[i, 'entropy_retest'] = entropy(np.bincount(confidence_retest_.astype(int) - 1, minlength=nratings)) / np.log(nratings)
            df.loc[i, 'conf_retest'] = np.nanmean(confidence_norm_retest_)
        else:
            df.loc[i, 'nsamples_retest'] = 0

        for v in ('', '_test', '_retest'):
            df[f'mratio_con0{v}'] = df[f'mratio{v}']
            df.loc[~df[f'mratio_con0{v}'].isna() & (df[f'mratio_con0{v}'] < -0.5), f'mratio_con0{v}'] = -0.5
            df.loc[~df[f'mratio_con0{v}'].isna() & (df[f'mratio_con0{v}'] > 2), f'mratio_con0{v}'] = 2

            df[f'mratio_con{v}'] = df[f'mratio{v}']
            df.loc[~df[f'mratio_con{v}'].isna() & (df[f'mratio_con{v}'] < 0), f'mratio_con{v}'] = 0
            df.loc[~df[f'mratio_con{v}'].isna() & (df[f'mratio_con{v}'] > 1.5), f'mratio_con{v}'] = 1.5

            df[f'mratio_con2{v}'] = df[f'mratio{v}']
            df.loc[~df[f'mratio_con2{v}'].isna() & (df[f'mratio_con2{v}'] < 0), f'mratio_con2{v}'] = 0
            df.loc[~df[f'mratio_con2{v}'].isna() & (df[f'mratio_con2{v}'] > 1), f'mratio_con2{v}'] = 1

            df[f'mratio_reg0{v}'] = df[f'mratio{v}']
            df.loc[df[f'd1{v}'] < 0.1, f'mratio_reg0{v}'] = np.nan
            df.loc[df[f'entropy{v}'] < 0.1, f'mratio_reg0{v}'] = np.nan
            df.loc[~df[f'mratio_reg0{v}'].isna() & (df[f'mratio_reg0{v}'] < -0.5), f'mratio_reg0{v}'] = -0.5
            df.loc[~df[f'mratio_reg0{v}'].isna() & (df[f'mratio_reg0{v}'] > 2), f'mratio_reg0{v}'] = 2

            df[f'mratio_reg{v}'] = df[f'mratio{v}']
            df.loc[df[f'd1{v}'] < 0.1, f'mratio_reg{v}'] = np.nan
            df.loc[df[f'entropy{v}'] < 0.1, f'mratio_reg{v}'] = np.nan
            df.loc[~df[f'mratio_reg{v}'].isna() & (df[f'mratio_reg{v}'] < 0), f'mratio_reg{v}'] = 0
            df.loc[~df[f'mratio_reg{v}'].isna() & (df[f'mratio_reg{v}'] > 1.5), f'mratio_reg{v}'] = 1.5

            df[f'mratio_reg2{v}'] = df[f'mratio{v}']
            df.loc[df[f'd1{v}'] < 0.1, f'mratio_reg2{v}'] = np.nan
            df.loc[df[f'entropy{v}'] < 0.3, f'mratio_reg2{v}'] = np.nan
            df.loc[~df[f'mratio_reg2{v}'].isna() & (df[f'mratio_reg2{v}'] < 0), f'mratio_reg2{v}'] = 0
            df.loc[~df[f'mratio_reg2{v}'].isna() & (df[f'mratio_reg2{v}'] > 1), f'mratio_reg2{v}'] = 1

    try:
        fit_hmeta = type2_SDT_MLE_group(
            [v for v in stimulus_all if v is not None],
            [v for v in choice_all if v is not None],
            [v for v in confidence_all if v is not None],
            nratings
        )
        df.loc[df.subject.isin(subjects_all_valid), 'mratio_hmeta'] = np.array(fit_hmeta.M_ratio[0])
    except:
        print(f'Hmeta failed for subject {i} (s={s})')

    try:
        fit_hmeta = type2_SDT_MLE_group(
            [v for v in stimulus_all_test if v is not None],
            [v for v in choice_all_test if v is not None],
            [v for v in confidence_all_test if v is not None],
            nratings
        )
        df.loc[df.subject.isin(subjects_all_testretest_valid), 'mratio_hmeta_test'] = np.array(fit_hmeta.M_ratio[0])
    except:
        print(f'Test Hmeta failed for subject {i} (s={s})')

    try:
        fit_hmeta = type2_SDT_MLE_group(
            [v for v in stimulus_all_retest if v is not None],
            [v for v in choice_all_retest if v is not None],
            [v for v in confidence_all_retest if v is not None],
            nratings
        )
        df.loc[df.subject.isin(subjects_all_testretest_valid), 'mratio_hmeta_retest'] = np.array(fit_hmeta.M_ratio[0])
    except:
        print(f'Retest Hmeta failed for subject {i} (s={s})')

    try:
        fit_hmeta = type2_SDT_MLE_groupCorr(
            [v for v in stimulus_all_test if v is not None],
            [v for v in choice_all_test if v is not None],
            [v for v in confidence_all_test if v is not None],
            [v for v in stimulus_all_retest if v is not None],
            [v for v in choice_all_retest if v is not None],
            [v for v in confidence_all_retest if v is not None],
            nratings
        )
        df.loc[df.subject.isin(subjects_all_testretest_valid), 'mratio_hmetacorr_test'] = np.array(fit_hmeta.M_ratio)[:, 0]
        df.loc[df.subject.isin(subjects_all_testretest_valid), 'mratio_hmetacorr_retest'] = np.array(fit_hmeta.M_ratio)[:, 1]
    except:
        print(f'Test-retest hmeta failed for subject {i} (s={s})')

    print('\t', timedelta(seconds=timer()-start))
    return df

dfs = [None]*nfiles
for f in range(nfiles):
    dfs[f] = loop(f)
# dfs = multiprocessing.Pool(8).map(loop, range(nfiles))

z = pd.concat(dfs).reset_index(drop=True)


z = preprocess2(z)
z = meta_info(z)

z = z[list(columns.keys())].reset_index(drop=True).astype(columns)


z.to_pickle('../data/mca_test_retest.pkl')
