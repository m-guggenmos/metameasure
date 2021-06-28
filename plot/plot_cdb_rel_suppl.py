import os
import sys
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_util import set_fontsize, savefig

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.regression import regress  # noqa


z_orig = pd.read_pickle('../data/mca_test_retest.pkl')
z_orig = z_orig[~z_orig.mratio.isna()]
z_orig['logmratio'] = np.log(np.maximum(0.01, z_orig['mratio'].values))
z_orig['logmratio_test'] = np.log(np.maximum(0.01, z_orig['mratio_test'].values))
z_orig['logmratio_retest'] = np.log(np.maximum(0.01, z_orig['mratio_retest'].values))

nsamples_list = np.arange(0, 701, 100)
nsamples_list_len = len(nsamples_list)
mca_test, mca_retest = dict(), dict()

nresample = 1000

reload = True

color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

if reload:

    for i, m in enumerate(('perf', 'd1', 'metad1', 'auc', 'mdiff')):
        print(f'mca: {m} ({i + 1} / 5)')

        mca_test[m], mca_retest[m] = [None] * nsamples_list_len, [None] * nsamples_list_len
        for j in range(1, nsamples_list_len):
            z = z_orig[(z_orig.nsamples_test >= nsamples_list[j - 1]) & (z_orig.nsamples_retest < nsamples_list[j])].copy()

            mca_test_orig = z.loc[~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna(), f'{m}_test'].values

            if m == 'mratio_hmetacorr':
                mca_origmean = (z[f'{m}_test'].mean() + z[f'{m}_retest'].mean()) / 2
            else:
                mca_origmean = z[m].mean()
            mca_test_origmean = z[f'{m}_test'].mean()
            for k, study in enumerate(sorted(z.study_id.unique())):
                cond = (z.study_id == study) & ~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna()
                if (len(z.loc[cond, f'{m}_test'].values) > 10) and (len(z.loc[cond, f'{m}_retest'].values) > 10):
                    z.loc[cond, f'{m}_test_uncon'] = z.loc[cond, f'{m}_test']
                    z.loc[cond, f'{m}_retest_uncon'] = z.loc[cond, f'{m}_retest']
                    z.loc[cond, f'{m}_test_uncon'] -= (np.hstack((z.loc[cond, f'{m}_test'].values, z.loc[cond, f'{m}_retest'].values)).mean() - mca_origmean)
                    z.loc[cond, f'{m}_retest_uncon'] -= (np.hstack((z.loc[cond, f'{m}_test'].values, z.loc[cond, f'{m}_retest'].values)).mean() - mca_origmean)

            mca_test[m][j - 1] = z.loc[~z[f'{m}_test_uncon'].isna(), f'{m}_test_uncon'].values
            mca_retest[m][j - 1] = z.loc[~z[f'{m}_retest_uncon'].isna(), f'{m}_retest_uncon'].values

    pickle.dump((mca_test, mca_retest), open(f'../data/{Path(__file__).stem}.pkl', 'wb'))


mca_test, mca_retest = pickle.load(open(f'../data/{Path(__file__).stem}.pkl', 'rb'))


mapping = dict(d1=r"$d'$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$", mratio_hmetacorr=r"$M_{ratio}$ (hier.)", logmratio=r"$\log\, M_{ratio}$", mratio_con2=r"bounded $M_{ratio}$", mdiff=r"$M_{diff}$")


df = pd.DataFrame()

measures = ['d1', 'metad1', 'auc', 'mdiff']
axes = [None]*len(measures)*2
xticks = np.arange(0, 701, 100)
plt.figure(figsize=(10, 3.5))

order = [1, 5, 2, 6, 3, 7, 4, 8]

for i, m in enumerate(measures):

    for k in range(2):

        axes[i*2+k] = plt.subplot(2, 4, order[i*2+k])

        rel = np.full(nsamples_list_len - 1, np.nan)
        rel_se = np.full(nsamples_list_len - 1, np.nan)
        acc = np.full(nsamples_list_len - 1, np.nan)
        acc_se = np.full(nsamples_list_len - 1, np.nan)

        for j in range(nsamples_list_len - 1):

            acc[j] = (mca_test['perf'][j].mean() + mca_retest['perf'][j].mean()) / 2

            if k == 0:
                rel[j] = regress(mca_test[m][j], mca_retest[m][j], method='linregress').r
            else:
                mca_test_zb = mca_test[m][j] - min(mca_test[m][j].min(), mca_retest[m][j].min())
                mca_retest_zb = mca_retest[m][j] - min(mca_test[m][j].min(), mca_retest[m][j].min())
                rel[j] = np.mean(np.abs(mca_test_zb - mca_retest_zb)) / (0.5*(np.mean(np.abs(mca_test_zb - np.mean(mca_retest_zb))) + np.mean(np.abs(mca_retest_zb - np.mean(mca_test_zb)))))

            rel_bs = np.full(nresample, np.nan)
            acc_bs = np.full(nresample, np.nan)
            for b in range(nresample):
                ind = np.random.choice(range(len(mca_test[m][j])), len(mca_test[m][j]), replace=True)
                if k == 0:
                    rel_bs[b] = regress(mca_test[m][j][ind], mca_retest[m][j][ind], method='linregress').r
                else:
                    rel_bs[b] = np.mean(np.abs(mca_test_zb[ind] - mca_retest_zb[ind])) / (0.5*(np.mean(np.abs(mca_test_zb[ind] - np.mean(mca_retest_zb[ind]))) + np.mean(np.abs(mca_retest_zb[ind] - np.mean(mca_test_zb[ind])))))
                acc_bs[b] = (mca_test['perf'][j][ind].mean() + mca_retest['perf'][j][ind].mean()) / 2
            rel_se[j] = np.std(rel_bs)
            acc_se[j] = np.std(acc_bs)

            plt.plot([nsamples_list[j + 1], nsamples_list[j + 1]], [0, 1.5], color=(0.6, 0.6, 0.6), lw=0.5)

        if k == 0:
            plt.fill_between(nsamples_list[1:]-50, np.minimum(1, rel + rel_se), np.maximum(0, rel - rel_se), fc=(color_r, color_nmae)[k], ec=(0.4, 0.4, 0.4), alpha=0.5)
            plt.plot(nsamples_list[1:]-50, rel, 'o-', color=(color_r, color_nmae)[k], lw=1.5, markersize=3)
        else:
            plt.fill_between(nsamples_list[1:]-50, rel + rel_se, rel - rel_se, fc=(color_r, color_nmae)[k], ec=(0.4, 0.4, 0.4), alpha=0.5)
            plt.plot(nsamples_list[1:]-50, rel, 'o-', color=(color_r, color_nmae)[k], lw=1.5, markersize=3)

        plt.xlim(0, 700)
        if i == 0:
            plt.ylabel(('Pearson $r$', 'NMAE')[k])
        else:
            plt.yticks([])
        if k == 0:
            plt.title(mapping[m])
            plt.xticks(xticks, [])
            plt.ylim([0, 1])
        else:
            plt.xticks(xticks)
            plt.xlabel('Number of trials')
            plt.ylim([0, 1.5])
        for v in np.arange(0.25, 1.5, 0.25):
            plt.plot([0, 700], [v, v], color=(0.6, 0.6, 0.6), lw=0.5)
        if k == 0:
            plt.text((-0.08, -0.35)[int(i in (0, 4))], 1.07, 'ABCD'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=16)


set_fontsize(label=13, tick=10, title=14)
plt.tight_layout()
plt.subplots_adjust(hspace=0.2, wspace=0.1)

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()