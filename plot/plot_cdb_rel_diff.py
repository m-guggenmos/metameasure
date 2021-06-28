import os
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence.analysis.regression import regress
from plot_util import set_fontsize, savefig


HOME = os.path.expanduser('~')


z_orig = pd.read_pickle('../data/mca_test_retest.pkl')
z_orig = z_orig[~z_orig.mratio.isna()]
z_orig['logmratio'] = np.log(np.maximum(0.1, z_orig['mratio'].values))
z_orig['logmratio_test'] = np.log(np.maximum(0.1, z_orig['mratio_test'].values))
z_orig['logmratio_retest'] = np.log(np.maximum(0.1, z_orig['mratio_retest'].values))
z_orig['mratio_bounded'] = np.minimum(1.6, np.maximum(0, z_orig['mratio'].values))
z_orig['mratio_bounded_test'] = np.minimum(1.6, np.maximum(0, z_orig['mratio_test'].values))
z_orig['mratio_bounded_retest'] = np.minimum(1.6, np.maximum(0, z_orig['mratio_retest'].values))
z_orig['mratio_logistic'] = 1 / (1 + np.exp(-(z_orig.mratio-0.8)))
z_orig['mratio_logistic_test'] = 1 / (1 + np.exp(-(z_orig.mratio_test-0.8)))
z_orig['mratio_logistic_retest'] = 1 / (1 + np.exp(-(z_orig.mratio_retest-0.8)))


nsamples_list = np.arange(0, 701, 100)
nsamples_list_len = len(nsamples_list)
mca_test, mca_retest = dict(), dict()
mca_test_cor, mca_retest_cor = dict(), dict()

nresample = 10000

reload = False

color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

if reload:

    measures = ('perf', 'd1', 'metad1', 'auc', 'mdiff', 'mratio', 'mratio_bounded', 'mratio_logistic', 'logmratio', 'mratio_hmeta')
    for i, m in enumerate(measures):
        print(f'mca: {m} ({i + 1} / {len(measures)})')

        mca_test[m], mca_retest[m] = [None] * nsamples_list_len, [None] * nsamples_list_len
        mca_test_cor[m], mca_retest_cor[m] = [None] * nsamples_list_len, [None] * nsamples_list_len
        for j in range(1, nsamples_list_len):
            z = z_orig[(z_orig.nsamples_test >= nsamples_list[j - 1]) & (z_orig.nsamples_retest < nsamples_list[j])].copy()
            if m == 'mratio_hmetacorr':
                mca_origmean = (z[f'{m}_test'].mean() + z[f'{m}_retest'].mean()) / 2
            else:
                mca_origmean = z[m].mean()
            for k, study in enumerate(sorted(z.study_id.unique())):
                cond = (z.study_id == study) & ~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna()
                z.loc[cond, f'{m}_test_cor'] = z.loc[cond, f'{m}_test']
                z.loc[cond, f'{m}_retest_cor'] = z.loc[cond, f'{m}_retest']
                z.loc[cond, f'{m}_test_cor'] -= (np.hstack((z.loc[cond, f'{m}_test'].values, z.loc[cond, f'{m}_retest'].values)).mean() - mca_origmean)
                z.loc[cond, f'{m}_retest_cor'] -= (np.hstack((z.loc[cond, f'{m}_test'].values, z.loc[cond, f'{m}_retest'].values)).mean() - mca_origmean)

            mca_test[m][j - 1] = z.loc[~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna(), f'{m}_test'].values
            mca_retest[m][j - 1] = z.loc[~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna(), f'{m}_retest'].values
            mca_test_cor[m][j - 1] = z.loc[~z[f'{m}_test_cor'].isna() & ~z[f'{m}_retest_cor'].isna(), f'{m}_test_cor'].values
            mca_retest_cor[m][j - 1] = z.loc[~z[f'{m}_test_cor'].isna() & ~z[f'{m}_retest_cor'].isna(), f'{m}_retest_cor'].values

    pickle.dump((mca_test, mca_retest, mca_test_cor, mca_retest_cor), open(f'../data/{Path(__file__).stem}.pkl', 'wb'))

mca_test, mca_retest, mca_test_cor, mca_retest_cor = pickle.load(open(f'../data/{Path(__file__).stem}.pkl', 'rb'))


mapping = dict(
    mratio_bounded_minus_logmratio=r"bounded $M_{ratio} - \log\, M_{ratio}$",
    mratio_bounded_minus_mratio_logistic=r"bounded $M_{ratio} -$ logistic $M_{ratio}$",
    mratio_bounded_minus_mratio_logistic2=r"bounded $M_{ratio} -$ logistic $M_{ratio}$",
    logmratio_minus_mratio_logistic=r"$\log\, M_{ratio} -$ logistic $M_{ratio}$",
    mratio_logistic_minus_logmratio=r"logistic $M_{ratio} - \log\, M_{ratio}$",
    logmratio_minus_mratio_logistic2=r"$\log\, M_{ratio} -$ logistic $M_{ratio}$",
    mratio_hmeta_minus_mratio_bounded=r"hierarchical $M_{ratio} -$ bounded $M_{ratio}$",
    mratio_hmeta_minus_logmratio=r"hierarchical $M_{ratio} - \log\, M_{ratio}$",
    mratio_hmeta_minus_mratio_logistic=r"hierarchical $M_{ratio} -$ logistic $M_{ratio}$",
    mratio_hmeta_minus_mratio_logistic2=r"hierarchical $M_{ratio} -$ logistic $M_{ratio}$"
)
df = pd.DataFrame()

xlim = [0, 700]
xticks = np.arange(0, 701, 100)

color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

fig = plt.figure(figsize=(8, 5.5))
ax = [None] * 12
order = [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]

for i, m in enumerate(('mratio_bounded_minus_logmratio', 'mratio_logistic_minus_logmratio', 'mratio_bounded_minus_mratio_logistic', 'mratio_hmeta_minus_mratio_bounded', 'mratio_hmeta_minus_logmratio', 'mratio_hmeta_minus_mratio_logistic')):
    for k in range(2):

        if k == 0:
            test, retest = mca_test_cor, mca_retest_cor
        else:
            test, retest = mca_test, mca_retest

        ax[i*2 + k] = plt.subplot(4, 3, order[i*2 + k])

        rel = np.full(nsamples_list_len - 1, np.nan)
        rel_se = np.full(nsamples_list_len - 1, np.nan)
        acc = np.full(nsamples_list_len - 1, np.nan)
        acc_se = np.full(nsamples_list_len - 1, np.nan)

        if 'minus' in m:
            m1, m2 = m.split('_minus_')

        for j in range(nsamples_list_len - 1):

            if k == 0:
                rel[j] = regress(test[m1][j], retest[m1][j], method='linregress').r - regress(test[m2][j], retest[m2][j], method='linregress').r
            else:
                test_zb1 = test[m1][j] - min(test[m1][j].min(), retest[m1][j].min())
                retest_zb1 = retest[m1][j] - min(test[m1][j].min(), retest[m1][j].min())
                test_zb2 = test[m2][j] - min(test[m2][j].min(), retest[m2][j].min())
                retest_zb2 = retest[m2][j] - min(test[m2][j].min(), retest[m2][j].min())
                rel[j] = np.mean(np.abs(test_zb1 - retest_zb1)) / (0.5*(np.mean(np.abs(test_zb1 - np.mean(retest_zb1))) + np.mean(np.abs(retest_zb1 - np.mean(test_zb1))))) - \
                            np.mean(np.abs(test_zb2 - retest_zb2)) / (0.5*(np.mean(np.abs(test_zb2 - np.mean(retest_zb2))) + np.mean(np.abs(retest_zb2 - np.mean(test_zb2)))))

            rel_bs = np.full(nresample, np.nan)
            acc_bs = np.full(nresample, np.nan)
            for b in range(nresample):
                ind = np.random.choice(range(len(test[m1][j])), len(test[m1][j]), replace=True)
                if k == 0:
                    rel_bs[b] = regress(test[m1][j][ind], retest[m1][j][ind], method='linregress').r - regress(test[m2][j][ind], retest[m2][j][ind], method='linregress').r
                else:
                    rel_bs[b] = np.mean(np.abs(test_zb1[ind] - retest_zb1[ind])) / (0.5*(np.mean(np.abs(test_zb1[ind] - np.mean(retest_zb1[ind]))) + np.mean(np.abs(retest_zb1[ind] - np.mean(test_zb1[ind]))))) - \
                                    np.mean(np.abs(test_zb2[ind] - retest_zb2[ind])) / (0.5*(np.mean(np.abs(test_zb2[ind] - np.mean(retest_zb2[ind]))) + np.mean(np.abs(retest_zb2[ind] - np.mean(test_zb2[ind])))))
            rel_se[j] = np.std(rel_bs)

            if not ((k == 1) and 'hmeta' in m):
                plt.plot([nsamples_list[j + 1], nsamples_list[j + 1]], [-1.5, 1.6], color=(0.6, 0.6, 0.6), lw=0.5)

        if not ((k == 1) and 'hmeta' in m):
            if ('minus' in m) or k == 1:
                plt.fill_between(nsamples_list[1:]-50, rel + rel_se, rel - rel_se, fc=(color_r, color_nmae)[k], ec=(0.4, 0.4, 0.4), alpha=0.5)
            else:
                plt.fill_between(nsamples_list[1:]-50, np.minimum(1, rel + rel_se), np.maximum(0, rel - rel_se), fc=(color_r, color_nmae)[k], ec=(0.4, 0.4, 0.4), alpha=0.5)
            plt.plot(nsamples_list[1:]-50, rel, 'o-', color=(color_r, color_nmae)[k], lw=1.5, markersize=3)
        if k == 0:
            title = plt.title(mapping[m])
            if i == 3:
                title.set_position((0.42, 1))
            if i == 5:
                title.set_position((0.52, 1))

        if (i > 2) & (k == 0):
            plt.xticks(xticks)
            plt.xlabel('Number of trials')
        else:
            plt.xticks(xticks, [])
        if (i in (0, 3)) and (k == 0):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'Pearson $r$')
        elif (i in (0, 3)) and (k == 1):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'NMAE')
        else:
            plt.yticks(plt.gca().get_yticks(), [])

        if (i < 3) or (k == 0):
            plt.plot(xlim, [0, 0], color='k', lw=0.5)

        if k == 0:
            plt.text((-0.1, -0.3)[int(i in [0, 3])], 1.1, 'ABCDEF'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
            if i in (0, 3):
                plt.yticks([-0.2, -0.1, 0, 0.1])
            else:
                plt.yticks([-0.2, -0.1, 0, 0.1], [])
            plt.ylim((-0.2, 0.18))
        else:
            if i in (0, 3):
                plt.yticks([-0.3, -0.2, -0.1, 0, 0.1])
            else:
                plt.yticks([-0.3, -0.2, -0.1, 0, 0.1], [])
            plt.ylim((-0.3, 0.15))

        plt.xlim(xlim)

        if (i > 2) and (k == 1):
            ax[i*2 + k].axis('off')




set_fontsize(label=10, tick=9, title=10)
plt.tight_layout()
plt.subplots_adjust(hspace=0.25, wspace=0.1, top=0.94, bottom=0.09, right=0.97, left=0.09)

for i in (1, 3, 5):
    ax[i].set_position([*(np.array(ax[i]._position)[0] + (0, 0.028)), ax[i]._position.width, ax[i]._position.height])

for i in (6, 8, 10):
    ax[i].set_position([*(np.array(ax[i]._position)[0] - (0, 0.028)), ax[i]._position.width, ax[i]._position.height])

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()