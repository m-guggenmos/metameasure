import os
import sys
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_util import set_fontsize, savefig

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.regression import linear_regression, regress  # noqa

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

nresample = 1000

reload = False

color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

if reload:

    measures = ('perf', 'd1', 'metad1', 'auc', 'mdiff', 'mratio', 'mratio_bounded', 'mratio_logistic', 'logmratio', 'mratio_hmeta')
    for i, m in enumerate(measures):
        print(f'{m} ({i + 1} / {len(measures)})')

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
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$",
    mratio_bounded=r"bounded $M_{ratio}$", mratio_logistic=r"logistic $M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$",
    mdiff=r"$M_{diff}$",
    mratio_bounded_minus_mratio=r"bounded $M_{ratio} - M_{ratio}$",
    logmratio_minus_mratio=r"$\log M_{ratio} - M_{ratio}$",
    mratio_logistic_minus_mratio=r"logistic $M_{ratio} - M_{ratio}$",
    mratio_hmeta_minus_mratio=r"hierarchical $M_{ratio} - M_{ratio}$",
    mratio_bounded_minus_logmratio=r"bnd. $M_{ratio} - \log M_{ratio}$"
)
df = pd.DataFrame()

xlim = [0, 700]
xticks = np.arange(0, 701, 100)

color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

fig = plt.figure(figsize=(10, 5.5))
ax = [None] * 18
order = [1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 12, 17, 13, 18, 14, 19, 15, 20]

rows = []

for i, m in enumerate(('mratio', 'mratio_bounded', 'logmratio', 'mratio_logistic', 'mratio_hmeta', 'mratio_bounded_minus_mratio', 'logmratio_minus_mratio', 'mratio_logistic_minus_mratio', 'mratio_hmeta_minus_mratio')):
    for k in range(2):

        if k == 0:
            test, retest = mca_test_cor, mca_retest_cor
        else:
            test, retest = mca_test, mca_retest

        ax[i*2 + k] = plt.subplot(4, 5, order[i*2 + k])

        rel = np.full(nsamples_list_len - 1, np.nan)
        rel_se = np.full(nsamples_list_len - 1, np.nan)
        acc = np.full(nsamples_list_len - 1, np.nan)
        acc_se = np.full(nsamples_list_len - 1, np.nan)

        if 'minus' in m:
            m1, m2 = m.split('_minus_')

        for j in range(nsamples_list_len - 1):

            if 'minus' in m:
                if k == 0:
                    rel[j] = regress(test[m1][j], retest[m1][j], method='bivariate').r - regress(test[m2][j], retest[m2][j], method='bivariate').r
                else:
                    test_zb1 = test[m1][j] - min(test[m1][j].min(), retest[m1][j].min())
                    retest_zb1 = retest[m1][j] - min(test[m1][j].min(), retest[m1][j].min())
                    test_zb2 = test[m2][j] - min(test[m2][j].min(), retest[m2][j].min())
                    retest_zb2 = retest[m2][j] - min(test[m2][j].min(), retest[m2][j].min())
                    rel[j] = np.mean(np.abs(test_zb1 - retest_zb1)) / (0.5*(np.mean(np.abs(test_zb1 - np.mean(retest_zb1))) + np.mean(np.abs(retest_zb1 - np.mean(test_zb1))))) - \
                                np.mean(np.abs(test_zb2 - retest_zb2)) / (0.5*(np.mean(np.abs(test_zb2 - np.mean(retest_zb2))) + np.mean(np.abs(retest_zb2 - np.mean(test_zb2)))))
            else:
                acc[j] = (test['perf'][j].mean() + retest['perf'][j].mean()) / 2
                if k == 0:
                    rel[j] = regress(test[m][j], retest[m][j], method='bivariate').r
                else:
                    test_zb = test[m][j] - min(test[m][j].min(), retest[m][j].min())
                    retest_zb = retest[m][j] - min(test[m][j].min(), retest[m][j].min())
                    rel[j] = np.mean(np.abs(test_zb - retest_zb)) / (0.5*(np.mean(np.abs(test_zb - np.mean(retest_zb))) + np.mean(np.abs(retest_zb - np.mean(test_zb)))))


            rel_bs = np.full(nresample, np.nan)
            acc_bs = np.full(nresample, np.nan)
            for b in range(nresample):
                if 'minus' in m:
                    ind = np.random.choice(range(len(test[m1][j])), len(test[m1][j]), replace=True)
                else:
                    ind = np.random.choice(range(len(test[m][j])), len(test[m][j]), replace=True)
                acc_bs[b] = (test['perf'][j][ind].mean() + retest['perf'][j][ind].mean()) / 2

                if 'minus' in m:
                    if k == 0:
                        rel_bs[b] = regress(test[m1][j][ind], retest[m1][j][ind], method='bivariate').r - regress(test[m2][j][ind], retest[m2][j][ind], method='bivariate').r
                    else:
                        rel_bs[b] = np.mean(np.abs(test_zb1[ind] - retest_zb1[ind])) / (0.5*(np.mean(np.abs(test_zb1[ind] - np.mean(retest_zb1[ind]))) + np.mean(np.abs(retest_zb1[ind] - np.mean(test_zb1[ind]))))) - \
                                        np.mean(np.abs(test_zb2[ind] - retest_zb2[ind])) / (0.5*(np.mean(np.abs(test_zb2[ind] - np.mean(retest_zb2[ind]))) + np.mean(np.abs(retest_zb2[ind] - np.mean(test_zb2[ind])))))
                else:
                    if k == 0:
                        rel_bs[b] = regress(test[m][j][ind], retest[m][j][ind], method='bivariate').r
                    else:
                        rel_bs[b] = np.mean(np.abs(test_zb[ind] - retest_zb[ind])) / (0.5*(np.mean(np.abs(test_zb[ind] - np.mean(retest_zb[ind]))) + np.mean(np.abs(retest_zb[ind] - np.mean(test_zb[ind])))))
            rel_se[j] = np.std(rel_bs)
            acc_se[j] = np.std(acc_bs)

            plt.plot([nsamples_list[j + 1], nsamples_list[j + 1]], [-1.5, 1.6], color=(0.6, 0.6, 0.6), lw=0.5)

        if not ((k == 1) and 'hmeta' in m):
            if ('minus' in m) or k == 1:
                plt.fill_between(nsamples_list[1:]-50, rel + rel_se, rel - rel_se, fc=(color_r, color_nmae)[k], ec=(0.4, 0.4, 0.4), alpha=0.5)
            else:
                plt.fill_between(nsamples_list[1:]-50, np.minimum(1, rel + rel_se), np.maximum(0, rel - rel_se), fc=(color_r, color_nmae)[k], ec=(0.4, 0.4, 0.4), alpha=0.5)
            plt.plot(nsamples_list[1:]-50, rel, 'o-', color=(color_r, color_nmae)[k], lw=1.5, markersize=3)
        if (i == 0) & (k == 0):
            plt.fill_between(nsamples_list[1:]-50, acc + acc_se, acc - acc_se, fc='k', ec=(0.4, 0.4, 0.4), alpha=0.5)
            lh_acc = plt.plot(nsamples_list[1:]-50, acc, 'o-', color='k', lw=1, label='Type 1 performance\n(Proportion correct)', markersize=3)
        if (i == 0) & (k == 1):
            leg = plt.legend([lh_acc[0]], [lh_acc[0].get_label()], loc='upper left', handlelength=0.8, fontsize=10, bbox_to_anchor=(-0.4, -0.5), borderpad=0.4)
            ax[i*2 + k].add_artist(leg)
        if k == 0:
            title = plt.title(mapping[m])
            if i == 8:
                title.set_position((0.515, 1))

        if (i > 4) & (k == 1):
            plt.xticks(xticks)
            plt.xlabel('Number of trials')
        else:
            plt.xticks(xticks, [])
        if (i in (0, 5)) and (k == 0):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'Pearson $r$')
        elif (i in (0, 5)) and (k == 1):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'NMAE')
        else:
            plt.yticks(plt.gca().get_yticks(), [])


        if 'minus' in m:
            plt.plot(xlim, [0, 0], color='k', lw=0.5)

        if k == 0:
            plt.text((-0.08, -0.35)[int(i in [0, 5])] - 0.05*(i==5) - 0.02*(i==8), 1.1, 'ABCDEFGHIJ'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
            if 'minus' in m:
                if i in (0, 5):
                    plt.yticks([-0.5, 0, 0.5])
                else:
                    plt.yticks([-0.5, 0, 0.5], [])
                plt.ylim((-0.52, 0.5))
            else:
                if i in (0, 5):
                    plt.yticks([0, 0.5, 1])
                else:
                    plt.yticks([0, 0.5, 1], [])
                plt.ylim((0, 1))
        else:
            if 'minus' in m:
                if i in (0, 5):
                    plt.yticks([-0.4, -0.2, 0, 0.2])
                else:
                    plt.yticks([-0.4, -0.2, 0, 0.2], [])
                plt.ylim((-0.45, 0.12))
            else:
                if i in (0, 5):
                    plt.yticks([0.5, 1, 1.5])
                else:
                    plt.yticks([0.5, 1, 1.5], [])
                plt.ylim((0.35, 1.6))

        plt.xlim(xlim)


set_fontsize(label=11, tick=9, title=11)
plt.tight_layout()
plt.subplots_adjust(hspace=0.25, wspace=0.1, top=0.94, bottom=0.09, right=0.987, left=0.064)

for i in (1, 3, 5, 7, 9):
    ax[i].set_position([*(np.array(ax[i]._position)[0] + (0, 0.028)), ax[i]._position.width, ax[i]._position.height])

for i in (10, 12, 14, 16):
    ax[i].set_position([*(np.array(ax[i]._position)[0] - (0, 0.028)), ax[i]._position.width, ax[i]._position.height])

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()