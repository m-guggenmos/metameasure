import os
import sys
import pickle
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import sem
import pandas as pd

from plot_util import set_fontsize, savefig

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.regression import linear_regression, regress  # noqa

HOME = os.path.expanduser('~')


z_orig = pd.read_pickle('../data/mca_test_retest.pkl')

for k, sid in enumerate(sorted(z_orig.study_id.unique())):
    z_orig.loc[z_orig.study_id == sid, 'study_nsamples_test'] = z_orig[z_orig.study_id == sid].nsamples_test.mean()
    z_orig.loc[z_orig.study_id == sid, 'study_nsamples_test_std'] = z_orig[z_orig.study_id == sid].nsamples_test.std()

stepsize = 200
# nsamples_list = np.arange(0, 701, 100)
nsamples_list = np.arange(0, 1001, stepsize)
nsamples_list_len = len(nsamples_list)
mca_test, mca_retest = dict(), dict()
mca_test_cor, mca_retest_cor = dict(), dict()


for i in range(1, len(nsamples_list)):
    inc = (z_orig.study_nsamples_test >= nsamples_list[i - 1]) & (z_orig.study_nsamples_test < nsamples_list[i])
    drop = (z_orig.nsamples_test < nsamples_list[i - 1]) | (z_orig.nsamples_test >= nsamples_list[i])
    print(f'{nsamples_list[i-1]}-{nsamples_list[i]}: drop {(inc & drop).sum()} / {inc.sum()}')  # noqa
    z_orig = z_orig.drop(z_orig[inc & drop].index)

nresample = 1000

reload = True

color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

if reload:

    measures = ('perf', 'd1', 'metad1', 'auc', 'mdiff')
    for i, m in enumerate(measures):
        print(f'{m} ({i + 1} / {len(measures)})')

        mca_test[m], mca_retest[m] = [None] * nsamples_list_len, [None] * nsamples_list_len
        mca_test_cor[m], mca_retest_cor[m] = [None] * nsamples_list_len, [None] * nsamples_list_len
        for j in range(1, nsamples_list_len):
            z = z_orig[(z_orig.study_nsamples_test >= nsamples_list[j - 1]) &
                       (z_orig.study_nsamples_test < nsamples_list[j])].copy()
            if m == 'mratio_hmetacorr':
                mca_origmean = (z[f'{m}_test'].mean() + z[f'{m}_retest'].mean()) / 2
            else:
                mca_origmean = z[m].mean()
            nstudies = len(z.study_id.unique())
            mca_test[m][j - 1], mca_retest[m][j - 1] = [None] * nstudies, [None] * nstudies
            mca_test_cor[m][j - 1], mca_retest_cor[m][j - 1] = [None] * nstudies, [None] * nstudies
            for k, sid in enumerate(sorted(z.study_id.unique())):
                cond = (z.study_id == sid) & ~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna()
                z.loc[cond, f'{m}_test_cor'] = z.loc[cond, f'{m}_test']
                z.loc[cond, f'{m}_retest_cor'] = z.loc[cond, f'{m}_retest']
                z.loc[cond, f'{m}_test_cor'] -= (np.hstack((z.loc[cond, f'{m}_test'].values, z.loc[cond, f'{m}_retest'].values)).mean() - mca_origmean)  # noqa
                z.loc[cond, f'{m}_retest_cor'] -= (np.hstack((z.loc[cond, f'{m}_test'].values, z.loc[cond, f'{m}_retest'].values)).mean() - mca_origmean)  # noqa

                mca_test[m][j - 1][k] = z.loc[cond, f'{m}_test'].values  # noqa
                mca_retest[m][j - 1][k] = z.loc[cond, f'{m}_retest'].values  # noqa
                mca_test_cor[m][j - 1][k] = z.loc[cond, f'{m}_test_cor'].values  # noqa
                mca_retest_cor[m][j - 1][k] = z.loc[cond, f'{m}_retest_cor'].values  # noqa

    pickle.dump((mca_test, mca_retest, mca_test_cor, mca_retest_cor), open(f'../data/{Path(__file__).stem}.pkl', 'wb'))


mca_test, mca_retest, mca_test_cor, mca_retest_cor = pickle.load(open(f'../data/{Path(__file__).stem}.pkl', 'rb'))


mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$",
    mratio=r"$M_{ratio}$", mratio_exclude=r"$M_{ratio}$ (excl.)", mratio_bounded=r"bounded $M_{ratio}$",
    mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$", mdiff=r"$M_{diff}$",
    mratio_bounded_minus_mratio=r"bounded $M_{ratio} - M_{ratio}$",
    logmratio_minus_mratio=r"$\log M_{ratio} - M_{ratio}$",
    mratio_exclude_minus_mratio=r"$M_{ratio}$ (excl.)$ - M_{ratio}$",
    mratio_hmeta_minus_mratio=r"hierarchical $M_{ratio} - M_{ratio}$",
    mratio_bounded_minus_logmratio=r"bnd. $M_{ratio} - \log M_{ratio}$"
)
df = pd.DataFrame()

xticks = nsamples_list
xlim = [0, xticks[-1]]


color_r = (0, 0.2, 0.6)
color_nmae = (0, 1/3, 0)

plt.figure(figsize=(10, 3.5))
ax = [None] * 8
order = [1, 5, 2, 6, 3, 7, 4, 8]

rows = []

measures = ['d1', 'metad1', 'auc', 'mdiff']

for i, m in enumerate(measures):
    for k in range(2):

        if k == 0:
            test, retest = mca_test_cor, mca_retest_cor
        else:
            test, retest = mca_test, mca_retest

        ax[i*2 + k] = plt.subplot(2, 4, order[i*2 + k])

        rel = np.full(nsamples_list_len - 1, np.nan)
        rel_se = np.full(nsamples_list_len - 1, np.nan)

        for j in range(nsamples_list_len - 1):

            if k == 0:
                rel[j] = np.tanh(np.nanmean([np.arctanh(regress(test_, retest_, method='bivariate').r) if len(test_) > 1
                                             else np.nan for test_, retest_ in zip(test[m][j], retest[m][j])]))
                rel_studies = np.array([regress(test_, retest_, method='bivariate').r if len(test_) > 1 else np.nan for
                                        test_, retest_ in zip(test[m][j], retest[m][j])])
                if len(rel_studies) > 1:
                    rel_se[j] = sem(rel_studies[~np.isnan(rel_studies)])
            else:
                test_zb = [test_ - min(test_.min(), retest_.min()) for test_, retest_ in zip(test[m][j], retest[m][j])]
                retest_zb = [retest_ - min(test_.min(), retest_.min()) for test_, retest_ in zip(test[m][j], retest[m][j])]  # noqa
                rel_studies = [np.mean(np.abs(test_zb_ - retest_zb_)) /
                               (0.5*(np.mean(np.abs(test_zb_ - np.mean(retest_zb_))) + np.mean(np.abs(retest_zb_ - np.mean(test_zb_)))))  # noqa
                               for test_zb_, retest_zb_ in zip(test_zb, retest_zb)]
                rel[j] = np.mean(rel_studies)
                if len(rel_studies) > 1:
                    rel_se[j] = sem(rel_studies)

            plt.plot([nsamples_list[j + 1], nsamples_list[j + 1]], [-1.5, 1.6], color=(0.7, 0.7, 0.7), lw=0.5,
                     zorder=-10)

            plt.fill_between(nsamples_list[1:]-stepsize/2, rel + rel_se, rel - rel_se,
                             fc=(np.array([127, 152, 203])/255, np.array([127, 169, 127])/255)[k], ec=(0.4, 0.4, 0.4),
                             clip_on=False)
            plt.plot(nsamples_list[1:]-stepsize/2, rel, 'o-', color=(color_r, color_nmae)[k], lw=1.5, markersize=3,
                     clip_on=False)

        if k == 0:
            title = plt.title(mapping[m])

        if k == 1:
            plt.xticks(xticks, [0, 200, 400, 600, 800, '1k'])
            plt.xlabel('Number of trials')
        else:
            plt.xticks(xticks, [])

        if (i == 0) and (k == 0):
            plt.ylabel('Pearson $r$')
        elif (i == 0) and (k == 1):
            plt.ylabel('NMAE')
        else:
            plt.yticks(plt.gca().get_yticks(), [])

        if k == 0:
            plt.text((-0.08, -0.25)[int(i in [0, 5])] - 0.05*(i == 5) - 0.02*(i == 8), 1.05, 'ABCD'[i],
                     transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)

            yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
            if i in (0, 5):
                plt.yticks(yticks)
            else:
                plt.yticks(yticks, [])
            plt.ylim((0, 1))
        else:
            yticks = [0, 0.4, 0.8, 1.2, 1.6]
            if i in (0, 5):
                plt.yticks(yticks)
            else:
                plt.yticks(yticks, [])
            plt.ylim((0, 1.63))
        for ytick in yticks:
            plt.plot(xlim, [ytick, ytick], color=(0.7, 0.7, 0.7), lw=0.5, zorder=-10)

        plt.xlim(xlim)


set_fontsize(label=11, tick=9, title=11)
plt.tight_layout()
plt.subplots_adjust(hspace=0.15, wspace=0.1, top=0.9, bottom=0.13, right=0.987, left=0.064)

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
