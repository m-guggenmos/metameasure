import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress, sem, spearmanr

from plot_util import set_fontsize, savefig

HOME = os.path.expanduser('~')

z_orig = pd.read_pickle('../data/mca_test_retest.pkl')
xlim = np.array([0.1, 3.1])

N = 400

z = z_orig[(z_orig.nsamples >= N) & (z_orig.d1 >= xlim[0]) & (z_orig.d1 <= xlim[1])].copy()
z['logmratio_test'] = np.log(np.maximum(0.1, z['mratio_test']))
z['mratio_bounded_test'] = np.minimum(1.6, np.maximum(0, z['mratio_test']))
z['mratio_logistic_test'] = 1 / (1 + np.exp(-(z.mratio_test-z.mratio_test.median())))
z['mratio_exclude_test'] = z['mratio_test']
z.loc[z.mratio_exclude_test < 0, 'mratio_exclude_test'] = np.nan
z.loc[z.mratio_exclude_test > 1.6, 'mratio_exclude_test'] = np.nan


colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*2

mapping = dict(
    auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$",
    mratio_hmeta=r"hierarchical $M_{ratio}$", mratio_hmetacorr=r"$M_{ratio}$ (hier.)",  logmratio=r"$\log\, M_{ratio}$",
    mratio_exclude=r"$M_{ratio}$ (excl.)", mratio_bounded=r"bounded $M_{ratio}$", mdiff=r"$M_{diff}$"
)

axes = [None]*6

plt.figure(figsize=(10, 7))
for i, m in enumerate(('mdiff', 'mratio', 'mratio_exclude', 'mratio_bounded', 'logmratio', 'mratio_hmeta')):

    axes[i] = plt.subplot(3, 2, i + 1)

    d1 = z.d1_retest.values
    mca = z[f'{m}_test'].values
    d1 = d1[~np.isnan(mca)]
    mca = mca[~np.isnan(mca)]

    hws = 0.2
    ds = np.arange(xlim[0]+hws, xlim[1]-hws+1e-3, hws)
    mca_mean = np.full(len(ds), np.nan)
    mca_sem = np.full(len(ds), np.nan)
    for di in range(len(ds)):
        ind = np.where((d1 >= ds[di]-hws) & (d1 < ds[di]+hws))
        mca_mean[di] = np.mean(mca[ind])
        mca_sem[di] = sem(mca[ind])

    lr_all = linregress(d1, mca)
    rs, ps = spearmanr(d1[d1 > 0.5], mca[d1 > 0.5])
    text_pearson = f"$r_P$ = {lr_all.rvalue:.3f} (p = {lr_all.pvalue:{('.3f', '.1E')[int(lr_all.pvalue < 0.0005)]}})" \
                   + f"\n$r_S$ = {rs:.3f} (p = {ps:{('.3f', '.1E')[int(ps < 0.0005)]}})"
    text_spearman = f"$r_S$ = {rs:.3f} (p = {ps:{('.3f', '.1E')[int(ps < 0.0005)]}})"
    lh_pearson = plt.plot([0.5, xlim[1]], lr_all.slope * np.array([0.5, xlim[1]]) + lr_all.intercept, color=(0, 0.6, 0),
                          label=text_pearson, zorder=10)

    lh_scatter = plt.scatter(d1, mca, 3, marker='o', c=[[0.85, 0.85, 0.85]], edgecolors=[(0.7, 0.7, 0.7)],
                             linewidths=0.5, zorder=-1, label='individual subjects')
    lh_bins = plt.errorbar(ds, mca_mean, yerr=mca_sem, ls='-', lw=1.5, color='k')

    plt.ylabel(mapping[m])
    plt.text(0.035, 0.92, mapping[m], transform=axes[i].transAxes, fontsize=13,  # noqa
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='k', lw=0.4))
    if i >= 4:
        plt.xlabel("Type 1 performance d' [proportion correct]")
        axes[i].xaxis.set_label_coords(0.5, -0.3)  # noqa
        plt.xticks(np.arange(0.1, 3.11, 0.5), ['0.1', '0.6', '1.1', '1.6', '2.1', '2.6', '3.1'])
        for k, p in enumerate(['[.52]', '[.62]', '[.71]', '[.79]', '[.85]', '[.90]', '[.94]']):
            plt.text(k*0.5/3, -0.2, p, transform=axes[i].transAxes, fontsize=9, ha='center')  # noqa
    else:
        plt.xticks(np.arange(0.1, 3.11, 0.5), [])
    plt.xlim(xlim)

    ind = np.argsort(np.abs(mca - np.median(mca)))[:int(np.round(0.95*len(mca)))]
    if m in ('mratio_con2', 'mratio_bounded'):
        plt.ylim((-0.1, 2))
    else:
        plt.ylim(mca[ind].min(), mca[ind].max())
    plt.text((-0.16, -0.15)[int(np.mod(i + 1, 2) == 0)], 0.95, 'ABCDEF'[i], transform=plt.gca().transAxes,
             color=(0, 0, 0), fontsize=15)

    leg = plt.legend([lh_pearson[0]], [text_pearson], loc='upper right', fontsize=9, labelspacing=0.5, framealpha=1)
    plt.gca().add_artist(leg)
    set_fontsize(label=12, tick=11, title=15)

    plt.gca().yaxis.set_label_coords(-0.10, 0.5)

plt.tight_layout()
plt.subplots_adjust(hspace=0.16, wspace=0.18)
savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
