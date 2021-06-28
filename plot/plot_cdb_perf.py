import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import linregress, sem

from plot_util import set_fontsize, savefig

HOME = os.path.expanduser('~')

z_orig = pd.read_pickle('../data/mca_test_retest.pkl')
xlim = np.array([0.1, 3.1])

N = 400

z = z_orig[(z_orig.nsamples >= N) & (z_orig.d1 >= xlim[0]) & (z_orig.d1 <= xlim[1])].copy()
z['logmratio'] = np.log(np.maximum(0.1, z['mratio']))
z['mratio_bounded'] = np.minimum(1.6, np.maximum(0, z['mratio']))
z['mratio_logistic'] = 1 / (1 + np.exp(-(z.mratio-z.mratio.median())))

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']*2

def weibull(x, k, l, cp, slope, yshift):
    y = np.full(len(x), np.nan)
    y[x <= cp] = 1 - np.exp(-(x[x <= cp]/l)**k)
    intercept = 1 - np.exp(-(cp/l)**k) - slope * cp
    y[x > cp] = slope * x[x > cp] + intercept
    return y + yshift

mapping = dict(auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$", mratio_logistic=r"logistic $M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$", mratio_hmetacorr=r"$M_{ratio}$ (hier.)", logmratio=r"$\log\, M_{ratio}$", mratio_con2=r"bounded $M_{ratio}$", mratio_bounded=r"bounded $M_{ratio}$", mdiff=r"$M_{diff}$")

axes = [None]*6

plt.figure(figsize=(10, 7))
for i, m in enumerate(('mdiff', 'mratio', 'mratio_bounded', 'logmratio', 'mratio_logistic', 'mratio_hmeta')):

    axes[i] = plt.subplot(3, 2, i + 1)

    d1 = z.d1.values
    mca = z[m].values
    d1 = d1[~np.isnan(mca)]
    mca = mca[~np.isnan(mca)]

    hws = 0.2
    ds = np.arange(xlim[0]+hws, xlim[1]-hws+1e-3, hws)
    mca_mean = np.full(len(ds), np.nan)
    mca_sem = np.full(len(ds), np.nan)
    for di in range(len(ds)):
        ind = np.where((d1>=ds[di]-hws) & (d1<ds[di]+hws))
        mca_mean[di] = np.mean(mca[ind])
        mca_sem[di] = sem(mca[ind])

    lr_all = linregress(d1, mca)
    text_all = f"r = {lr_all.rvalue:.3f} (p = {lr_all.pvalue:{('.3f', '.1E')[int(lr_all.pvalue < 0.0005)]}})"
    lh_linear = plt.plot(xlim, lr_all.slope*xlim+lr_all.intercept, color=(0, 0.6, 0), label=text_all)
    if m == 'logmratio':
        lr_log = linregress(d1[d1>1], mca[d1>1])
        text_log = f"r = {lr_log.rvalue:.3f} (p = {lr_log.pvalue:{('.3f', '.1E')[int(lr_log.pvalue < 0.0005)]}})"
        lh_log = plt.plot([1, xlim[1]], lr_log.slope*np.array([1, xlim[1]])+lr_log.intercept, color=(0.8, 0.2, 0.2), label=text_log, zorder=10)

    lh_scatter = plt.scatter(z.d1.values, z[m].values, 3, marker='o', c=[[0.85, 0.85, 0.85]], edgecolors=[(0.7, 0.7, 0.7)], linewidths=0.5, zorder=-1, label='individual subjects')
    lh_bins = plt.errorbar(ds, mca_mean, yerr=mca_sem, ls='-', lw=1.5, color='k')

    plt.ylabel(mapping[m])
    plt.text(0.035, 0.92, mapping[m], transform=axes[i].transAxes, fontsize=13, bbox=dict(facecolor='white', alpha=0.95, edgecolor='k', lw=0.4))
    if i >= 4:
        plt.xlabel("Type 1 performance d' [proportion correct]")
        # plt.xticks(np.arange(0.1, 3.11, 0.5), ['0.1\n[.52]', '0.6\n[.62]', '1.1\n[.71]', '1.6\n[.79]', '2.1\n[.85]', '2.6\n[.90]', '3.1\n[.94]'])
        plt.xticks(np.arange(0.1, 3.11, 0.5), ['0.1', '0.6', '1.1', '1.6', '2.1', '2.6', '3.1'])
    else:
        plt.xticks(np.arange(0.1, 3.11, 0.5), [])
    plt.xlim(xlim)

    ind = np.argsort(np.abs(z[m].values - np.median(z[m].values)))[:int(np.round(0.95*len(z[m].values)))]
    if m in ('mratio_con2', 'mratio_bounded'):
        plt.ylim((-0.1, 2))
    else:
        plt.ylim(z[m].values[ind].min(), z[m].values[ind].max())
    plt.text((-0.16, -0.15)[int(np.mod(i + 1, 2) == 0)], 0.95, 'ABCDEF'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=15)

    if m == 'logmratio':
        leg = plt.legend([lh_linear[0], lh_log[0]], [text_all, text_log], loc='upper right', fontsize=9, labelspacing=0.5, framealpha=1)
        plt.gca().add_artist(leg)
    else:
        leg = plt.legend([lh_linear[0]], [text_all], loc='upper right', fontsize=9, labelspacing=0.5, framealpha=1)
    plt.gca().add_artist(leg)
    set_fontsize(label=12, tick=11, title=15)

    plt.gca().yaxis.set_label_coords(-0.10, 0.5)

plt.tight_layout()
plt.subplots_adjust(hspace=0.16, wspace=0.18)
savefig(f'../img/{Path(__file__).stem}.png')
plt.show()