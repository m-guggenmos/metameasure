from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm, skew, shapiro

from plot_util import set_fontsize, savefig

z_orig = pd.read_pickle('../data/mca_test_retest.pkl')
z = z_orig[(z_orig.nsamples >= 400) & (z_orig.d1 >= 0.5)].copy()

z['mratio_exclude'] = z['mratio']
z.loc[(z.mratio_exclude < 0) | (z.mratio_exclude > 1.6), 'mratio_exclude'] = np.nan

color_hist = (0.58, 0.71, 0.88)
ylim = (0, 205)

xlims = dict(
    d1=(-2.2, 5),
    auc=(z.auc.min(), 1),
    metad1=(z.metad1.min(), z.metad1.max()),
    mdiff=(-5, 2),
    mratio=(-1.5, 2.75),
    mratio_exclude=(-1.5, 2.75),
    mratio_bounded=(-1.5, 2.75),
    logmratio=(-2.5, 1.2),
    mratio_logistic=(0, 1),
    mratio_hmeta=(-1.5, 2.75),
)

mapping = dict(
    d1=r"$d'$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$",
    mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$",
    mratio_exclude=r"$M_{ratio}$ (excl.)",
    mratio_logistic=r"logistic $M_{ratio}$",
    mratio_bounded=r"bounded $M_{ratio}$",
    mdiff=r"$M_{diff}$"
)

plt.figure(figsize=(8, 5.5))

measures = ('d1', 'auc', 'metad1', 'mdiff', 'mratio', 'mratio_exclude', 'mratio_bounded', 'logmratio', 'mratio_hmeta')

for i, m in enumerate(measures):

    ax = plt.subplot(3, 3, i + 1)
    if m in ('d1', 'auc', 'metad1', 'mdiff', 'mratio', 'mratio_exclude', 'mratio_hmeta'):
        data = z[~z[m].isna()][m].values
    elif m == 'mratio_bounded':
        data = np.minimum(1.6, np.maximum(0, z.mratio))
    elif m == 'logmratio':
        data = np.log(np.maximum(0.1, z.mratio))
    elif m == 'mratio_logistic':
        data = (1 / (1 + np.exp(-1*(z.mratio-z.mratio.median()))))

    xlim = xlims[m]
    bins = np.linspace(xlim[0]-1e-8, (1.6 if m in ('mratio_exclude', 'mratio_bounded') else xlim[1])+1e-8, 75)
    if m in ('mratio_exclude', 'mratio_bounded'):
        bins = np.linspace(xlims['mratio'][0]-1e-8, xlims['mratio'][1]+1e-8, 75)
    else:
        bins = np.linspace(xlim[0]-1e-8, xlim[1]+1e-8, 75)

    if m in ('mdiff', 'mratio', 'mratio_bounded', 'mratio_exclude'):
        ws = bins[1] - bins[0]
        vals, bins = np.histogram(data, bins=np.arange(data.min()-1e-8, data.max()+ws+1e-8, ws))  # noqa
        density = np.histogram(data, bins=np.arange(data.min()-1e-8, data.max()+ws+1e-8, ws), density=True)[0]
    else:
        vals = np.histogram(data, bins=bins)[0]
        density = np.histogram(data, bins=bins, density=True)[0]
    x_hist = bins[1:]+((bins[1] - bins[0])/2)
    if m in ('mratio', 'mratio_bounded', 'mratio_exclude'):
        plt.bar(x_hist, vals, width=ws*1.05, facecolor=color_hist, label='Data distribution')  # noqa
    else:
        plt.bar(x_hist, vals, width=(xlim[1] - xlim[0])/70, facecolor=color_hist, label='Data distribution')

    x = np.linspace(xlim[0], xlim[1], 200)
    mu, sigma = norm.fit(data)
    plt.plot(x, norm.pdf(x, mu, sigma) * np.max(vals) / np.max(density), 'g-', lw=1.5, label='Gaussian fit')
    test = shapiro(data)
    print(f'{m} W = {test.statistic:.2f}, p = {test.pvalue:}')  # noqa
    plt.text(0.04, 0.85, f'$W={test.statistic:.2f}$', fontsize=9, transform=ax.transAxes, color='g')  # noqa
    plt.text(0.04, 0.73, f'$min={np.min(data):.2f}$', fontsize=9, transform=ax.transAxes, color='k')
    plt.text(0.04, 0.61, f'$max={np.max(data):.2f}$', fontsize=9, transform=ax.transAxes, color='k')
    plt.text(0.04, 0.49, f'$skew={skew(data):.2f}$', fontsize=9, transform=ax.transAxes, color='k')

    plt.xlim(xlim)
    plt.ylim(ylim)
    if i in (0, 3, 6):
        plt.ylabel('Histogram')
    if m == 'd1':
        plt.xticks([0, 2, 4], ['0\n[0.5]', '2\n[0.84]', '4\n[0.98]'], linespacing=0.85)
        ax.xaxis.set_tick_params(pad=0.5)
    plt.text(0.04, 1.02, mapping[m], transform=ax.transAxes, fontsize=11,
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='k', lw=0.4, pad=2.5), ha='left')
    plt.text((-0.23, -0.3)[int(i in [0, 3, 6])], 1.02, 'ABCDEFGHI'[i], transform=plt.gca().transAxes, color=(0, 0, 0),
             fontsize=15)

plt.tight_layout()
set_fontsize(label=12, tick=10, title=13)
plt.subplots_adjust(hspace=0.375, top=0.95, bottom=0.045, wspace=0.27, right=0.97)
savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
