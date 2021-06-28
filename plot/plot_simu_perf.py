from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from plot_util import savefig
from plot_util import set_fontsize

df = pd.read_pickle('../data/simu_perf_mca_beta.pkl')
df_hmeta = pd.read_pickle('../data/simu_perf_mca_beta_hmeta.pkl')
df['mratio_hmeta'] = df_hmeta['mratio_hmeta'].values
df = df[df.d1_fit <= 5]

mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$",
    mratio=r"$M_{ratio}$", mratio_bounded=r"bounded $M_{ratio}$",
    mratio_logistic=r"logistic $M_{ratio}$", mratio_logistic2=r"logistic $M_{ratio}$",
    mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$", mdiff=r"$M_{diff}$"
)

plt.figure(figsize=(10, 5))

for j, measure in enumerate(
        ['auc', 'metad1', 'mdiff', 'mratio', 'mratio_bounded', 'logmratio', 'mratio_logistic', 'mratio_hmeta']):
    ax = plt.subplot(2, 4, j + 1)
    for i in range(6):
        sigma_sens_ids = df[df.sigma_meta_id == i].groupby('sigma_sens_id').d1_fit.mean().index.values
        x = df[df.sigma_meta_id == i].groupby('sigma_sens_id').d1_fit.mean().values
        mca = df[df.sigma_meta_id == i].groupby('sigma_sens_id')[measure].mean().values
        # mca_std = df[df.sigma_meta_id == i].groupby('sigma_sens')[measure].std().values
        mca_std_neg = np.array([np.sqrt(np.square(df[(df.sigma_meta_id == i) & (df.sigma_sens_id == id) & (df[measure] <= df[(df.sigma_meta_id == i) & (df.sigma_sens_id == id)][measure].mean() + 1e-15)][measure].values -\
                                                  df[(df.sigma_meta_id == i) & (df.sigma_sens_id == id)][measure].mean()).mean()) for id in sigma_sens_ids])
        mca_std_pos = np.array([np.sqrt(np.square(df[(df.sigma_meta_id == i) & (df.sigma_sens_id == id) & (df[measure] >= df[(df.sigma_meta_id == i) & (df.sigma_sens_id == id)][measure].mean() - 1e-15)][measure].values -
                                                  df[(df.sigma_meta_id == i) & (df.sigma_sens_id == id)][measure].mean()).mean()) for id in sigma_sens_ids])
        plt.plot(x, mca, label=f'{df[df.sigma_meta_id == i].sigma_meta.values[0]:.2g}')
        # plt.fill_between(x, mca - mca_std, mca + mca_std, alpha=0.5)
        plt.fill_between(x, mca - mca_std_neg, mca + mca_std_pos, alpha=0.5)
    plt.text(0.5, 0.98, mapping[measure], transform=ax.transAxes, fontsize=12,
             bbox=dict(facecolor='white', alpha=0.95, edgecolor='k', lw=0.4, pad=1.5), ha='center')
    plt.xlim((0, 5))
    if measure in ('mratio', 'mratio_bounded', 'mratio_hmeta'):
        plt.ylim((-0.15, 1.35))
    elif measure == 'logmratio':
        plt.ylim((-3, 1))
    elif measure == 'mratio_logistic':
        plt.ylim((0.25, 0.65))
    if j >= 4:
        plt.xticks(range(0, 6))
        # for d1, p in enumerate(['[.50]', '[.69]', '[.84]', '[.93]', '[.98]', '[.99]']):
        #     plt.text(d1 / 5, -0.175, p, transform=ax.transAxes, fontsize=9, ha='center')
        if j == 5:
            # plt.text(1.1, -0.32, "Type 1 performance d' [proportion correct]", transform=ax.transAxes, fontsize=12,
            #          ha='center')
            plt.text(1.1, -0.27, "Type 1 performance d' [proportion correct]", transform=ax.transAxes, fontsize=12,
                     ha='center')
        ax.xaxis.set_tick_params(pad=2)
    else:
        plt.xticks(range(0, 6), [])
    plt.text((-0.23, -0.25)[int(j in (0, 4))], 1.03, 'ABCDEFGH'[j], transform=plt.gca().transAxes, color=(0, 0, 0),
             fontsize=17)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.2g'))
    if j == 3:
        plt.legend(bbox_to_anchor=(1.03, 0.4), loc="upper left", title='Metacognitive\nnoise ' + r'$\sigma_\mathrm{m}$',
                   fontsize=10, title_fontsize=10, borderpad=0.25)

set_fontsize(label=11, tick=10)
plt.subplots_adjust(hspace=0.2, wspace=0.27, left=0.05, right=0.88, top=0.94, bottom=0.13)
savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
