import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_util import savefig, set_fontsize

HOME = os.path.expanduser('~')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

noise_model = 'beta'

df = pd.read_pickle(f'../data/simu_rel_mca_{noise_model}.pkl')
df = df[df.nsamples <= 1000]
sigma_sens_list = [2, 1, 0.64, 0.48, 0.41, 0.3, 0.24, 0.195, 0.155, 0]
sigma_sens_ids = 9-np.array([1, 4, 6, 8, 9])  # 95, 80, 70, 60, 55

mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$",
    mratio=r"$M_{ratio}$", mratio_bounded=r"bounded $M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$",
    logmratio=r"$\log\, M_{ratio}$", mdiff=r"$M_{diff}$"
)

xlim = [0, 1000]
xticks = np.arange(0, 1001, 200)

color_r = np.array([0, 1, 3])
color_nmae = np.array([0, 1, 0])

fig = plt.figure(figsize=(10, 3.5))
ax = [None] * 8
order = [1, 5, 2, 6, 3, 7, 4, 8]
for i, m in enumerate(('d1_fit', 'metad1', 'auc', 'mdiff')):
    for k, suffix in enumerate(['_pearson', '_nmae']):

        ax[i*2 + k] = plt.subplot(2, 4, order[i*2 + k])
        if m == 'd1_fit':
            mean = df.groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].mean()  # noqa
            std = df.groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].sem()  # noqa
            plt.fill_between(mean.index.values, mean.values-std.values, mean.values+std.values, color='r', alpha=0.6)
            plt.plot(mean.index.values, mean.values, color='r', lw=2, clip_on=False)
        else:
            for j, sigma_sens_id in enumerate(sigma_sens_ids):
                mean = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].mean()  # noqa
                std = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].sem()  # noqa
                color = (np.log(0.85 * (4-j) + 2.5) - np.log(2.5)) * np.array(([0.5, 0.5, 1], [0.5, 1, 0.5])[k])
                # plt.fill_between(mean.index.values, mean.values-std.values, mean.values+std.values, color=color, alpha=0.6, clip_on=False)  # noqa
                label = f'{df[df.sigma_sens_id == sigma_sens_id].performance.mean():.2f}' if k == 1 else ' '
                plt.plot(mean.index.values, mean.values, color=color, lw=2, clip_on=False, label=label)

        for xtick in xticks:
            plt.plot([xtick, xtick], [-1, 2], color=(0.7, 0.7, 0.7), lw=0.5, zorder=-10)

        if i == 3:
            if k == 0:
                leg = plt.legend(title='Prop. correct', loc='upper left', bbox_to_anchor=(1.015, 0.5), borderpad=0.5,
                                 handletextpad=2.9, handlelength=1)
            else:
                plt.legend(loc='upper left', bbox_to_anchor=(1.215, 1.465), frameon=False, handlelength=1,
                           handletextpad=0.4)
        if k == 0:
            plt.title(mapping[m])
        if k == 1:
            plt.xticks(xticks, [0, 200, 400, 600, 800, '1k'])
            plt.xlabel('Number of trials')
        else:
            plt.xticks(xticks, [])
        if (i == 0) and (k == 0):
            plt.ylabel(r'Pearson $r$')
        elif (i == 0) and (k == 1):
            plt.ylabel('NMAE')
        else:
            plt.yticks(plt.gca().get_yticks(), [])
        ax[i*2 + k].tick_params(axis="y", direction='in')  # noqa

        if k == 0:
            plt.text((-0.08, -0.34)[int(i == 0)], 1.1, 'ABCD'[i], transform=plt.gca().transAxes, color=(0, 0, 0),
                     fontsize=17)
            plt.yticks(np.arange(0, 1.01, 0.2))
            plt.ylim((0, 1))
        else:
            plt.yticks(np.arange(0, 1.61, 0.4))
            plt.ylim((0, 1.63))
        plt.xlim(xlim)
        for ytick in plt.gca().get_yticks():
            plt.plot(xlim, [ytick, ytick], color=(0.7, 0.7, 0.7), lw=0.5, zorder=-10)

set_fontsize(label=11, tick=11, title=13)
plt.subplots_adjust(wspace=0.1, hspace=0.15, left=0.075, right=0.86, bottom=0.15, top=0.9)

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
