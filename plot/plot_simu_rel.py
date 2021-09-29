import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_util import savefig, set_fontsize

pd.set_option('use_inf_as_na', True)

HOME = os.path.expanduser('~')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

noise_model = 'beta'

reload = False

if reload:
    df = pd.read_pickle(f'../data/simu_rel_mca_{noise_model}.pkl')
    df_hmeta = pd.read_pickle(f'../data/simu_rel_mca_hmeta_{noise_model}.pkl')
    iter_hmeta = sorted(df_hmeta.iter.unique())
    for j, i in enumerate(iter_hmeta):
        for sigma_sens in [2, 1, 0.64, 0.48, 0.3, 0.24, 0.19, 0.15]:
            for col in ('mratio_hmeta', 'mratio_hmeta_pearson', 'mratio_hmeta_mae', 'mratio_hmeta_nmae'):
                df.loc[(df.iter == j) & np.isclose(df.sigma_sens, sigma_sens, atol=0.005), col] = \
                    df_hmeta[(df_hmeta.iter == i) & np.isclose(df_hmeta.sigma_sens, sigma_sens, atol=0.005)][col].values
    df.to_pickle(f'../data/simu_rel_mca_{noise_model}_full.pkl')
else:
    df = pd.read_pickle(f'../data/simu_rel_mca_{noise_model}_full.pkl')

df = df[(df.nsamples <= 1000)]
sigma_sens_list = [2, 1, 0.64, 0.48, 0.41, 0.3, 0.24, 0.195, 0.155, 0]
sigma_sens_ids = [8, 5, 3, 1, 0]  # 95, 80, 70, 60, 55

for suffix in ['pearson', 'nmae']:
    df[f'mratio_bounded_minus_mratio_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'mratio_{suffix}']
    df[f'logmratio_minus_mratio_{suffix}'] = df[f'logmratio_{suffix}'] - df[f'mratio_{suffix}']
    df[f'mratio_bounded_minus_logmratio_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'logmratio_{suffix}']
    df[f'mratio_bounded_minus_mratio_bound10_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'mratio_bound10_{suffix}']  # noqa
    df[f'logmratio_minus_mratio_bound10_{suffix}'] = df[f'logmratio_{suffix}'] - df[f'mratio_bound10_{suffix}']
    df[f'mratio_exclude_minus_mratio_bound10_{suffix}'] = df[f'mratio_exclude_{suffix}'] - df[f'mratio_bound10_{suffix}']  # noqa
    df[f'mratio_bounded_minus_logmratio_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'logmratio_{suffix}']
for suffix in ['pearson']:
    df[f'mratio_hmeta_minus_mratio_bound10_{suffix}'] = df[f'mratio_hmeta_{suffix}'] - df[f'mratio_bound10_{suffix}']
    df[f'mratio_hmeta_minus_mratio_{suffix}'] = df[f'mratio_hmeta_{suffix}'] - df[f'mratio_{suffix}']

mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$",
    mratio=r"$M_{ratio}$", mratio_bound10=r"$M_{ratio}$",
    mratio_bounded=r"bounded $M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$",
    mratio_exclude=r"$M_{ratio}$ (excl.)", mdiff=r"$M_{diff}$",
    mratio_exclude_minus_mratio=r"$M_{ratio}$ (excl.)$ - M_{ratio}$",
    mratio_bounded_minus_mratio=r"bounded $M_{ratio} - M_{ratio}$",
    logmratio_minus_mratio=r"$\log\,M_{ratio} - M_{ratio}$",
    mratio_hmeta_minus_mratio=r"hierarchical $M_{ratio} - M_{ratio}$",
    mratio_exclude_minus_mratio_bound10=r"$M_{ratio}$ (excl.)$\,-\,M_{ratio}$",
    mratio_bounded_minus_mratio_bound10=r"bounded $M_{ratio} - M_{ratio}$",
    logmratio_minus_mratio_bound10=r"$\log\,M_{ratio} - M_{ratio}$",
    mratio_hmeta_minus_mratio_bound10=r"hierarchical $M_{ratio} - M_{ratio}$",
)

xlim = [0, 1000]
xticks = np.arange(0, 1001, 200)

color_r = np.array([0, 1, 3])
color_nmae = np.array([0, 1, 0])

fig = plt.figure(figsize=(10, 5.5))
ax = [None] * 18
order = [1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 12, 17, 13, 18, 14, 19, 15, 20]
measures = ('mratio_bound10', 'mratio_exclude', 'mratio_bounded', 'logmratio', 'mratio_hmeta',
            'mratio_exclude_minus_mratio_bound10', 'mratio_bounded_minus_mratio_bound10',
            'logmratio_minus_mratio_bound10', 'mratio_hmeta_minus_mratio_bound10')
for i, m in enumerate(measures):
    for k, suffix in enumerate(['_pearson', '_nmae']):

        ax[i*2 + k] = plt.subplot(4, 5, order[i*2 + k])
        for j, sigma_sens_id in enumerate(sigma_sens_ids):
            if not ((k == 1) and 'hmeta' in m):
                mean = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].mean()  # noqa
                # sem = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].std().values   # noqa
                color = (np.log(0.85 * (4-j) + 2.5) - np.log(2.5)) * np.array(([0.5, 0.5, 1], [0.5, 1, 0.5])[k])
                label = f'{df[df.sigma_sens_id == sigma_sens_id].performance.mean():.2f}' if k == 0 else ' '
                # plt.fill_between(mean.index.values, mean - sem, mean + sem, color=color, alpha=0.6, label=label)
                plt.plot(mean.index.values, mean, color=color, lw=2, label=label, clip_on=False)
        if i == 0:
            if k == 0:
                leg = plt.legend(title='Prop. correct', loc='upper left', bbox_to_anchor=(-0.2, -1.905), borderpad=0.5,
                                 handletextpad=2, handlelength=1)
            else:
                plt.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.965), frameon=False, handlelength=1)
        if k == 0:
            title = plt.title(mapping[m])
            if i == 6:
                title.set_position((0.52, 1))
            if i == 8:
                title.set_position((0.515, 1))
        if (k == 1) and 'hmeta' in m:
            plt.xticks([])
        elif ((i > 4) & (k == 1)) | ((k == 0) and 'hmeta' in m):
            plt.xticks(xticks, [0, 200, 400, 600, 800, '1k'])
            plt.xlabel('Number of trials')
        elif (i <= 4) & (k == 1):
            plt.xticks(xticks, [0, 200, 400, 600, 800, '1k'])
        else:
            plt.xticks(xticks, [])
        if (i in (0, 5)) and (k == 0):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'Pearson $r$')
        elif (i in (0, 5)) and (k == 1):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'NMAE')
        ax[i*2 + k].tick_params(axis="y", direction='in')  # noqa

        if not ((k == 1) and 'hmeta' in m):
            if 'minus' in m:
                plt.plot(xlim, [0, 0], color='k', lw=0.75)
            for xtick in xticks:
                plt.plot([xtick, xtick], [-1, 2], color=(0.7, 0.7, 0.7), lw=0.5, zorder=-10)

        if k == 0:
            plt.text((-0.08, -0.3)[int(i in [0, 5])] - (i == 5)*0.1 - (i in [6, 8])*0.03, 1.1, 'ABCDEFGHI'[i],
                     transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
            if 'minus' in m:
                yticks = [0, 0.2, 0.4]
                if i in (0, 5):
                    plt.yticks(yticks)
                else:
                    plt.yticks(yticks, [])
                plt.ylim((-0.15, 0.4))
            else:
                yticks = [0, 0.2, 0.4, 0.6, 0.8, 1]
                if i in (0, 5):
                    plt.yticks(yticks)
                else:
                    plt.yticks(yticks, [])
                plt.ylim((0, 1))
        else:
            if 'minus' in m:
                yticks = [-0.4, -0.2, 0]
                if i in (0, 5):
                    plt.yticks(yticks)
                else:
                    if not ((k == 1) and 'hmeta' in m):
                        plt.yticks(yticks, [])
                    else:
                        plt.box(False)
                        plt.yticks([])
                plt.ylim((-0.42, 0.1))
            else:
                yticks = [0, 0.4, 0.8, 1.2, 1.6]
                if i in (0, 5):
                    plt.yticks(yticks)
                else:
                    if not ((k == 1) and 'hmeta' in m):
                        plt.yticks(yticks, [])
                    else:
                        plt.box(False)
                        plt.yticks([])
                plt.ylim((0, 1.63))
        if not ((k == 1) and 'hmeta' in m):
            for ytick in yticks:
                plt.plot(xlim, [ytick, ytick], color=(0.7, 0.7, 0.7), lw=0.5, zorder=-10)

        plt.xlim(xlim)

set_fontsize(label=11, tick=9, title=11)
plt.subplots_adjust(hspace=0.35, wspace=0.1, top=0.94, bottom=0.09, right=0.987, left=0.064)

for i in (1, 3, 5, 7, 9):
    ax[i].set_position([*(np.array(ax[i]._position)[0] + (0, 0.028)), ax[i]._position.width, ax[i]._position.height])  # noqa

for i in (10, 12, 14, 16):
    ax[i].set_position([*(np.array(ax[i]._position)[0] - (0, 0.028)), ax[i]._position.width, ax[i]._position.height])  # noqa

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
