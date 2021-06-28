from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import rcParams

from plot_util import savefig, set_fontsize

rcParams['text.usetex'] = True
colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

noise_model = 'beta'

reload = False

if reload:
    df = pd.read_pickle(f'../data/simu_rel_mca_{noise_model}_nohmeta.pkl')
    df_hmeta = pd.read_pickle(f'../data/simu_rel_mca_hmeta_{noise_model}.pkl')
    iter_hmeta = sorted(df_hmeta.iter.unique())
    for j, i in enumerate(iter_hmeta):
        for sigma_sens in [2, 1, 0.64, 0.48, 0.3, 0.24, 0.19, 0.15]:
            for col in ('mratio_hmeta', 'mratio_hmeta_pearson', 'mratio_hmeta_mae', 'mratio_hmeta_nmae'):
                df.loc[(df.iter == j) & np.isclose(df.sigma_sens, sigma_sens, atol=0.005), col] = df_hmeta[(df_hmeta.iter == i) & np.isclose(df_hmeta.sigma_sens, sigma_sens, atol=0.005)][col].values
    df.to_pickle(f'../data/simu_rel_mca_{noise_model}.pkl')
else:
    df = pd.read_pickle(f'../data/simu_rel_mca_{noise_model}.pkl')

df['mratio_logistic_pearson'] = df['mratio_logistic2_pearson']
df['mratio_logistic_nmae'] = df['mratio_logistic2_nmae']
sigma_sens_list = [2, 1, 0.64, 0.48, 0.41, 0.3, 0.24, 0.195, 0.155, 0]
sigma_sens_ids = 9-np.array([1, 4, 6, 8, 9])

for suffix in ['pearson', 'nmae']:
    df[f'mratio_bounded_minus_logmratio_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'logmratio_{suffix}']
    df[f'mratio_bounded_minus_mratio_logistic_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'mratio_logistic_{suffix}']
    df[f'mratio_logistic_minus_logmratio_{suffix}'] = df[f'mratio_logistic_{suffix}'] - df[f'logmratio_{suffix}']
    df[f'mratio_hmeta_minus_mratio_bounded_{suffix}'] = df[f'mratio_hmeta_{suffix}'] - df[f'mratio_bounded_{suffix}']
    df[f'mratio_hmeta_minus_logmratio_{suffix}'] = df[f'mratio_hmeta_{suffix}'] - df[f'logmratio_{suffix}']
    df[f'mratio_hmeta_minus_mratio_logistic_{suffix}'] = df[f'mratio_hmeta_{suffix}'] - df[f'mratio_logistic_{suffix}']


mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$",
    mratio_bounded=r"bounded $M_{ratio}$", mratio_logistic=r"logistic $M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$",
    mdiff=r"$M_{diff}$",
    mratio_bounded_minus_logmratio=r"bounded $M_{ratio} - \log\, M_{ratio}$",
    mratio_bounded_minus_mratio_logistic=r"bounded $M_{ratio} -$ logistic $M_{ratio}$",
    mratio_logistic_minus_logmratio=r"logistic $M_{ratio} - \log\, M_{ratio}$",
    mratio_hmeta_minus_mratio_bounded=r"hierarchical $M_{ratio} -$ bounded $M_{ratio}$",
    mratio_hmeta_minus_logmratio=r"hierarchical $M_{ratio} - \log\, M_{ratio}$",
    mratio_hmeta_minus_mratio_logistic=r"hierarchical $M_{ratio} -$ logistic $M_{ratio}$",
)

xlim = [0, 1000]
xticks = [0, 250, 500, 750, 1000]

color_r = np.array([0, 1, 3])
color_nmae = np.array([0, 1, 0])

fig = plt.figure(figsize=(8, 5.5))
ax = [None] * 12
# order = [2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16]
order = [1, 4, 2, 5, 3, 6, 7, 10, 8, 11, 9, 12]

for i, m in enumerate(('mratio_bounded_minus_logmratio', 'mratio_logistic_minus_logmratio', 'mratio_bounded_minus_mratio_logistic', 'mratio_hmeta_minus_mratio_bounded', 'mratio_hmeta_minus_logmratio', 'mratio_hmeta_minus_mratio_logistic')):
    for k, suffix in enumerate(['_pearson', '_nmae']):

        ax[i*2 + k] = plt.subplot(4, 3, order[i*2 + k])
        for j, sigma_sens_id in enumerate(sigma_sens_ids):
            mean = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].mean()
            sem = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].sem().values
            color = (np.log(0.85 * (4-j) + 2.5) - np.log(2.5)) * np.array(([0.5, 0.5, 1], [0.5, 1, 0.5])[k])

            if not ((k == 1) and 'hmeta' in m):
                # if 'hmeta' in m:
                means_ = np.array([mean.values[max(0, i-1):i+2].mean() for i in range(len(mean))])
                plt.fill_between(mean.index.values, means_ - sem / np.sqrt(3), means_ + sem / np.sqrt(3), color=color, alpha=0.6, label=f'{df[df.sigma_sens_id == sigma_sens_id].performance.mean():.2f}' if k == 0 else ' ')
                plt.plot(mean.index.values, means_, color=color, lw=1.5)

        if i == 2:
            if k == 0:
                leg = plt.legend(title='Proportion\ncorrect', loc='upper left', bbox_to_anchor=(1.015, 0.195), borderpad=1.1, handletextpad=2.1, handlelength=1)
            else:
                plt.legend(loc='upper left', bbox_to_anchor=(1.17, 0.8667), frameon=False, handlelength=1, handletextpad=0.4)
        if k == 0:
            title = plt.title(mapping[m])
            if i == 3:
                title.set_position((0.45, 1))
            if i == 5:
                title.set_position((0.515, 1))
        # if (i > 2) & (k == 1):
        if (i > 2) & (k == 0):
            plt.xticks(xticks)
            plt.xlabel('Number of trials')
        else:
            plt.xticks(xticks, [])
        if (i in (0, 3)) and (k == 0):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'Pearson $r$')
        elif (i in (0, 3)) and (k == 1):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'NMAE')
        ax[i*2 + k].tick_params(axis="y", direction='in')

        if (i < 3) or (k == 0):
            plt.plot(xlim, [0, 0], color='k', lw=0.75)

        if k == 0:
            plt.text((-0.07, -0.3)[int(i in [0, 3])], 1.1, 'ABCDEF'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
            if i < 3:

                if i == 0:
                    plt.yticks([-0.02, 0, 0.02, 0.04])
                else:
                    plt.yticks([-0.02, 0, 0.02, 0.04], [])
                plt.ylim((-0.02, 0.05))
            else:
                if i == 3:
                    plt.yticks([-0.1, 0, 0.1, 0.2, 0.3])
                else:
                    plt.yticks([-0.1, 0, 0.1, 0.2, 0.3], [])
                plt.ylim((-0.07, 0.26))
        else:
            if i < 3:
                if i == 0:
                    plt.yticks([-0.05, 0, 0.05, 0.1])
                else:
                    plt.yticks([-0.05, 0, 0.05, 0.1], [])
                plt.ylim((-0.05, 0.11))
            else:
                plt.yticks([-0.4, -0.2, 0, 0.2])
                plt.ylim((-0.4, 0.3))

        plt.xlim(xlim)
        if (i > 2) and (k == 1):
            ax[i*2 + k].axis('off')

set_fontsize(label=10.5, tick=10.5, title=10)
plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.94, bottom=0.09, left=0.08, right=0.86)

for i in (1, 3, 5):
    ax[i].set_position([*(np.array(ax[i]._position)[0] + (0, 0.03)), ax[i]._position.width, ax[i]._position.height])

for i in (6, 8, 10):
    ax[i].set_position([*(np.array(ax[i]._position)[0] - (0, 0.03)), ax[i]._position.width, ax[i]._position.height])

rcParams['text.usetex'] = False

savefig(f'../img/{Path(__file__).stem}.png', pad_inches=0.03)
plt.show()