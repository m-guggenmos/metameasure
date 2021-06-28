import os
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.transforms
import numpy as np
import pandas as pd

from plot_util import savefig, set_fontsize

pd.set_option('use_inf_as_na', True)

HOME = os.path.expanduser('~')

colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

noise_model = 'beta'

reload = True

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

df = df[(df.nsamples <= 1000)]
# df['mratio_logistic_pearson'] = df['mratio_logistic2_pearson']
# df['mratio_logistic_nmae'] = df['mratio_logistic2_nmae']
sigma_sens_list = [2, 1, 0.64, 0.48, 0.41, 0.3, 0.24, 0.195, 0.155, 0]
sigma_sens_ids = [8, 5, 3, 1, 0]  # 95, 80, 70, 60, 55

for suffix in ['pearson', 'nmae']:
    df[f'mratio_bounded_minus_mratio_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'mratio_{suffix}']
    df[f'logmratio_minus_mratio_{suffix}'] = df[f'logmratio_{suffix}'] - df[f'mratio_{suffix}']
    df[f'mratio_logistic_minus_mratio_{suffix}'] = df[f'mratio_logistic_{suffix}'] - df[f'mratio_{suffix}']
    df[f'mratio_hmeta_minus_mratio_{suffix}'] = df[f'mratio_hmeta_{suffix}'] - df[f'mratio_{suffix}']
    df[f'mratio_bounded_minus_logmratio_{suffix}'] = df[f'mratio_bounded_{suffix}'] - df[f'logmratio_{suffix}']
    # df[f'mratio_hmeta_{suffix}'] = df[f'logmratio_{suffix}']

# mapping = dict(d1_nmae=r"$d'$", d1_fit_nmae=r"$d'$", perf_nmae=r"$Accuracy$", auc_nmae=r"$\mathit{AUROC2}}$", metad1_nmae=r"$\mathit{meta}$-$d'$", mratio_nmae=r"$M_{ratio}$",
#                mratio_hmeta_nmae=r"hierarchical $M_{ratio}$", logmratio_nmae=r"$\log\, M_{ratio}$",
#                mdiff_nmae=r"$M_{diff}$")
mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$", mratio=r"$M_{ratio}$",
    mratio_bounded=r"bounded $M_{ratio}$", mratio_logistic=r"logistic $M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$",
    mdiff=r"$M_{diff}$",
    mratio_bounded_minus_mratio=r"bounded $M_{ratio} - M_{ratio}$",
    logmratio_minus_mratio=r"$\log\,M_{ratio} - M_{ratio}$",
    mratio_logistic_minus_mratio=r"logistic $M_{ratio} - M_{ratio}$",
    mratio_hmeta_minus_mratio=r"hierarchical $M_{ratio} - M_{ratio}$",
    # mratio_bounded_minus_logmratio=r"bnd. $M_{ratio} - \log M_{ratio}$"
)

xlim = [0, 1000]
xticks = [0, 250, 500, 750, 1000]

color_r = np.array([0, 1, 3])
color_nmae = np.array([0, 1, 0])

fig = plt.figure(figsize=(10, 6.5))
ax = [None] * 18
# order = [2, 6, 3, 7, 4, 8, 9, 13, 10, 14, 11, 15, 12, 16]
order = [1, 6, 2, 7, 3, 8, 4, 9, 5, 10, 12, 17, 13, 18, 14, 19, 15, 20]
# for i, m in enumerate(('metad1_nmae', 'auc_nmae', 'mdiff_nmae', 'mratio_nmae', 'mratio_bounded_nmae', 'logmratio_nmae', 'mratio_hmeta_nmae')):
for i, m in enumerate(('mratio', 'logmratio', 'mratio_bounded', 'mratio_logistic', 'mratio_hmeta', 'logmratio_minus_mratio', 'mratio_bounded_minus_mratio', 'mratio_logistic_minus_mratio', 'mratio_hmeta_minus_mratio')):
    for k, suffix in enumerate(['_pearson', '_nmae']):

        ax[i*2 + k] = plt.subplot(4, 5, order[i*2 + k])
        # plt.plot(xlim, [0, 0], 'k-', lw=0.5, zorder=-100)
        for j, sigma_sens_id in enumerate(sigma_sens_ids):
            mean = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].mean()
            sem = df[df.sigma_sens_id == sigma_sens_id].groupby(['iter', 'nsamples'], as_index=False)[f'{m}{suffix}'].mean().groupby('nsamples')[f'{m}{suffix}'].sem().values
            # plt.fill_between(mean.index.values, mean.values-sem.values, mean.values+sem.values, color=(np.log(0.45 * j + 2.5) - np.log(2.5)) * np.ones(3), alpha=0.6)
            # plt.plot(mean.index.values, mean.values, color=(np.log(0.45 * j + 2.5) - np.log(2.5)) * np.ones(3), label=f'{df[df.sigma_sens == sigma_sens].performance.mean():.2f}')
            color = (np.log(0.85 * (4-j) + 2.5) - np.log(2.5)) * np.array(([0.5, 0.5, 1], [0.5, 1, 0.5])[k])
            if not ((k == 1) and 'hmeta' in m):
                # if m == 'mratio_hmeta_minus_mratio':
                means_ = np.array([mean.values[max(0, i-1):i+2].mean() for i in range(len(mean))])
                plt.fill_between(mean.index.values, means_ - sem / np.sqrt(3), means_ + sem / np.sqrt(3), color=color, alpha=0.6, label=f'{df[df.sigma_sens_id == sigma_sens_id].performance.mean():.2f}' if k == 0 else ' ')
                plt.plot(mean.index.values, means_, color=color, lw=1.5)
                # else:
                #     plt.fill_between(mean.index.values, mean.values - sem, mean.values + sem, color=color, alpha=0.6, label=f'{df[df.sigma_sens_id == sigma_sens_id].performance.mean():.2f}' if k == 0 else ' ')
                #     plt.plot(mean.index.values, mean.values, color=color, lw=1.5)
            # plt.plot(mean.index.values, mean.values, color=color, lw=3, label=f'{df[df.sigma_sens == sigma_sens].performance.mean():.2f}' if k == 0 else ' ')
        if i == 0:
            if k == 0:
                leg = plt.legend(title='Prop. correct', loc='upper left', bbox_to_anchor=(-0.2, -1.905), borderpad=0.5, handletextpad=2, handlelength=1)
            else:
                plt.legend(loc='upper left', bbox_to_anchor=(-0.02, -0.96), frameon=False, handlelength=1)
        if k == 0:
            plt.title(mapping[m])
        if (i > 4) & (k == 1):
            plt.xticks(xticks)
            plt.xlabel('Number of trials')
        else:
            plt.xticks(xticks, [])
        if (i in (0, 5)) and (k == 0):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'Pearson $r$')
        elif (i in (0, 5)) and (k == 1):
            plt.ylabel((r'$\Delta\,$' if 'minus' in m else '') + 'NMAE')
        # else:
        #     plt.yticks(plt.gca().get_yticks(), [])
        ax[i*2 + k].tick_params(axis="y", direction='in')

        for z, label in enumerate(ax[i*2 + k].xaxis.get_majorticklabels()[1:]):
            label.set_transform(label.get_transform() - matplotlib.transforms.ScaledTranslation(7/72+(z==3)*4/72, 0, fig.dpi_scale_trans))

        if 'minus' in m:
            plt.plot(xlim, [0, 0], color='k', lw=0.75)

        if k == 0:
            plt.text((-0.08, -0.3)[int(i in [0, 5])]-(i==5)*0.1, 1.1, 'ABCDEFGHI'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=17)
            if 'minus' in m:
                # if 'hmeta' in m:
                #     plt.yticks([0, 0.2, 0.4])
                #     plt.ylim((-0.15, 0.43))
                # else:
                if i in (0, 5):
                    plt.yticks([0, 0.1, 0.2])
                else:
                    plt.yticks([0, 0.1, 0.2], [])
                plt.ylim((-0.08, 0.26))
            else:
                if i in (0, 5):
                    plt.yticks([0, 0.5, 1])
                else:
                    plt.yticks([0, 0.5, 1], [])
                plt.ylim((0, 1))
        else:
            if 'minus' in m:
                if i in (0, 5):
                    plt.yticks([-0.4, -0.2, 0])
                else:
                    plt.yticks([-0.4, -0.2, 0], [])
                plt.ylim((-0.5, 0.03))
            else:
                if i in (0, 5):
                    plt.yticks([0, 0.5, 1, 1.5])
                else:
                    plt.yticks([0, 0.5, 1, 1.5], [])
                plt.ylim((0, 1.8))

        plt.xlim(xlim)

set_fontsize(label=10.5, tick=10.5, title=10)
# plt.tight_layout()
plt.subplots_adjust(wspace=0.1, hspace=0.3, top=0.94, bottom=0.05, left=0.07, right=0.98)

for i in (1, 3, 5, 7, 9):
    ax[i].set_position([*(np.array(ax[i]._position)[0] + (0, 0.03)), ax[i]._position.width, ax[i]._position.height])

for i in (10, 12, 14, 16):
    ax[i].set_position([*(np.array(ax[i]._position)[0] - (0, 0.01)), ax[i]._position.width, ax[i]._position.height])

for i in (11, 13, 15, 17):
    ax[i].set_position([*(np.array(ax[i]._position)[0] + (0, 0.02)), ax[i]._position.width, ax[i]._position.height])


# for i in (16, 17):
#     ax[i].set_position([*(np.array(ax[i]._position)[0] + (0.017, 0)), ax[i]._position.width, ax[i]._position.height])

savefig(f'../img/{Path(__file__).stem}.png')
plt.show()