from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FormatStrFormatter

from plot_util import savefig
from plot_util import set_fontsize

noise_distributions = ['censored_norm', 'truncated_norm', 'censored_lognorm', 'truncated_lognorm']


mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$",
    mratio=r"$M_{ratio}$", mratio_exclude=r"$M_{ratio}$ (excl.)", mratio_bounded=r"bounded $M_{ratio}$",
    mratio_hmeta=r"hierarchical $M_{ratio}$", logmratio=r"$\log\, M_{ratio}$", mdiff=r"$M_{diff}$"
)

dist_names = dict(
    censored_norm='Censored Normal',
    truncated_norm='Truncated Normal',
    censored_lognorm='Censored Lognormal',
    truncated_lognorm='Truncated Lognormal'
)

sigma_meta_ids_ = dict(
    censored_norm=[0, 2, 3, 6, 12, 25],
    truncated_norm=[0, 3, 5, 8, 12, 21],
    censored_lognorm=[0, 1, 2, 14],
    truncated_lognorm=[0, 2, 3, 7, 10, 15]
)


measure = 'mratio'

plt.figure(figsize=(7, 5))

for j, dist in enumerate(noise_distributions):
    df = pd.read_pickle(f'../data/simu_perf_mca_{dist}_new2.pkl')
    df = df[df.d1_fit <= 5]
    ax = plt.subplot(2, 2, j + 1)
    sigma_meta_ids = sigma_meta_ids_[dist]
    for i in range(len(sigma_meta_ids)):
        sigma_sens_ids = df[df.sigma_meta_id == sigma_meta_ids[i]].groupby('sigma_sens_id').d1_fit.mean().index.values
        x = df[df.sigma_meta_id == sigma_meta_ids[i]].groupby('sigma_sens_id').d1_fit.mean().values
        mca = df[df.sigma_meta_id == sigma_meta_ids[i]].groupby('sigma_sens_id')[measure].mean().values
        mca_std_neg = np.array([np.sqrt(np.square(
            df[(df.sigma_meta_id == sigma_meta_ids[i]) & (df.sigma_sens_id == sid) &
               (df[measure] <= df[(df.sigma_meta_id == sigma_meta_ids[i]) & (df.sigma_sens_id == sid)][measure].mean() + 1e-15)][measure].values -  # noqa
            df[(df.sigma_meta_id == sigma_meta_ids[i]) & (df.sigma_sens_id == sid)][measure].mean()).mean())
                                for sid in sigma_sens_ids])
        mca_std_pos = np.array([np.sqrt(np.square(
            df[(df.sigma_meta_id == sigma_meta_ids[i]) & (df.sigma_sens_id == sid) &
               (df[measure] >= df[(df.sigma_meta_id == sigma_meta_ids[i]) & (df.sigma_sens_id == sid)][measure].mean() - 1e-15)][measure].values -  # noqa
            df[(df.sigma_meta_id == sigma_meta_ids[i]) & (df.sigma_sens_id == sid)][measure].mean()).mean())
                                for sid in sigma_sens_ids])
        label = np.format_float_positional(df[df.sigma_meta_id == sigma_meta_ids[i]].sigma_meta.values[0], trim='-',
                                           precision=2)
        plt.plot(x, mca, label=label)
        plt.fill_between(x, mca - mca_std_neg, mca + mca_std_pos, alpha=0.5)
    plt.title(dist_names[dist])
    plt.xlim((0, 5))
    plt.ylim((-0.15, 1.35))
    yticks = np.arange(0, 1.3, 0.25)
    plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.3g'))
    if j in (1, 3):
        plt.yticks(yticks, [])
    else:
        plt.yticks(yticks)
        plt.ylabel('$M_{ratio}$')
    if j >= 2:
        plt.xticks(range(0, 6))
        for d1, p in enumerate(['[.50]', '[.69]', '[.84]', '[.93]', '[.98]', '[.99]']):
            plt.text(d1 / 5, -0.175, p, transform=ax.transAxes, fontsize=9, ha='center')
        if j == 2:
            plt.text(1.5, -0.32, "Type 1 performance d' [proportion correct]", transform=ax.transAxes, fontsize=12,
                     ha='center')
        ax.xaxis.set_tick_params(pad=2)
    else:
        plt.xticks(range(0, 6), [])
    plt.legend(bbox_to_anchor=(1.03, 1), loc="upper left", title=r'$\sigma_\mathrm{m}$',
               fontsize=10, title_fontsize=10, borderpad=0.25)

set_fontsize(label=11, tick=10, title=12)
plt.subplots_adjust(top=0.93, bottom=0.13, left=0.1, right=0.82, hspace=0.3, wspace=0.8)
savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
