import os
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from confidence.analysis.regression import regress
from plot_util import set_fontsize, savefig

z = pd.read_pickle('../data/mca_test_retest.pkl')
z = z[~z.mratio_test.isna() & ~z.mratio_retest.isna()]

nstudies = len(z.study_id.unique())

rel_pearson, rel_nmae, ntrials = np.full(nstudies, np.nan), np.full(nstudies, np.nan), np.full(nstudies, np.nan)

for i, study_id in enumerate(z.study_id.unique()):
    cond = z.study_id == study_id
    rel_pearson[i] = regress(z[cond].mratio_test.values, z[cond].mratio_retest.values, method='linregress').r
    test_zb = z[cond].mratio_test.values - min(z[cond].mratio_test.values.min(), z[cond].mratio_retest.values.min())
    retest_zb = z[cond].mratio_retest.values - min(z[cond].mratio_test.values.min(), z[cond].mratio_retest.values.min())
    rel_nmae[i] = np.mean(np.abs(test_zb - retest_zb)) / (0.5*(np.mean(np.abs(test_zb - np.mean(retest_zb))) + np.mean(np.abs(retest_zb - np.mean(test_zb)))))
    ntrials[i] = z[cond].nsamples.mean()

plt.figure(figsize=(9, 3))
plt.suptitle('Test-retest reliability of $M_\mathrm{ratio}$', fontsize=14)
ax = plt.subplot(121)
bins = np.arange(-1, 1.1, 0.2)
hist = np.histogram(rel_pearson, bins=bins)[0]
nt = np.array([np.mean(ntrials[(rel_pearson <= bins[i + 1]) & (rel_pearson >= bins[i])]) for i in range(len(bins)-1)])
colors = [tuple(0.85*(1-(n / np.nanmax(nt)))*np.ones(3)) for n in nt]
plt.bar(bins[:-1] + (bins[1] - bins[0]) / 2, hist, color=colors, width=2 / (len(bins)-1))
plt.xlim((-1, 1))
cmap = mpl.colors.LinearSegmentedColormap.from_list('grey_range', [np.nanmax([c[0] for c in colors])*np.ones(3), (0, 0, 0)], N=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=int(np.round(np.nanmin(nt))), vmax=int(np.round(np.nanmax(nt)))))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Number of trials')
plt.xlabel('Pearson correlation')
plt.ylabel('Number of studies')

ax = plt.subplot(122)
bins = np.linspace(0, rel_nmae.max(), 11)
hist = np.histogram(rel_nmae, bins=bins)[0]
nt = np.array([np.mean(ntrials[(rel_nmae <= bins[i + 1]) & (rel_nmae >= bins[i])]) for i in range(len(bins)-1)])
colors = [tuple(0.85*(1-(n / np.nanmax(nt)))*np.ones(3)) for n in nt]
plt.bar(bins[:-1] + (bins[1] - bins[0]) / 2, hist, color=colors, width=rel_nmae.max() / (len(bins)-1))
plt.xlim((0, rel_nmae.max()))
cmap = mpl.colors.LinearSegmentedColormap.from_list('grey_range', [np.nanmax([c[0] for c in colors])*np.ones(3), (0, 0, 0)], N=100)
sm = plt.cm.ScalarMappable(cmap=cmap, norm=mpl.colors.Normalize(vmin=int(np.round(np.nanmin(nt))), vmax=int(np.round(np.nanmax(nt)))))
sm.set_array([])
cbar = plt.colorbar(sm)
cbar.set_label('Number of trials')
plt.xlabel('NMAE')
plt.ylabel('Number of studies')


set_fontsize(label=12, tick=11, title=11)
plt.tight_layout()
plt.subplots_adjust(wspace=0.35)
savefig(f'../img/{Path(__file__).stem}.png')