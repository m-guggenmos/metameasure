import os
import sys
import numpy as np
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import StrMethodFormatter

from plot_coefficients import plot_coefficients
from plot_util import set_fontsize, savefig
sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.regression import linear_regression, regress  # noqa


HOME = os.path.expanduser('~')

z = pd.read_pickle('../data/mca_test_retest.pkl')
z = z[~z.mratio.isna()]
z['logmratio'] = np.log(np.maximum(0.1, z['mratio'].values))
z['logmratio_test'] = np.log(np.maximum(0.1, z['mratio_test'].values))
z['logmratio_retest'] = np.log(np.maximum(0.1, z['mratio_retest'].values))
# z = z[z.study_id.isin(np.where((z.groupby('study_id').nsamples.mean() >= 400).values)[0])]
z = z[z.study_id.isin(np.where((z.groupby('study_id').nsamples.mean() >= 400).values)[0]) & (z.d1 > 0.5)]

mapping = dict(
    d1=r"$d'$", d1_fit=r"$d'$", perf=r"$Accuracy$", auc=r"$\mathit{AUROC2}}$", metad1=r"$\mathit{meta}$-$d'$",
    mratio=r"$M_{ratio}$", mratio_hmeta=r"hierarchical $M_{ratio}$",
    logmratio=r"$\log\, M_{ratio}$", mdiff=r"$M_{diff}$"
)


ps_mapping = dict(
    entropy='entropy',
    min_trials_per_subject='ntrials',
    nratings='nratings',
    continuous='continuous',
    conf_dec_simu='simultaneous',
    any_feedback='feedback',
    online_staircase='staircase',
    conf_norm='confidence',
    mratio=r'$M_{ratio}$',
    d1_orig="d'",
    d1="d'",
    metad1="meta-d'",
    nsubjects="nsubjects",
)

zstudy = z.groupby('study_id').mean().reset_index()
zstudy['study_name'] = [z[z.study_id == sid].study_name.values[0] for sid in zstudy.study_id.unique()]
db = pd.read_excel('../data/Database_Information.xlsx')
dbvars = ['nsubjects', 'min_trials_per_subject', 'nratings', 'continuous', 'conf_dec_simu', 'trial_feedback',
          'block_feedback', 'online_staircase']
for name in sorted(zstudy.study_name.unique()):
    for var in dbvars:
        zstudy.loc[zstudy.study_name == name, var] = db.loc[db.Name_in_database == name, var].values[0]
        if var == 'nratings':
            zstudy.loc[(zstudy.study_name == name), 'conf_norm'] = zstudy[zstudy.study_name == name].conf / \
                                                                   zstudy[zstudy.study_name == name].nratings
zstudy['any_feedback'] = (zstudy['trial_feedback'] == 1) | (zstudy['block_feedback'] == 1)
zstudy['entropy'] = zstudy['entropy'] / np.log(zstudy['nratings'])


dbvars = ['nsubjects', 'min_trials_per_subject',  'nratings', 'continuous', 'conf_dec_simu', 'any_feedback',
          'online_staircase']


measure = 'mratio'

axes = [None]*2
plt.figure(figsize=(11, 3.5))

for i, modus in enumerate(('correlation', 'nmae')):

    axes[i] = plt.subplot(1, 2, i + 1)

    for k, study in enumerate(sorted(z.study_id.unique())):
        test = z.loc[(z.study_id == study) & ~z[f'{measure}_test'].isna() & ~z[f'{measure}_retest'].isna(),  f'{measure}_test'].values  # noqa
        retest = z.loc[(z.study_id == study) & ~z[f'{measure}_test'].isna() & ~z[f'{measure}_retest'].isna(), f'{measure}_retest'].values  # noqa
        test_zb = test - min(test.min(), retest.min())
        retest_zb = retest - min(test.min(), retest.min())
        if (len(test) >= 10) and (len(retest) >= 10):
            if modus == 'correlation':
                zstudy.loc[zstudy.study_id == study, f'{measure}_reliability'] = \
                    np.arctanh(regress(test, retest, method='bivariate', outlier_stds=4).r)
            elif modus == 'nmae':
                zstudy.loc[zstudy.study_id == study, f'{measure}_reliability'] = \
                    np.mean(np.abs(test_zb - retest_zb)) / (0.5*(np.mean(np.abs(test_zb - np.mean(retest_zb))) +
                                                                 np.mean(np.abs(retest_zb - np.mean(test_zb)))))

    ps = ['d1', 'conf_norm', 'mratio'] + dbvars
    model = linear_regression(
        zstudy[~zstudy[f'{measure}_reliability'].isna()],
        patsy_string=f'{measure}_reliability ~ ' + ' + '.join(ps),
        model_blocks=False,
        ols=True,
        ignore_warnings=True, reml=False, print_data=False, print_corr_table=False, return_model=True,
        standardize_vars_excl=['category_id']
    )

    plot_coefficients(model, zstudy[~zstudy[f'{measure}_reliability'].isna()], exclude_categorical=True,
                      striped_background=True, asterisk_yshift=-0.015, asterisk_fontsize=9, circle_fontsize=11,
                      circle_yshift=-0.0, mapping=ps_mapping, max_x_asterisks=0.95,
                      stripe_color_dark=(0.15, 0.15, 0.15), show_trend=True)

    plt.text((-0.38, -0.34)[i], 1.01, 'AB'[i], transform=plt.gca().transAxes, color=(0, 0, 0), fontsize=19)
    plt.xlim((-1, 1))
    plt.xlabel('regression / correlation coefficent')
    ttl = plt.title(('Pearson correlation (z-trans.)', 'NMAE')[i])
    ttl.set_position((0.5, 1.05))
    if i == 1:
        leg = plt.legend(loc='upper left', fontsize=11, bbox_to_anchor=(1.02, 0.7))

    axes[i].xaxis.set_major_formatter(StrMethodFormatter("{x}"))  # noqa

set_fontsize(label=12, xtick=11, ytick=11, title=15)

plt.tight_layout()
plt.subplots_adjust(hspace=0.14, wspace=0.4)
savefig(f'../img/{Path(__file__).stem}.png')
plt.show()
