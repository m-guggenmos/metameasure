import os
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import rcParams
from scipy.stats import spearmanr
from statsmodels.iolib.summary2 import _simple_tables

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.regression import regress  # noqa

rcParams['text.usetex'] = True

# import warnings
# warnings.filterwarnings('error')

z = pd.read_pickle('../data/mca_test_retest.pkl')
z = z[~z.mratio.isna()]
z['logmratio'] = np.log(np.maximum(0.1, z['mratio'].values))
z['logmratio_test'] = np.log(np.maximum(0.1, z['mratio_test'].values))
z['logmratio_retest'] = np.log(np.maximum(0.1, z['mratio_retest'].values))
z['mratio_bounded'] = np.minimum(1.6, np.maximum(0, z['mratio'].values))
z['mratio_bounded_test'] = np.minimum(1.6, np.maximum(0, z['mratio_test'].values))
z['mratio_bounded_retest'] = np.minimum(1.6, np.maximum(0, z['mratio_retest'].values))
z['mratio_logistic'] = 1 / (1 + np.exp(-(z.mratio-0.8)))
z['mratio_logistic_test'] = 1 / (1 + np.exp(-(z.mratio_test-0.8)))
z['mratio_logistic_retest'] = 1 / (1 + np.exp(-(z.mratio_retest-0.8)))

nstudies = len(z.study_id.unique())
measures = ['d1', 'metad1', 'auc', 'mdiff', 'mratio', 'mratio_bounded', 'logmratio', 'mratio_logistic', 'mratio_hmeta']
nmeasures = len(measures)

reload = True
if reload:

    df = pd.DataFrame(index=range(nstudies*nmeasures))
    df['study'] = np.tile(range(nstudies), nmeasures)
    df['measure'] = np.repeat(range(nmeasures), nstudies)

    for i, m in enumerate(measures):
        print(f'mca: {m} ({i + 1} / {len(measures)})')

        for j, study in enumerate(sorted(z.study_id.unique())):
            zcond = (z.study_id == study) & ~z[f'{m}_test'].isna() & ~z[f'{m}_retest'].isna()
            if (len(z.loc[zcond, f'{m}_test'].values) > 10) and (len(z.loc[zcond, f'{m}_retest'].values) > 10):

                mca_test = z.loc[zcond, f'{m}_test'].values
                mca_retest = z.loc[zcond, f'{m}_retest'].values
                mca_test_zb = mca_test - min(mca_test.min(), mca_retest.min())
                mca_retest_zb = mca_retest - min(mca_test.min(), mca_retest.min())
                cond = (df.measure == i) & (df.study == j)
                df.loc[cond, 'measure_name'] = m
                if np.all(mca_test == 1) and np.all(mca_retest == 1):
                    df.loc[cond, 'pearson_r_ztrans'] = np.arctanh(0.99)
                    df.loc[cond, 'spearman_r_ztrans'] = np.arctanh(0.99)
                    df.loc[cond, 'NMAE'] = 0
                else:
                    df.loc[cond, 'pearson_r_ztrans'] = regress(mca_test, mca_retest, method='bivariate').r
                    df.loc[cond, 'spearman_r_ztrans'] = np.arctanh(np.minimum(0.99, spearmanr(mca_test, mca_retest)[0]))
                    df.loc[cond, 'NMAE'] = np.mean(np.abs(mca_test_zb - mca_retest_zb)) / (0.5*(np.mean(np.abs(mca_test_zb - np.mean(mca_retest_zb))) + np.mean(np.abs(mca_retest_zb - np.mean(mca_test_zb)))))
                df.loc[cond, 'ntrials'] = z[zcond].nsamples.mode().values[0]
                df.loc[cond, 'type1_perf'] = z[zcond].perf.mean()

    df[~df.measure.isna()].to_pickle(f'../data/{Path(__file__).stem}.pkl')


df = pd.read_pickle(f'../data/{Path(__file__).stem}.pkl')

comparisons = [
    ('mratio_bounded', 'mratio'),
    ('logmratio', 'mratio'),
    ('mratio_logistic', 'mratio'),
    ('mratio_hmeta', 'mratio'),
    ('mratio_bounded', 'logmratio'),
    ('mratio_bounded', 'mratio_logistic'),
    ('mratio_bounded', 'mratio_hmeta'),
    ('logmratio', 'mratio_logistic'),
    ('logmratio', 'mratio_hmeta'),
    ('mratio_logistic', 'mratio_hmeta')
]

mapping = dict(
    mratio=r"$\mathbf{M_{ratio}}$",
    mratio_con2=r"bounded $\mathbf{M_{ratio}}$",
    mratio_bounded=r"bounded $\mathbf{M_{ratio}}$",
    mratio_logistic=r"logistic $\mathbf{M_{ratio}}$",
    logmratio=r"log$\,\mathbf{M_{ratio}}$",
    mratio_hmeta=r"hierarchical $\mathbf{M_{ratio}}$",
)

def as_latex(model, title=None, print_meta=True, single_table=False):
    '''Generate LaTeX Summary Table
    '''
    tables = model.tables
    settings = model.settings

    if title is not None:
        title = r'\caption*{' + title + '}'

    simple_tables = _simple_tables(tables, settings)
    if not print_meta:
        simple_tables = simple_tables[1:]
    tab = [x.as_latex_tabular() for x in simple_tables]
    tab = '\n\\hline\n'.join(tab)

    to_replace = ('\\\\hline\\n\\\\hline\\n\\\\'
                  'end{tabular}\\n\\\\begin{tabular}{.*}\\n')

    if model._merge_latex:
        # create single tabular object for summary_col
        tab = re.sub(to_replace, r'\\midrule\n', tab)

    if title is not None:
        out = '\\begin{table}', title, tab, '\\end{table}'
    else:
        out = '\\begin{table}', tab, '\\end{table}'
    out = '\n'.join(out)
    if single_table:
        out = out.replace('\\end{tabular}\n\\begin{tabular}{lrrrrrr}\n', '')
    return out

DV_mapping = dict(
    pearson_r_ztrans='arctanh(Pearson $r$)',
    NMAE='NMAE'
)

for DV in ('pearson_r_ztrans', 'NMAE'):
    for comparison in comparisons:
        print(f'reg_{comparison[0]}_vs_{comparison[1]}_{DV}.png')
        data = df[df.measure_name.isin(comparison)].copy()
        data.loc[data.measure_name == comparison[0], 'measure'] = 1
        data.loc[data.measure_name == comparison[1], 'measure'] = 0
        # for col in [col for col in data.columns if col not in ('study, ')]:
        #     if data[col].dtype.kind != 'O':
        #         data[col] -= data[col].mean()
        #         data[col] /= data[col].std()
        print(f'\t{comparison}')
        model = sm.MixedLM.from_formula(f'{DV} ~ measure + type1_perf + ntrials', groups='study', data=data).fit()
        print('\t', model.summary())
        print(model.params.measure)
        beginningtex = r"""\documentclass[preview]{standalone}\thispagestyle{empty}\usepackage{booktabs}\usepackage[font=bf,aboveskip=0pt]{caption}\begin{document}"""
        endtex = r"\end{document}"
        title = f'DV: {DV_mapping[DV]}'
        content = as_latex(model.summary(), print_meta=False, single_table=True, title=title)
        latex = beginningtex + content + endtex

        f = open('latex/document.tex', 'w')
        f.write(latex)
        f.close()
        os.system('pdflatex -output-directory latex latex/document.tex  > /dev/null 2>&1 && pdfcrop latex/document.pdf latex/document.pdf > /dev/null 2>&1 && pdftoppm -r 300 latex/document.pdf|pnmtopng > latex/document.png')
        os.rename('latex/document.png', f'../data/reg/reg_{comparison[0]}_vs_{comparison[1]}_{DV}.png')
