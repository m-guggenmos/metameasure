import re
import warnings
from collections import namedtuple
from functools import partial

import numpy as np
import pandas as pd
import pingouin as pg
import statsmodels.api as sm
from scipy.stats import pearsonr, linregress, theilslopes, zscore

import matplotlib.pyplot as plt

pd.options.mode.chained_assignment = None



def nbprint(text, fontsize=13, **kwargs):
    fig, ax = plt.subplots(figsize=(1, 0.03))
    plt.text(0, 0, text, transform=ax.transAxes,
             fontsize=fontsize, **kwargs)
    ax.axis('off')
    plt.show()


def linear_regression(data, patsy_string, standardize_vars=True, print_summary=True,
                      model_blocks=None, ignore_warnings=True,
                      print_patsy=True, print_corr_table=True, print_data=False, print_extra=True,
                      vars_corr=None,
                      reml=True,
                      notebook_print=False,
                      silent=False, print_short=False, ols=False,
                      standardize_vars_excl=(),
                      groupname='subject', return_model=True, return_data=False):

    if model_blocks is None:
        model_blocks = 'block' in data

    DV = patsy_string.split('~')[0].strip()
    IVs = list(dict.fromkeys([iv.split('(')[1].split(')')[0] if '(' in iv else iv.strip() for iv in patsy_string.split('~')[1].replace('*', '+').replace(':', '+').replace('-', '+').split('+')]))

    allVars = [DV] + IVs + [groupname]
    if model_blocks:
        allVars += ['block']

    dtypes = dict()
    for var in allVars:
        if isinstance(data[var].dtype, (pd.Int64Dtype, object)):
            dtypes.update({var: float})
        if data[var].dtype == np.dtype('bool'):
            dtypes.update({var: float})
    data = data.astype(dtypes).reset_index()

    if standardize_vars:
        for var in allVars:
            if not var in ([groupname] if 'block' in patsy_string else [groupname, 'block']) and not var in standardize_vars_excl:
                data[var] = ((data[var] - data[var].mean()) / data[var].std()).values

    if return_data:
        return data

    print_ = partial(nbprint, family='monospace') if notebook_print else print

    if not silent and print_data:
        print_(data[allVars])
        for v in allVars:
            print_({v: data[v].dtype})

    if not silent and print_corr_table:
        vars = [DV] + IVs if vars_corr is None else vars_corr
        corrtab = data[vars].corr()
        pcorrtab = data[vars].corr(method=lambda x, y: pearsonr(x, y)[1])
        corrtab_sorted = dict(sorted({k:v for el in [{f'{c1} x {c2}': corrtab.loc[c1, c2] for c2 in corrtab.columns if list(corrtab.columns).index(c2) > list(corrtab.columns).index(c1)} for c1 in corrtab.columns] for k,v in el.items()}.items(), key=lambda x: x[1]))
        pcorrtab_sorted = dict(sorted({k:v for el in [{f'{c1} x {c2}': pcorrtab.loc[c1, c2] for c2 in pcorrtab.columns if list(pcorrtab.columns).index(c2) > list(pcorrtab.columns).index(c1)} for c1 in pcorrtab.columns] for k,v in el.items()}.items(), key=lambda x: x[1]))
        for k, v in corrtab_sorted.items():
            print_(f'{k}: {v:.3f} (p={pcorrtab_sorted[k]:.4f})')

    with warnings.catch_warnings():
        if ignore_warnings:
            warnings.simplefilter('ignore')
        if model_blocks:
            model = sm.MixedLM.from_formula(patsy_string, groups=groupname, re_formula='1', vc_formula={'block': '0+C(block)'}, data=data).fit(reml=reml)
        elif ols:
            model = sm.OLS.from_formula(patsy_string, data=data).fit(reml=reml)
        else:
            model = sm.MixedLM.from_formula(patsy_string, groups=groupname, data=data).fit(reml=reml)
        if not silent:
            if print_patsy:
                print_(patsy_string + '\n')
            if print_summary:
                if print_short:
                    a = str(model.summary())
                    if ols:
                        pos = np.array([v.start() for v in re.finditer('=', a)])
                        pos_parts = np.where(np.diff(pos) > 1)[0]
                        print_('='*80 + '\n' + a[pos[pos_parts[1]]+2:pos[pos_parts[2]]+1] + '==')
                    else:
                        pos2 = np.array([v.start() for v in re.finditer('--', a)])
                        pos_parts2 = np.where(np.diff(pos2) > 2)[0]
                        print_(a[pos2[pos_parts2[0]]+2:])
                else:
                    print_(model.summary())
                    if print_extra and (model.aic is not None):
                        print_(f'AIC: {model.aic} BIC: {model.bic}')

    predictions = model.predict(data)
    r, p = pearsonr(predictions, data[DV])
    if not silent and not print_short and print_extra:
        print_(f'Pearson r = {r} (p = {p})')

        if len(data[~data[DV].isna()][DV].unique()) == 2:
            print_(f'Accuracy: {np.mean((data[~data[DV].isna()][DV].values > 0) == (predictions > 0))}')


    if return_model:
        return model

def regress(x, y, method='linregress', return_outliers=False, outlier_stds=3):

    Result = namedtuple('Result', 'slope p r intercept')

    outliers = None
    if method == 'linregress':
        lr = linregress(x, y)
        result = Result(slope=lr.slope, p=lr.pvalue, r=lr.rvalue, intercept=lr.intercept)

    if method == 'outlier_x':
        outliers = np.abs(zscore(x)) > outlier_stds
        lr = linregress(x[~outliers], y[~outliers])
        result = Result(slope=lr.slope, p=lr.pvalue, r=lr.rvalue, intercept=lr.intercept)

    if method == 'bivariate':
        model = sm.OLS(y, x).fit()
        outliers = np.abs(zscore(model.resid)) > outlier_stds
        if (len(np.unique(x[~outliers])) == 1) and (len(np.unique(y[~outliers])) == 1):
            outliers = np.full(len(x), False)
        lr = linregress(x[~outliers], y[~outliers])
        result = Result(slope=lr.slope, p=lr.pvalue, r=lr.rvalue, intercept=lr.intercept)

    if method == 'outlier_xy':
        if (len(np.unique(x)) == 1) and (len(np.unique(y)) == 1):
            outliers = np.full(len(x), False)
        elif (len(np.unique(x)) == 1):
            outliers = (np.abs(zscore(y)) > outlier_stds)
        elif (len(np.unique(y)) == 1):
            outliers = (np.abs(zscore(x)) > outlier_stds)
        else:
            outliers = (np.abs(zscore(x)) > outlier_stds) | (np.abs(zscore(y)) > outlier_stds)
        lr = linregress(x[~outliers], y[~outliers])
        result = Result(slope=lr.slope, p=lr.pvalue, r=lr.rvalue, intercept=lr.intercept)

    elif method in ('shepherd', 'skipped'):
        if method == 'shepherd':
            if (len(np.unique(x)) == 1) or (len(np.unique(y)) == 1):
                outliers = np.full(len(x), False)
            else:
                with warnings.catch_warnings():
                    warnings.filterwarnings('error', category=RuntimeWarning)
                    try:
                        r, p, outliers = pg.correlation.shepherd(x, y)
                    except Warning as e:
                        print('\tNo outliers are removed due to a warning in shepherd correlation:', e)
                        outliers = np.full(len(x), False)
        elif method == 'skipped':
            with warnings.catch_warnings():
                warnings.filterwarnings('error', category=RuntimeWarning)
                try:
                    r, p, outliers = pg.correlation.skipped(x, y)
                except Warning as e:
                    print('\tNo outliers are removed due to a warning in skipped correlation:', e)
                    outliers = np.full(len(x), False)
        lr = linregress(x[~outliers], y[~outliers])
        # r, p = spearmanr(x[~outliers], y[~outliers])
        result = Result(slope=lr.slope, p=lr.pvalue, r=lr.rvalue, intercept=lr.intercept)

    # elif method == 'ransac':
    #     ransac = RANSACRegressor().fit(x.reshape(-1, 1), y)
    #     lr = linregress(x[ransac.inlier_mask_], y[ransac.inlier_mask_])
    #     result = Result(slope=lr.slope, p=lr.pvalue, r=lr.rvalue)

    elif method == 'theil':
        if (len(np.unique(x)) == 1) or (len(np.unique(y)) == 1):
            lr = linregress(x, y)
            result = Result(slope=lr.slope, p=lr.pvalue, r=None, intercept=lr.intercept)
            warnings.warn('x and/or y consist of only one unique value. Switching to linregress.', RuntimeWarning)
        else:
            lr = theilslopes(x, y)
            # compute p-value based on CI, see https://www.bmj.com/content/343/bmj.d2304
            SE = (lr[3] - lr[2]) / (2 * 1.96)
            z = lr[0] / SE
            pval = np.exp(-0.717 * z - 0.416 * z**2)
            result = Result(slope=lr[0], p=pval, r=None, intercept=lr[1])

    if return_outliers:
        return result, outliers
    else:
        return result




if __name__ == '__main__':
    x = np.random.rand(100)
    y = np.random.rand(100)
    np.random.seed(3)
    result = regress(x, y, method='linregress')
    # result = regress(x, y, method='theil')
    print(result)