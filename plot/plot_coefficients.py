from collections import namedtuple

import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def pearsonr_ci(x, y, alpha=0.05):
    ''' calculate Pearson correlation along with the confidence interval using scipy and numpy
    Parameters
    ----------
    x, y : iterable object such as a list or np.array
      Input for correlation calculation
    alpha : float
      Significance level. 0.05 by default
    Returns
    -------
    r : float
      Pearson's correlation coefficient
    pval : float
      The corresponding p value
    SE : float
      The corresponding standard error of the correlation coefficient
    lo, hi : float
      The lower and upper bound of confidence intervals
    '''

    x_ = np.array(x)[~np.isnan(x) & ~np.isnan(y)]
    y_ = np.array(y)[~np.isnan(x) & ~np.isnan(y)]

    r, p = stats.pearsonr(x_, y_)
    r_z = np.arctanh(r)
    se_ztransform = 1 / np.sqrt(len(x_)-3)
    z = stats.norm.ppf(1 - alpha / 2)
    lo_z, hi_z = r_z - z * se_ztransform, r_z + z * se_ztransform
    lo, hi = np.tanh((lo_z, hi_z))
    # SE = (hi - lo) / (2 * 1.96)
    # For the below SE equation, see: Cohen, J., & Cohen, J. (Eds.). (2003). Applied multiple regression/correlation analysis for the behavioral sciences (3rd ed). Mahwah, N.J: L. Erlbaum Associates.
    SE = np.sqrt((1 - r**2) / (len(x_) - 2))

    Result = namedtuple('Result', 'r p SE CI_lower CI_higher')

    return Result(r=r, p=p, SE=SE, CI_lower=lo, CI_higher=hi)


def plot_coefficients(model, data, exclude_categorical=False, colors_reg=None, colors_corr=None,
                                  striped_background=False, stripe_color_light=(1, 1, 1), stripe_color_dark=(0.75, 0.75, 0.75),
                                  asterisk_yshift=0, circle_yshift=0,
                                  asterisk_fontsize=12, circle_fontsize=14,
                                  mapping=None, min_x_asterisks=-np.inf, max_x_asterisks=np.inf,
                                  show_trend=True):

    params = model.params[1:]
    se = model.bse[1:]
    pval = model.pvalues[1:]
    param_names = params.index
    if exclude_categorical:
        param_names = [p for p in params.index if not p.startswith('C(')]
        params = params[[p for p in params.index if not p.startswith('C(')]]
        se = se[[p for p in se.index if not p.startswith('C(')]]
        pval = pval[[p for p in se.index if not p.startswith('C(')]]
    nparams = len(params)

    DV = model.model.endog_names
    r = np.full(nparams, np.nan)
    r_pval = np.full(nparams, np.nan)
    r_SE = np.full(nparams, np.nan)
    for i, p in enumerate(param_names):
        corrstats = pearsonr_ci(data[p].values, data[DV].values)
        r[i] = corrstats.r
        r_pval[i] = corrstats.p
        r_SE[i] = corrstats.SE


    if colors_reg is None:
        colors_reg = [(0.36, 0.42, 0.62)]*nparams
    else:
        colors_reg = [colors_reg]*nparams
    if colors_corr is None:
        colors_corr = [(0.51, 0.71, 0.42)]*nparams
    else:
        colors_corr = [colors_corr]*nparams

    if striped_background:
        for i in range(nparams+1):
            plt.axhspan(i-0.5, i+0.5, facecolor=[stripe_color_light, stripe_color_dark][int(np.mod(i, 2)==0)], alpha=0.1)

    for i, (param, val) in enumerate(params.iteritems()):

        plt.barh(nparams - i - 1 + 0.25, r[i], fc=colors_corr[i], xerr=r_SE[i], height=0.5, label='Bivariate\ncorrelation' if i == 0 else None)
        plt.barh(nparams - i - 1 - 0.25, val, fc=colors_reg[i], xerr=se[i], height=0.5, label='Multinomial\nregression' if i == 0 else None)
        if (r_pval[i] < 0.1) and not ((r_pval[i] >= 0.05) and not show_trend):
            asterisks = ('◦', '*', '**', '***')[int(r_pval[i] < 0.05) + int(r_pval[i] < 0.01) + int(r_pval[i] < 0.001)]
            yshift = (asterisk_yshift, circle_yshift)[int(r_pval[i] >= 0.05)]
            fontsize = (asterisk_fontsize, circle_fontsize)[int(r_pval[i] >= 0.05)]
            boldfont = (None, 'bold')[int(r_pval[i] >= 0.05)]
            plt.text((max(r[i]-r_SE[i]-0.01, min_x_asterisks), min(r[i]+r_SE[i]+0.01, max_x_asterisks))[int(r[i]>0)], nparams - i + 0.25 - 1 - 0.22 + yshift + 0.3*(((r[i]-r_SE[i]+0.01) > max_x_asterisks) | ((r[i]-r_SE[i]+0.01) < min_x_asterisks)), asterisks, ha=('right', 'left')[int(r[i]>0)], fontsize=fontsize, fontweight=boldfont)
        if (pval[i] < 0.1) and not ((pval[i] >= 0.05) and not show_trend):
            asterisks = ('◦', '*', '**', '***')[int(pval[i] < 0.05) + int(pval[i] < 0.01) + int(pval[i] < 0.001)]
            yshift = (asterisk_yshift, circle_yshift)[int(pval[i] >= 0.05)]
            fontsize = (asterisk_fontsize, circle_fontsize)[int(pval[i] >= 0.05)]
            boldfont = (None, 'bold')[int(pval[i] >= 0.05)]
            plt.text((max(val-se[i]-0.01, min_x_asterisks), min(val+se[i]+0.01, max_x_asterisks))[int(val>0)], nparams - i - 1 - 0.25 - 0.22 + yshift + 0.3*(((val+se[i]+0.01) > max_x_asterisks) | ((val+se[i]+0.01) < min_x_asterisks)), asterisks, ha=('right', 'left')[int(val>0)], fontsize=fontsize, fontweight=boldfont)

    if mapping is None:
        yticklabels = params.index[::-1]
    else:
        yticklabels = [mapping[p] for p in params.index[::-1]]

    plt.yticks(range(nparams), yticklabels)
    plt.ylim((-0.5, nparams-0.5))
