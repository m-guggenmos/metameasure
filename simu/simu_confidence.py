#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Author: Matthias Guggenmos <mg.corresponding@gmail.com>
# 2021

# Simulation: generative model

from pathlib import Path
from scipy.stats import beta, lognorm, norm, uniform

import numpy as np
from fast_truncnorm import truncnorm

cwd = Path.cwd()


def conf(x, bounds_):
    confidence_ = np.full(x.shape, np.nan)
    bounds_ = np.hstack((bounds_, np.inf))
    for i, b in enumerate(bounds_[:-1]):
        confidence_[(bounds_[i] <= x) & (x < bounds_[i + 1])] = i + 1
    return confidence_


bounds = np.arange(0, 0.81, 0.2)


def _lognorm_params(mode, stddev):
    a = stddev**2 / mode**2
    x = 1/4*np.sqrt(np.maximum(1e-300, -(16*(2/3)**(1/3)*a)/(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3) +
                               2*(2/3)**(2/3)*(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3)+1)) + \
        1/2*np.sqrt((4*(2/3)**(1/3)*a)/(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3) -
                    (np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3)/(2**(1/3)*3**(2/3)) +
                    1/(2*np.sqrt(np.maximum(1e-300, -(16*(2/3)**(1/3)*a)/(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3) +  # noqa
                                            2*(2/3)**(2/3)*(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3)+1)))+1/2) + 1/4  # noqa
    shape = np.sqrt(np.log(x))
    scale = mode * x  # scale = np.exp(mu) -> mu = np.log(mode * x)
    return shape, scale


def gen_data(nsubjects, nsamples, sigmas_sens=0.1, sigma_meta=0, mu=0.5, noise_model='beta'):

    stimulus = np.random.randint(0, 2, (nsubjects, nsamples))
    percept = ((mu/2) * (2*(stimulus-0.5))) + sigmas_sens * np.random.randn(nsamples)

    choice_prob = 1 / (1 + np.exp(-mu * percept / (max(1e-3, sigmas_sens**2))))

    choice = (choice_prob > 0.5).astype(int)
    posterior = np.full((nsubjects, nsamples), np.nan)
    posterior[choice == 1] = choice_prob[choice == 1]
    posterior[choice == 0] = 1 - choice_prob[choice == 0]
    confidence = 2*(posterior - 0.5)

    if hasattr(sigma_meta, '__len__'):
        sigma_meta = np.array(sigma_meta).reshape(-1, 1)

    if np.any(sigma_meta > 0):
        if noise_model == 'beta':
            a = confidence * (1 / sigma_meta - 2) + 1
            b = (1 - confidence) * (1 / sigma_meta - 2) + 1
            confidence = beta(a, b).rvs()
        elif noise_model == 'censored_norm':
            confidence = np.maximum(0, np.minimum(1, norm(loc=confidence, scale=sigma_meta).rvs()))
        elif noise_model == 'truncated_norm':
            confidence = truncnorm(-confidence / sigma_meta, (1 - confidence) / sigma_meta, loc=confidence,
                                   scale=sigma_meta).rvs()
        elif noise_model == 'censored_lognorm':
            shape, scale = _lognorm_params(np.maximum(1e-5, confidence), sigma_meta)
            confidence = np.minimum(1, lognorm(loc=0, scale=scale, s=shape).rvs())
        elif noise_model == 'truncated_lognorm':
            shape, scale = _lognorm_params(np.maximum(1e-5, confidence), sigma_meta)
            confidence = truncated_lognorm(loc=0, scale=scale, s=shape, b=1).rvs()

    confidence_disc = conf(confidence, bounds)
    correct = (stimulus == choice).astype(int)

    return stimulus, choice, correct, confidence, confidence_disc, len(bounds)


class truncated_lognorm:  # noqa
    """
    Implementation of the truncated lognormal distribution.
    Only the upper truncation bound is supported as the lognormal distribution is naturally lower-bounded at zero.

    Parameters
    ----------
    loc : float or array-like
        Scipy lognorm's loc parameter.
    scale : float or array-like
        Scipy lognorm's scale parameter.
    s : float or array-like
        Scipy lognorm's s parameter.
    b : float or array-like
        Upper truncation bound.
    """
    def __init__(self, loc, scale, s, b):
        self.loc = loc
        self.scale = scale
        self.s = s
        self.b = b
        self.dist = lognorm(loc=loc, scale=scale, s=s)
        self.lncdf_b = self.dist.cdf(self.b)

    def pdf(self, x):
        pdens = (x <= self.b) * self.dist.pdf(x) / self.lncdf_b
        return pdens

    def cdf(self, x):
        cdens = (x > self.b) + (x <= self.b) * self.dist.cdf(x) / self.lncdf_b
        return cdens

    def rvs(self, size=None):
        if size is None:
            if hasattr(self.scale, '__len__'):
                size = self.scale.shape
            else:
                size = 1
        cdens = uniform(loc=0, scale=self.b).rvs(size)
        x = self.cdf_inv(cdens)
        return x

    def cdf_inv(self, cdens):
        x = (cdens >= 1) * self.b + (cdens < 1) * self.dist.ppf(cdens * self.lncdf_b)
        return x
