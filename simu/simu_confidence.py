from pathlib import Path
from scipy.stats import beta, lognorm, norm

import numpy as np

cwd = Path.cwd()


def conf(x, bounds):
    confidence = np.full(x.shape, np.nan)
    bounds = np.hstack((bounds, np.inf))
    for i, b in enumerate(bounds[:-1]):
        confidence[(bounds[i] <= x) & (x < bounds[i + 1])] = i + 1
    return confidence
bounds = np.arange(0, 0.81, 0.2)


def _lognorm_params(mode, stddev):
    a = stddev**2 / mode**2
    x = 1/4*np.sqrt(np.maximum(1e-10, -(16*(2/3)**(1/3)*a)/(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3) +
                               2*(2/3)**(2/3)*(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3)+1)) + \
        1/2*np.sqrt((4*(2/3)**(1/3)*a)/(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3) -
                    (np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3)/(2**(1/3)*3**(2/3)) +
                    1/(2*np.sqrt(np.maximum(1e-10, -(16*(2/3)**(1/3)*a)/(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3) +
                                            2*(2/3)**(2/3)*(np.sqrt(3)*np.sqrt(256*a**3+27*a**2)-9*a)**(1/3)+1)))+1/2) + \
        1/4
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
        elif noise_model == 'lognorm':
            shape, scale = _lognorm_params(confidence, sigma_meta)
            confidence = lognorm(loc=0, scale=scale, s=shape).rvs()

    confidence_disc = conf(confidence, bounds)
    correct = (stimulus == choice).astype(int)

    return stimulus, choice, correct, confidence, confidence_disc, len(bounds)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    sigma_sens_list = np.arange(0.1, 10, 0.2)
    nsigma = len(sigma_sens_list)
    nsamples = 10000
    correct, confidence = np.full((nsigma, nsamples), np.nan), np.full((nsigma, nsamples), np.nan)
    for i, sigma_sens in enumerate(sigma_sens_list):
        stimulus, choice, correct[i], confidence[i] = gen_data(1, nsamples, sigmas_sens=sigma_sens)[:4]

    plt.figure()
    plt.plot(sigma_sens_list, correct.mean(axis=1))
    plt.plot(sigma_sens_list, confidence.mean(axis=1))