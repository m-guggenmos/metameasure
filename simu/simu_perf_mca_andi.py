import os
import sys
from multiprocessing_on_dill.pool import Pool
from timeit import default_timer
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.append(os.path.join(os.path.dirname(__file__), os.path.pardir))
from eval.andi import andi_disc  # noqa
from simu_confidence import gen_data

np.random.seed(1)

# Ns
####

nsamples = 10000
nsubjects = 1000
# nsamples = 100
# nsubjects = 3

mu = 0.5
sigma_sens_list = [0, 0.0005]
for i in range(98):
    sigma_sens_list += [sigma_sens_list[-1] + 1.0681355555555*(sigma_sens_list[-1] - sigma_sens_list[-2])]
print(sigma_sens_list)
nsigmasens = len(sigma_sens_list)
noise_model = 'beta'
# noise_model = 'lognorm'
# noise_model = 'censored_norm'

sigma_meta_list = dict(
    beta=np.arange(0, 0.51, 0.1),
    lognorm=np.exp(np.arange(0, 4, 0.2))-1,
    censored_norm=np.arange(0, 2, 0.1)
)[noise_model]
nsigmameta = len(sigma_meta_list)


def loop(i):
    np.random.seed(i)
    sigma_sens = sigma_sens_list[i]

    t0 = default_timer()

    df = pd.DataFrame(index=range(nsubjects*nsigmameta))
    df['subject'] = np.tile(range(nsubjects), nsigmameta)
    df['sigma_meta_id'] = np.repeat(range(nsigmameta), nsubjects)
    df['sigma_sens_id'] = i
    df['sigma_sens'] = sigma_sens

    for j, sigma_meta in enumerate(sigma_meta_list):
        stimulus, choices, correct, confidence, confidence_disc, nratings = gen_data(nsubjects, nsamples, sigmas_sens=sigma_sens, sigma_meta=sigma_meta, mu=mu, noise_model=noise_model)
        for s in range(nsubjects):
            cond = (df.subject == s) & (df.sigma_meta_id == j)
            df.loc[cond, 'andi'] = andi_disc(stimulus[s], choices[s], confidence_disc[s] - 1, nratings=nratings)

    print(f'\tFinished parameter {i + 1} / {nsigmasens}: {default_timer() - t0:.1f} secs')

    return df

def main():
    # with Pool(8) as pool:
    #     results = list(pool.map(loop, range(nsigmasens)))
    results = [None] * nsubjects
    for s in range(nsubjects):
        results[s] = loop(s)
    result = pd.concat(results).reset_index(drop=True)

    return result

if __name__ == '__main__':
    result = main()
    result.to_pickle(f'../data/{Path(__file__).stem}_{noise_model}.pkl', protocol=4)
