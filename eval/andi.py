import numpy as np

def andi_disc(stimuli, choices, confidence, nratings=6):
    # confidence ratings should start at zero!

    n = len(stimuli)
    dm = np.minimum(0.999, np.mean(choices == stimuli))

    d_j = np.full(nratings, np.nan)
    n_j = np.full(nratings, np.nan)
    for j in range(nratings):
        n_j[j] = np.sum(confidence == j)
        if n_j[j] > 0:
            d_j[j] = np.mean(choices[confidence == j] == stimuli[confidence == j])

    DI = np.nansum(n_j * (d_j - dm) ** 2) / n
    NDI = DI / (dm * (1 - dm))
    ANDI = (n * NDI - nratings + 1) / (n - nratings + 1)
    # print(ANDI)

    return ANDI