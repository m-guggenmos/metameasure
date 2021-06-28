import os
from pathlib import Path

import matplotlib.pyplot as plt

from plot_util import savefig

comparisons = [
    ('mratio_bounded', 'mratio'),
    ('logmratio', 'mratio'),
    ( 'mratio_logistic', 'mratio'),
    ('mratio_hmeta', 'mratio'),
]

mapping = dict(
    mratio=r"$\mathbf{M_{ratio}}$",
    mratio_con2=r"bounded $\mathbf{M_{ratio}}$",
    mratio_bounded=r"bounded $\mathbf{M_{ratio}}$",
    mratio_logistic=r"logistic $\mathbf{M_{ratio}}$",
    logmratio=r"log$\,\mathbf{M_{ratio}}$",
    mratio_hmeta=r"hierarchical $\mathbf{M_{ratio}}$",
)

plt.figure(figsize=(8.5, 20/3))

for i, comparison in enumerate(comparisons):
    for j, measure in enumerate(comparison):
        if not ('mratio_hmeta' in comparison and (j == 1)):
            ax = plt.subplot(4, 2, i*2 + j + 1)
            plt.axis("off")
            if j == 0:
                title = plt.title(f'{mapping[comparison[0]]} $>$ {mapping[comparison[1]]}', fontsize=14)
                title.set_position((1.03, 1))
            img = plt.imread(f"../data/reg/reg_{comparison[0]}_vs_{comparison[1]}_{('pearson_r_ztrans', 'NMAE')[j]}.png")
            plt.imshow(img, cmap='Greys_r')

plt.subplots_adjust(left=0, right=1, hspace=0.3, wspace=0.02, top=0.97, bottom=0.01)

savefig(f'../img/{Path(__file__).stem}.png')