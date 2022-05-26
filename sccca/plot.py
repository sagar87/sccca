import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize


def plot_factor(factor, var, lowest=4, highest=3, fat_bar=0.6, thin_bar=0.01, ax=None):
    y = factor
    other = len(factor) - (lowest + highest)
    factor_idx = np.argsort(factor)
    w = np.concatenate(
        [
            np.ones(lowest) * fat_bar,
            np.ones(other) * thin_bar,
            np.ones(highest) * fat_bar,
        ]
    )

    norm = Normalize(vmin=factor.min(), vmax=factor.max(), clip=True)
    mapper = cm.ScalarMappable(norm=norm, cmap=plt.get_cmap("RdBu"))

    colors = [mapper.to_rgba(v) for v in y[factor_idx]]
    xticks = []
    for n, c in enumerate(w):
        xticks.append(sum(w[:n]) + w[n] / 2)

    # xticks_labels = (
    #     var[factor_idx].values[:lowest].tolist()
    #     + [""] * other
    #     + var[factor_idx[::-1]].values[:highest].tolist()
    # )
    # w_new = [i / max(w) for i in w]

    if ax is None:
        plt.bar(xticks, height=y[factor_idx], width=w, color=colors, alpha=0.9)
        ax = plt.gca()
    else:
        ax.bar(xticks, height=y[factor_idx], width=w, color=colors, alpha=0.9)

    ax.set_xticks([])
    # _ = ax.set_xticklabels(xticks_labels, rotation=90)
    ax.margins(x=0.01)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    for name, xtick in zip(var[factor_idx].values[:lowest].tolist(), xticks[:lowest]):
        ax.text(
            x=xtick,
            y=-0.1,
            s=name,
            rotation=90,
            ha="center",
            color="white",
            va="top",
            fontweight="bold",
        )

    for name, xtick in zip(var[factor_idx].values[-highest:].tolist(), xticks[-highest:]):
        ax.text(
            x=xtick,
            y=0.15,
            s=name,
            rotation=90,
            ha="center",
            color="white",
            va="bottom",
            fontweight="bold",
        )
    return ax
