"""
Produce dataset-wide plots.
"""

import numpy as np
import pandas as pd
from sharp.utils._utils import _optional_import
from sharp.utils import check_feature_names, scores_to_ordering


def strata_boxplots(
    X,
    y,
    contributions,
    feature_names=None,
    n_strata=5,
    gap_size=1,
    cmap="Pastel1",
    ax=None,
    show=False,
    **kwargs,
):

    plt = _optional_import("matplotlib.pyplot")

    if feature_names is None:
        feature_names = check_feature_names(X)

    if ax is None:
        fig, ax = plt.subplots()

    df = pd.DataFrame(contributions, columns=feature_names)

    perc_step = 100 / n_strata
    stratum_size = X.shape[0] / n_strata

    df["target"] = scores_to_ordering(y, -1)
    df["target_binned"] = [
        (
            f"0-\n{int(perc_step)}%"
            if np.floor((rank - 1) / stratum_size) == 0
            else str(int(np.floor((rank - 1) / stratum_size) * perc_step))
            + "-\n"
            + str(int((np.floor((rank - 1) / stratum_size) + 1) * perc_step))
            + "%"
        )
        for rank in df["target"]
    ]
    df.sort_values(by=["target_binned"], inplace=True)
    df.drop(columns=["target"], inplace=True)

    df["target_binned"] = df["target_binned"].str.replace("<", "$<$")

    colors = [plt.get_cmap(cmap)(i) for i in range(len(feature_names))]
    bin_names = df["target_binned"].unique()
    pos_increment = 1 / (len(feature_names) + gap_size)
    boxes = []
    for i, bin_name in enumerate(bin_names):
        box = ax.boxplot(
            df[df["target_binned"] == bin_name][feature_names],
            widths=pos_increment,
            positions=[i + pos_increment * n for n in range(len(feature_names))],
            patch_artist=True,
            medianprops={"color": "black"},
            boxprops={"facecolor": "C0", "edgecolor": "black"},
            **kwargs,
        )
        boxes.append(box)

    for box in boxes:
        patches = []
        for patch, color in zip(box["boxes"], colors):
            patch.set_facecolor(color)
            patches.append(patch)

    ax.set_xticks(
        np.arange(0, len(bin_names)) + pos_increment * (len(feature_names) - 1) / 2,
        bin_names,
    )

    ax.legend(
        patches,
        feature_names,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=len(feature_names),
    )

    if show:
        plt.show()
    else:
        return ax
