import pandas as pd
import numpy as np
from pathlib import Path
import seaborn as sns
import matplotlib
import matplotlib.pyplot as plt
import requests as rq
from io import BytesIO
from math import comb

import shap

from itertools import combinations


from shap.utils import format_value, safe_isinstance
from shap.plots import colors

import sys

sys.path.append("../")

from .plots import importance_plot, importance_plot_test
from .qoi import RankingRank, RankingTopK, RankingScore, RankingNoFunction
from .ranking import Ranking


def waterfall(shap_values, max_display=10, show=True):
    # Turn off interactive plot
    if show is False:
        plt.ioff()

    base_values = float(shap_values["base_values"])
    features = shap_values["values"]
    feature_names = shap_values["feature_names"]
    # lower_bounds = shap_values["lower_bounds"]
    # upper_bounds = shap_values["upper_bounds"]
    lower_bounds = None
    upper_bounds = None
    values = shap_values["values"]

    # init variables we use for tracking the plot locations
    num_features = min(max_display, len(values))
    row_height = 0.5
    rng = range(num_features - 1, -1, -1)
    order = np.argsort(-np.abs(values))
    pos_lefts = []
    pos_inds = []
    pos_widths = []
    pos_low = []
    pos_high = []
    neg_lefts = []
    neg_inds = []
    neg_widths = []
    neg_low = []
    neg_high = []
    loc = base_values + values.sum()
    yticklabels = ["" for _ in range(num_features + 1)]

    # size the plot based on how many features we are plotting
    plt.gcf().set_size_inches(8, num_features * row_height + 1.5)

    # see how many individual (vs. grouped at the end) features we are plotting
    if num_features == len(values):
        num_individual = num_features
    else:
        num_individual = num_features - 1

    # compute the locations of the individual features and plot the dashed connecting lines
    for i in range(num_individual):
        sval = values[order[i]]
        loc -= sval
        if sval >= 0:
            pos_inds.append(rng[i])
            pos_widths.append(sval)
            if lower_bounds is not None:
                pos_low.append(lower_bounds[order[i]])
                pos_high.append(upper_bounds[order[i]])
            pos_lefts.append(loc)
        else:
            neg_inds.append(rng[i])
            neg_widths.append(sval)
            if lower_bounds is not None:
                neg_low.append(lower_bounds[order[i]])
                neg_high.append(upper_bounds[order[i]])
            neg_lefts.append(loc)
        if num_individual != num_features or i + 4 < num_individual:
            plt.plot(
                [loc, loc],
                [rng[i] - 1 - 0.4, rng[i] + 0.4],
                color="#bbbbbb",
                linestyle="--",
                linewidth=0.5,
                zorder=-1,
            )
        if features is None:
            yticklabels[rng[i]] = feature_names[order[i]]
        else:
            if np.issubdtype(type(features[order[i]]), np.number):
                yticklabels[rng[i]] = (
                    format_value(float(features[order[i]]), "%0.03f")
                    + " = "
                    + feature_names[order[i]]
                )
            else:
                yticklabels[rng[i]] = (
                    str(features[order[i]]) + " = " + str(feature_names[order[i]])
                )

    # add a last grouped feature to represent the impact of all the features we didn't show
    if num_features < len(values):
        yticklabels[0] = "%d other features" % (len(values) - num_features + 1)
        remaining_impact = base_values - loc
        if remaining_impact < 0:
            pos_inds.append(0)
            pos_widths.append(-remaining_impact)
            pos_lefts.append(loc + remaining_impact)
        else:
            neg_inds.append(0)
            neg_widths.append(-remaining_impact)
            neg_lefts.append(loc + remaining_impact)

    points = (
        pos_lefts
        + list(np.array(pos_lefts) + np.array(pos_widths))
        + neg_lefts
        + list(np.array(neg_lefts) + np.array(neg_widths))
    )
    dataw = np.max(points) - np.min(points)

    # draw invisible bars just for sizing the axes
    label_padding = np.array([0.1 * dataw if w < 1 else 0 for w in pos_widths])
    plt.barh(
        pos_inds,
        np.array(pos_widths) + label_padding + 0.02 * dataw,
        left=np.array(pos_lefts) - 0.01 * dataw,
        color=colors.red_rgb,
        alpha=0,
    )
    label_padding = np.array([-0.1 * dataw if -w < 1 else 0 for w in neg_widths])
    plt.barh(
        neg_inds,
        np.array(neg_widths) + label_padding - 0.02 * dataw,
        left=np.array(neg_lefts) + 0.01 * dataw,
        color=colors.blue_rgb,
        alpha=0,
    )

    # define variable we need for plotting the arrows
    head_length = 0.08
    bar_width = 0.8
    xlen = plt.xlim()[1] - plt.xlim()[0]
    fig = plt.gcf()
    ax = plt.gca()
    bbox = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    width = bbox.width
    bbox_to_xscale = xlen / width
    hl_scaled = bbox_to_xscale * head_length
    renderer = fig.canvas.get_renderer()

    # draw the positive arrows
    for i in range(len(pos_inds)):
        dist = pos_widths[i]
        arrow_obj = plt.arrow(
            pos_lefts[i],
            pos_inds[i],
            max(dist - hl_scaled, 0.000001),
            0,
            head_length=min(dist, hl_scaled),
            color=colors.red_rgb,
            width=bar_width,
            head_width=bar_width,
        )

        if pos_low is not None and i < len(pos_low):
            plt.errorbar(
                pos_lefts[i] + pos_widths[i],
                pos_inds[i],
                xerr=np.array(
                    [[pos_widths[i] - pos_low[i]], [pos_high[i] - pos_widths[i]]]
                ),
                ecolor=colors.light_red_rgb,
            )

        txt_obj = plt.text(
            pos_lefts[i] + 0.5 * dist,
            pos_inds[i],
            format_value(pos_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                pos_lefts[i] + (5 / 72) * bbox_to_xscale + dist,
                pos_inds[i],
                format_value(pos_widths[i], "%+0.02f"),
                horizontalalignment="left",
                verticalalignment="center",
                color=colors.red_rgb,
                fontsize=12,
            )

    # draw the negative arrows
    for i in range(len(neg_inds)):
        dist = neg_widths[i]

        arrow_obj = plt.arrow(
            neg_lefts[i],
            neg_inds[i],
            -max(-dist - hl_scaled, 0.000001),
            0,
            head_length=min(-dist, hl_scaled),
            color=colors.blue_rgb,
            width=bar_width,
            head_width=bar_width,
        )

        if neg_low is not None and i < len(neg_low):
            plt.errorbar(
                neg_lefts[i] + neg_widths[i],
                neg_inds[i],
                xerr=np.array(
                    [[neg_widths[i] - neg_low[i]], [neg_high[i] - neg_widths[i]]]
                ),
                ecolor=colors.light_blue_rgb,
            )

        txt_obj = plt.text(
            neg_lefts[i] + 0.5 * dist,
            neg_inds[i],
            format_value(neg_widths[i], "%+0.02f"),
            horizontalalignment="center",
            verticalalignment="center",
            color="white",
            fontsize=12,
        )
        text_bbox = txt_obj.get_window_extent(renderer=renderer)
        arrow_bbox = arrow_obj.get_window_extent(renderer=renderer)

        # if the text overflows the arrow then draw it after the arrow
        if text_bbox.width > arrow_bbox.width:
            txt_obj.remove()

            txt_obj = plt.text(
                neg_lefts[i] - (5 / 72) * bbox_to_xscale + dist,
                neg_inds[i],
                format_value(neg_widths[i], "%+0.02f"),
                horizontalalignment="right",
                verticalalignment="center",
                color=colors.blue_rgb,
                fontsize=12,
            )

    # draw the y-ticks twice, once in gray and then again with just the feature names in black
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ytick_pos = list(range(num_features)) + list(np.arange(num_features) + 1e-8)
    plt.yticks(
        ytick_pos,
        yticklabels[:-1] + [l.split("=")[-1] for l in yticklabels[:-1]],
        fontsize=13,
    )

    # put horizontal lines for each feature row
    for i in range(num_features):
        plt.axhline(i, color="#cccccc", lw=0.5, dashes=(1, 5), zorder=-1)

    # mark the prior expected value and the model prediction
    plt.axvline(
        base_values,
        0,
        1 / num_features,
        color="#bbbbbb",
        linestyle="--",
        linewidth=0.5,
        zorder=-1,
    )
    fx = base_values + values.sum()
    plt.axvline(fx, 0, 1, color="#bbbbbb", linestyle="--", linewidth=0.5, zorder=-1)

    # clean up the main axis
    plt.gca().xaxis.set_ticks_position("bottom")
    plt.gca().yaxis.set_ticks_position("none")
    plt.gca().spines["right"].set_visible(False)
    plt.gca().spines["top"].set_visible(False)
    plt.gca().spines["left"].set_visible(False)
    ax.tick_params(labelsize=13)
    # plt.xlabel("\nModel output", fontsize=12)

    # draw the E[f(X)] tick mark
    xmin, xmax = ax.get_xlim()
    ax2 = ax.twiny()
    ax2.set_xlim(xmin, xmax)
    ax2.set_xticks(
        [base_values, base_values + 1e-8]
    )  # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax2.set_xticklabels(
        ["\n$E[f(X)]$", "\n$ = " + format_value(base_values, "%0.03f") + "$"],
        fontsize=12,
        ha="left",
    )
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)
    ax2.spines["left"].set_visible(False)

    # draw the f(x) tick mark
    ax3 = ax2.twiny()
    ax3.set_xlim(xmin, xmax)
    # The 1e-8 is so matplotlib 3.3 doesn't try and collapse the ticks
    ax3.set_xticks([base_values + values.sum(), base_values + values.sum() + 1e-8])
    ax3.set_xticklabels(
        ["$f(x)$", "$ = " + format_value(fx, "%0.03f") + "$"], fontsize=12, ha="left"
    )
    tick_labels = ax3.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-10 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(12 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_color("#999999")
    ax3.spines["right"].set_visible(False)
    ax3.spines["top"].set_visible(False)
    ax3.spines["left"].set_visible(False)

    # adjust the position of the E[f(X)] = x.xx label
    tick_labels = ax2.xaxis.get_majorticklabels()
    tick_labels[0].set_transform(
        tick_labels[0].get_transform()
        + matplotlib.transforms.ScaledTranslation(-20 / 72.0, 0, fig.dpi_scale_trans)
    )
    tick_labels[1].set_transform(
        tick_labels[1].get_transform()
        + matplotlib.transforms.ScaledTranslation(
            22 / 72.0, -1 / 72.0, fig.dpi_scale_trans
        )
    )

    tick_labels[1].set_color("#999999")

    # color the y tick labels that have the feature values as gray
    # (these fall behind the black ones with just the feature name)
    tick_labels = ax.yaxis.get_majorticklabels()
    for i in range(num_features):
        tick_labels[i].set_color("#999999")

    if show:
        plt.show()
    else:
        return plt.gcf()


# The "dt_shapley_rank" here is the result from qoi method,
# should be a table with features, contributions., etc.


def prepare_waterfall(dt_shapley_rank):
    # Pick the upperbounds and lowerbounds
    rank = dt_shapley_rank.Rank.max()
    df_upperbounds = []  # List of max Contribution rank value of each attributes.
    df_lowerbounds = []  # List of min Contribution rank value of each attributes.
    for col in dt_shapley_rank.columns[:-8]:
        temp_bounds = dt_shapley_rank.copy()
        temp_bounds = temp_bounds[temp_bounds["Feature"] == col]
        df_upperbounds.append(temp_bounds["Contribution rank"].max())
        df_lowerbounds.append(temp_bounds["Contribution rank"].min())

    value_dict = {
        "upper_bounds": df_upperbounds,
        "lower_bounds": df_lowerbounds,
        "base_values": dt_shapley_rank["Contribution rank"].mean(),
        "features": [[] for _ in range(rank)],
        "feature_names": dt_shapley_rank.columns[:-8],
        "data": [[] for _ in range(dt_shapley_rank.Rank.max())],
        "values": [[] for _ in range(rank)],
    }

    temp_val = value_dict.copy()

    for i in range(dt_shapley_rank.shape[0]):
        print("turn ", i)

        temp_val["features"][int(i / 2)].append(dt_shapley_rank.iloc[i]["Feature"])
        temp_val["data"][int(i / 2)].append(
            dt_shapley_rank.iloc[i][dt_shapley_rank.iloc[i]["Feature"]]
        )
        temp_val["values"][int(i / 2)].append(
            dt_shapley_rank.iloc[i]["Contribution rank"]
        )

    return value_dict


def draw_waterfall(value_dict, rank):
    rank_dict = value_dict.copy()
    rank_dict["features"] = pd.Series(value_dict["features"][rank])
    rank_dict["data"] = pd.Series(value_dict["data"][rank])
    rank_dict["values"] = np.array(value_dict["values"][rank])
    rank_dict["upper_bounds"] = np.array(value_dict["upper_bounds"])
    rank_dict["lower_bounds"] = np.array(value_dict["lower_bounds"])
    waterfall(rank_dict, max_display=10, show=True)
