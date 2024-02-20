"""
Object where visualizations will be added.

NOTE: Matplotlib only. Must be an optional import.
"""

import pandas as pd
from sharp.utils._utils import _optional_import
from ._waterfall import _waterfall


class ShaRPViz:  # TODO
    def __init__(self, sharp):
        self.sharp = sharp

    def bar(self, scores, ax=None, **kwargs):
        """
        TODO
        """
        if ax is None:
            plt = _optional_import("matplotlib.pyplot")
            fig, ax = plt.subplots(1, 1)

        ax.bar(self.sharp.feature_names_.astype(str), scores, **kwargs)
        ax.set_ylabel("Contribution to QoI")
        ax.set_xlabel("Features")

        return ax

    def waterfall(self, scores, mean_shapley_value=0):
        """
        TODO: refactor waterfall plot code.
        """

        feature_names = self.sharp.feature_names_.astype(str).tolist()

        rank_dict = {
            "upper_bounds": None,
            "lower_bounds": None,
            "features": None,  # pd.Series(feature_names),
            "data": None,  # pd.Series(ind_values, index=feature_names),
            "base_values": mean_shapley_value,
            "feature_names": feature_names,
            "values": pd.Series(scores, index=feature_names),
        }
        return _waterfall(rank_dict, max_display=10)
