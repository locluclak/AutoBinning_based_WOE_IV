import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from core.config_loader import (MAX_BIN, MAX_BIN_SIZE, MIN_BIN, MIN_BIN_SIZE, MIN_DIFF_WOE, SPECIAL)
from core.woe_stats import special_mask


OPTIONS = {
    "1. Free optimization": "auto",
    "2. Monotonic": "auto_asc_desc",
    "3. U-shape / heuristic": "auto_heuristic",
}


class _InfeasibleModel:
    def __init__(self):
        self.status = "INFEASIBLE"
        self.splits = []
        self.binning_table = None


class _CategoricalModel:
    """Placeholder model for categorical features: each group is already a bin."""
    def __init__(self, groups):
        self.status = "OK"
        self.splits = list(groups)
        self.binning_table = None


def calculate_feature(feature, X_train, y, considerSPECIAL=False):
    x_original = X_train[feature].copy()

    special_values = list(SPECIAL)
    if considerSPECIAL:
        mask = ~special_mask(x_original, special=special_values)
        x = x_original.loc[mask]
        y_clean = y.loc[mask]
        display_x = x_original
        display_y = y
        feature_special = special_values
    else:
        mask = ~special_mask(x_original, special=special_values)
        x = x_original.loc[mask]
        y_clean = y.loc[mask]
        display_x = x
        display_y = y_clean
        feature_special = []

    n_old = len(X_train)
    n_new = len(x)

    if n_new == 0:
        raise ValueError("Feature has no usable observations.")

    if considerSPECIAL:
        min_bin_size = MIN_BIN_SIZE * n_old / n_new
        max_bin_size = min(MAX_BIN_SIZE * n_old / n_new, 1)
    else:
        min_bin_size = MIN_BIN_SIZE
        max_bin_size = MAX_BIN_SIZE

    infeasible = considerSPECIAL and min_bin_size > 0.5

    p_bar = np.mean(y_clean)
    min_event_rate_diff = 2 * p_bar * (1 - p_bar) * np.tanh(MIN_DIFF_WOE / 2)

    part = "considerSPECIAL" if considerSPECIAL else "removeSPECIAL"

    results = {}

    for option_name, trend in OPTIONS.items():
        if infeasible:
            optb = _InfeasibleModel()
        else:
            optb = OptimalBinning(name=feature, dtype="numerical", min_n_bins=MIN_BIN, max_n_bins=MAX_BIN, min_bin_size=min_bin_size, max_bin_size=max_bin_size, min_event_rate_diff=min_event_rate_diff, monotonic_trend=trend)
            optb.fit(x, y_clean)

        missing_first = considerSPECIAL and optb.status == "OPTIMAL"

        results[f"{part} | {option_name}"] = {
            "part": part,
            "option": option_name,
            "model": optb,
            "splits": optb.splits,
            "binning_table": optb.binning_table,
            "x": display_x,
            "y": display_y,
            "missing_first": missing_first,
            "special": feature_special,
        }

    return results


def calculate_categorical_feature(feature, X_train, y, considerSPECIAL=False):
    """
    Compute WOE / IV for a categorical feature. No OptimalBinning is used:
    every group (category value) is treated as one bin. The only choice is
    between the 'considerSPECIAL' and 'removeSPECIAL' versions.
    """
    x_original = X_train[feature].copy()

    special_values = list(SPECIAL)
    if considerSPECIAL:
        display_x = x_original
        display_y = y
        feature_special = special_values
    else:
        mask = ~special_mask(x_original, special=special_values)
        display_x = x_original[mask]
        display_y = y[mask]
        feature_special = []

    groups = [str(c) for c in pd.unique(display_x.dropna())]

    part = "considerSPECIAL" if considerSPECIAL else "removeSPECIAL"

    results = {}
    option_name = "Groups as bins"
    results[f"{part} | {option_name}"] = {
        "part": part,
        "option": option_name,
        "categorical": True,
        "model": _CategoricalModel(groups),
        "splits": groups,
        "binning_table": None,
        "x": display_x,
        "y": display_y,
        "missing_first": considerSPECIAL,
        "special": feature_special,
    }

    return results