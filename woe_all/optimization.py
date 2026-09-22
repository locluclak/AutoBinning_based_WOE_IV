import numpy as np
import pandas as pd
from optbinning import OptimalBinning

from core.config_loader import (MAX_BIN, MAX_BIN_SIZE, MIN_BIN, MIN_BIN_SIZE, MIN_DIFF_WOE, SPECIAL)
from core.woe_stats import create_woe_df, special_mask


OPTIONS = {
    "1. Free optimization": None,
    "2. Monotonic": "auto_asc_desc",
    "3. U-shape / peak-valley": "auto_peak_valley",
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


def _build_optb(name, min_bin_size, max_bin_size, min_event_rate_diff, trend):
    return OptimalBinning(name=name,
                          dtype="numerical",
                          min_n_bins=MIN_BIN,
                          max_n_bins=MAX_BIN,
                          min_bin_size=min_bin_size,
                          max_bin_size=max_bin_size,
                          min_event_rate_diff=min_event_rate_diff,
                          monotonic_trend=trend)


def _model_iv(optb, x, y, considerSPECIAL, special):
    """Total IV of a fitted model, computed the same way the report does it.

    Returns None when the model is not optimal.
    """
    if optb.status not in ("OPTIMAL", "OK"):
        return None
    missing_first = considerSPECIAL and optb.status == "OPTIMAL"
    woe_df = create_woe_df(x, y, optb.splits, missing_first=missing_first, special=special)
    return float(woe_df["IV_total"].iloc[0])


def _fit_peak_valley(name, x, y, min_bin_size, max_bin_size, min_event_rate_diff,
                     display_x, display_y, considerSPECIAL, special):
    """U-shape trend: run strict 'peak' and 'valley' and keep the higher IV one.

    If both are infeasible, return an infeasible model.
    """
    candidates = [
        _build_optb(name, min_bin_size, max_bin_size, min_event_rate_diff, trend)
        for trend in ("peak", "valley")
    ]
    for optb in candidates:
        optb.fit(x, y)

    feasible = [optb for optb in candidates if optb.status in ("OPTIMAL", "OK")]
    if not feasible:
        return _InfeasibleModel()

    def iv_key(optb):
        iv = _model_iv(optb, display_x, display_y, considerSPECIAL, special)
        return -np.inf if iv is None else iv

    return max(feasible, key=iv_key)


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

    part = "consider SPECIAL" if considerSPECIAL else "remove SPECIAL"

    results = {}

    for option_name, trend in OPTIONS.items():
        if infeasible:
            optb = _InfeasibleModel()
        elif trend == "auto_peak_valley":
            optb = _fit_peak_valley(feature, x, y_clean, min_bin_size, max_bin_size,
                                    min_event_rate_diff, display_x, display_y,
                                    considerSPECIAL, feature_special)
        else:
            optb = _build_optb(feature, min_bin_size, max_bin_size, min_event_rate_diff, trend)
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

    part = "consider SPECIAL" if considerSPECIAL else "remove SPECIAL"

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