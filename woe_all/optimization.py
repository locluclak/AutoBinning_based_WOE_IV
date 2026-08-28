import numpy as np
from optbinning import OptimalBinning

from core.config_loader import MAX_BIN, MIN_BIN, MIN_DIFF_WOE, SPECIALVALUE


OPTIONS = {
    "1. Free optimization": "auto",
    "2. Monotonic": "auto_asc_desc",
    "3. U-shape / heuristic": "auto_heuristic",
}


def calculate_feature(feature, X_train, y, considerMISSING=False):
    x_original = X_train[feature].copy()

    zero_pct = (x_original == SPECIALVALUE).mean()

    if zero_pct < 0.05:
        mask_nonmissing = x_original.notna()
    else:
        mask_nonmissing = x_original.notna() & (x_original != SPECIALVALUE)

    x = x_original.loc[mask_nonmissing]
    y_clean = y.loc[mask_nonmissing]

    n_old = len(X_train)
    n_new = len(x)

    if n_new == 0:
        raise ValueError("Feature has no usable observations.")

    if considerMISSING:
        min_bin_size = 0.05 * n_old / n_new
        max_bin_size = min(0.50 * n_old / n_new, 1)
    else:
        min_bin_size = 0.05
        max_bin_size = 0.50

    p_bar = np.mean(y_clean)
    min_event_rate_diff = 2 * p_bar * (1 - p_bar) * np.tanh(MIN_DIFF_WOE / 2)

    part = "considerMISSING" if considerMISSING else "removeMISSING"

    results = {}

    for option_name, trend in OPTIONS.items():
        optb = OptimalBinning(name=feature, dtype="numerical", min_n_bins=MIN_BIN, max_n_bins=MAX_BIN, min_bin_size=min_bin_size, max_bin_size=max_bin_size, min_event_rate_diff=min_event_rate_diff, monotonic_trend=trend)
        optb.fit(x, y_clean)
        results[f"{part} | {option_name}"] = {
            "part": part,
            "option": option_name,
            "model": optb,
            "splits": optb.splits,
            "binning_table": optb.binning_table,
            "x": x,
            "y": y_clean,
        }

    return results