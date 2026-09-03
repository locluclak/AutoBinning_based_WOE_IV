import numpy as np
from optbinning import OptimalBinning

from core.config_loader import MAX_BIN, MAX_BIN_SIZE, MIN_BIN, MIN_BIN_SIZE, MIN_DIFF_WOE, SPECIALVALUE


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


def calculate_feature(feature, X_train, y, considerMISSING=False):
    x_original = X_train[feature].copy()

    zero_pct = (x_original == SPECIALVALUE).mean()

    special_allowed = zero_pct >= MIN_BIN_SIZE
    if special_allowed:
        mask_nonmissing = x_original.notna() & (x_original != SPECIALVALUE)
    else:
        mask_nonmissing = x_original.notna()

    x = x_original.loc[mask_nonmissing]
    y_clean = y.loc[mask_nonmissing]

    n_old = len(X_train)
    n_new = len(x)

    if n_new == 0:
        raise ValueError("Feature has no usable observations.")

    if considerMISSING:
        min_bin_size = MIN_BIN_SIZE * n_old / n_new
        max_bin_size = min(MAX_BIN_SIZE * n_old / n_new, 1)
    else:
        min_bin_size = MIN_BIN_SIZE
        max_bin_size = MAX_BIN_SIZE

    infeasible = considerMISSING and min_bin_size > 0.5

    p_bar = np.mean(y_clean)
    min_event_rate_diff = 2 * p_bar * (1 - p_bar) * np.tanh(MIN_DIFF_WOE / 2)

    part = "considerMISSING" if considerMISSING else "removeMISSING"

    results = {}

    for option_name, trend in OPTIONS.items():
        if infeasible:
            optb = _InfeasibleModel()
        else:
            optb = OptimalBinning(name=feature, dtype="numerical", min_n_bins=MIN_BIN, max_n_bins=MAX_BIN, min_bin_size=min_bin_size, max_bin_size=max_bin_size, min_event_rate_diff=min_event_rate_diff, monotonic_trend=trend)
            optb.fit(x, y_clean)

        missing_first = considerMISSING and optb.status == "OPTIMAL"
        if special_allowed:
            if considerMISSING:
                display_x = x_original
                display_y = y
            else:
                display_x = x_original[x_original.notna()]
                display_y = y.loc[display_x.index]
        elif missing_first:
            display_x = x_original
            display_y = y
        else:
            display_x = x
            display_y = y_clean

        results[f"{part} | {option_name}"] = {
            "part": part,
            "option": option_name,
            "model": optb,
            "splits": optb.splits,
            "binning_table": optb.binning_table,
            "x": display_x,
            "y": display_y,
            "missing_first": missing_first,
            "specialvalue": SPECIALVALUE if special_allowed else None,
        }

    return results