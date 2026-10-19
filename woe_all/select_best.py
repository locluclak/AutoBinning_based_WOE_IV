import pandas as pd

from core.config_loader import CONFIG, IGNORE_COLUMN, resolve_bins
from core.woe_stats import create_woe_df, is_categorical
from .html_tables import compute_result_woe_df
from .optimization import FREE_OPTIONS, calculate_categorical_feature, calculate_feature


def result_iv(result):
    """Total IV of a result, computed the same way the report does."""
    woe_df = compute_result_woe_df(result, create_woe_df)
    return float(woe_df["IV_total"].iloc[0])


def compute_free_results(feature, X_train, y, min_n_bins=None, max_n_bins=None):
    """Compute only the Free optimization for both SPECIAL parts.

    No Monotonic / U-shape / Special-vs-Non-Special options are produced.
    Returns a dict of the two candidate results (remove / consider SPECIAL).
    """
    results = {}
    if is_categorical(X_train[feature]):
        for consider in (False, True):
            results.update(calculate_categorical_feature(feature, X_train, y, considerSPECIAL=consider))
    else:
        for consider in (False, True):
            results.update(calculate_feature(feature, X_train, y, considerSPECIAL=consider,
                                             min_n_bins=min_n_bins, max_n_bins=max_n_bins,
                                             options=FREE_OPTIONS))
    return results


def select_best_result(results):
    """Pick the result with the highest total IV.

    Any INFEASIBLE model is ignored. Returns (result, iv) or (None, None).
    """
    best = None
    best_iv = None
    for result in results.values():
        if result["model"].status not in ("OPTIMAL", "OK"):
            continue
        iv = result_iv(result)
        if best is None or iv > best_iv:
            best, best_iv = result, iv
    return best, best_iv


def describe_result(result, iv):
    """Build the JSON feature entry for the selected result (with per-bin scores)."""
    categorical = bool(result.get("categorical"))
    if categorical:
        splits = [str(s) for s in result["splits"]]
    else:
        splits = [float(s) for s in result["splits"]]
    return {
        "option": result["option"],
        "part": result["part"],
        "type": "categorical" if categorical else "continuous",
        "splits": splits,
        "iv": round(iv, 6),
        "bins": bin_scores(result),
    }


def bin_scores(result):
    """Per-bin scores for the selected result.

    ``score = sign(WOE) * IV_bin / IV_total * 100`` (no Cramer's V coefficient);
    negative scores are floored at 0.
    """
    woe_df = compute_result_woe_df(result, create_woe_df)
    iv_total = float(woe_df["IV_total"].iloc[0])
    bins = []
    for _, row in woe_df.iterrows():
        score = 0.0
        if iv_total != 0 and row["WOE"] > 0:
            score = row["IV_detail"] / iv_total * 100
        bins.append({
            "bin": str(row["Bin"]),
            "score": round(float(score), 6),
        })
    return bins


def build_best_config(df: pd.DataFrame, label_name="LABEL"):
    """Select the highest-IV Free split per feature and return the export payload.

    Returns (payload, skipped) where payload matches the HTML export format
    (``{"config": ..., "features": ...}``) and skipped lists (feature, reason).
    """
    df = df.drop(columns=[c for c in IGNORE_COLUMN if c in df.columns])
    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)
    features = [c for c in X_train.columns
                if c != label_name and not str(c).startswith(label_name)]

    feature_config = {}
    skipped = []
    unconfigured = []
    for feature in features:
        bins = resolve_bins(feature)
        if bins is None:
            unconfigured.append(feature)
            skipped.append((feature, "No bin count configured in feature_bins"))
            continue
        min_n_bins, max_n_bins = bins
        try:
            results = compute_free_results(feature, X_train, y, min_n_bins, max_n_bins)
            best, iv = select_best_result(results)
            if best is None:
                skipped.append((feature, "No OPTIMAL/OK binning option"))
                print(f"SKIP: {feature} -> no usable binning option")
                continue
            feature_config[feature] = describe_result(best, iv)
            print(f"OK: {feature} -> {best['part']} (bins={min_n_bins}-{max_n_bins}, IV={iv:.4f})")
        except Exception as exc:
            skipped.append((feature, exc))
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")

    if unconfigured:
        print(f"\n{len(unconfigured)} feature(s) missing a bin count in feature_bins:")
        for feature in unconfigured:
            print(f"  {feature}: 0")
        print("Add them under 'feature_bins' in config.yaml to include them.")

    payload = {"config": CONFIG, "features": feature_config}
    return payload, skipped
