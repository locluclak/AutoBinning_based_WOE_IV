# python transform_data.py selected_feature_splits.json transformed.csv
# Export format inferred from output file extension (.csv or .parquet)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.io_utils import load_data, write_data
from core.woe_stats import (create_woe_df, create_woe_df_categorical, is_categorical,
                            add_score_column, chi2_cramers, chi2_cramers_categorical,
                            _normalize_special, special_mask)


def transform_feature(feature, x, y, splits, special=None, min_bin_size=0.05, consider_special=True):
    if consider_special:
        woe_df = create_woe_df(x, y, splits, missing_first=True, special=special)
        _, v_c = chi2_cramers(x, y, splits, special=special)
    else:
        mask = ~special_mask(x, special=special)
        fit_x, fit_y = x[mask], y[mask]
        woe_df = create_woe_df(fit_x, fit_y, splits, missing_first=False, special=None)
        _, v_c = chi2_cramers(fit_x, fit_y, splits, special=None)

    score_df = add_score_column(woe_df, v_c)
    woe_map = dict(zip(woe_df['Bin'], woe_df['WOE']))
    score_map = dict(zip(score_df['Bin'], score_df['score']))

    result = pd.DataFrame(index=x.index)
    result[f"{feature}_WOE"] = np.nan
    result[f"{feature}_score"] = np.nan

    special_values = [v for v in _normalize_special(special) if v != 'MISSING']
    special_values = [v for v in special_values if (x == v).any()]

    bin_edges = [-np.inf] + list(splits) + [np.inf]
    numeric_mask = x.notna()
    if special_values:
        numeric_mask = numeric_mask & ~x.isin(special_values)

    cat = pd.cut(x[numeric_mask], bins=bin_edges, right=True, include_lowest=True)

    bins_in_order = []
    if consider_special:
        for v in special_values:
            bins_in_order.append(f"Special {v}")
    bins_in_order += list(cat.cat.categories.astype(str))
    has_missing = x.isna().any()
    if consider_special and has_missing:
        bins_in_order.append("Missing")

    special_lookup = {f"Special {v}": v for v in special_values}
    bin_number = {b: i + 1 for i, b in enumerate(bins_in_order)}

    for b in bins_in_order:
        col = f"{feature}_bin{bin_number[b]}" if b != "Missing" else f"{feature}_MISSING"
        result[col] = 0
        if b in special_lookup:
            sel = (x == special_lookup[b])
        elif b == "Missing":
            sel = x.isna()
        else:
            sel = numeric_mask & (cat.astype(str) == b)
        result.loc[sel, col] = 1
        result.loc[sel, f"{feature}_WOE"] = woe_map.get(b, np.nan)
        result.loc[sel, f"{feature}_score"] = score_map.get(b, np.nan)

    return result


def transform_categorical_feature(feature, x, y, categories, consider_special=True, special=None):
    if consider_special:
        fit_x, fit_y = x, y
    else:
        mask = ~special_mask(x, special=special)
        fit_x, fit_y = x[mask], y[mask]

    woe_df = create_woe_df_categorical(fit_x, fit_y, missing_first=consider_special)
    _, v_c = chi2_cramers_categorical(fit_x, fit_y)
    score_df = add_score_column(woe_df, v_c)
    woe_map = dict(zip(woe_df['Bin'], woe_df['WOE']))
    score_map = dict(zip(score_df['Bin'], score_df['score']))

    result = pd.DataFrame(index=x.index)
    result[f"{feature}_WOE"] = np.nan
    result[f"{feature}_score"] = np.nan

    bins_in_order = []
    has_missing = x.isna().any()
    if consider_special and has_missing:
        bins_in_order.append("Missing")
    bins_in_order += [str(c) for c in categories]

    bin_number = {b: i + 1 for i, b in enumerate(bins_in_order)}

    for b in bins_in_order:
        col = f"{feature}_bin{bin_number[b]}" if b != "Missing" else f"{feature}_MISSING"
        result[col] = 0
        if b == "Missing":
            sel = x.isna()
        else:
            sel = x.notna() & (x.astype(str) == b)
        result.loc[sel, col] = 1
        result.loc[sel, f"{feature}_WOE"] = woe_map.get(b, np.nan)
        result.loc[sel, f"{feature}_score"] = score_map.get(b, np.nan)

    return result


def build_transformed(df, features_config, label_name, special=None, min_bin_size=0.05):
    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)

    transformed = pd.DataFrame(index=df.index)
    skipped = []

    for feature, cfg in features_config.items():
        if feature == label_name or str(feature).startswith(label_name):
            continue
        if feature not in X_train.columns:
            skipped.append((feature, "Missing in data"))
            continue
        try:
            x = X_train[feature]
            part = cfg.get('part', 'consider SPECIAL')
            consider_special = part.startswith("consider")
            cfg_type = cfg.get('type')
            if cfg_type == 'categorical' or (cfg_type is None and is_categorical(x)):
                categories = [str(c) for c in cfg.get('splits', [])]
                if not categories:
                    categories = [str(c) for c in pd.unique(x.dropna())]
                tf = transform_categorical_feature(feature, x, y, categories,
                                                   consider_special=consider_special, special=special)
            else:
                splits = list(map(float, cfg['splits']))
                tf = transform_feature(feature, x, y, splits, special=special,
                                       min_bin_size=min_bin_size, consider_special=consider_special)
            transformed = pd.concat([transformed, tf], axis=1)
            print(f"OK: {feature}")
        except Exception as exc:
            skipped.append((feature, exc))
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")

    return transformed, skipped


def main(export_json: str, output_file: str = None):
    with open(export_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    features_config = payload.get("features", {})

    input_file = config.get("input_file", "")
    label_name = config.get("label_name", "LABEL")
    ignore_columns = config.get("ignore_column", [])
    specialvalue = config.get("specialvalue")
    special = config.get("special")
    if special is None:
        special = [specialvalue] if specialvalue is not None else []
    min_bin_size = config.get("min_bin_size", 0.05)

    output_file = output_file or config.get("output_file", "transformed.csv")

    df = load_data(input_file)
    if ignore_columns:
        df = df.drop(columns=[c for c in ignore_columns if c in df.columns])

    transformed, skipped = build_transformed(df, features_config, label_name,
                                             special=special, min_bin_size=min_bin_size)

    result = pd.concat([df, transformed], axis=1)
    write_data(result, output_file)

    if skipped:
        print("\nSkipped features:")
        for feature, exc in skipped:
            print(f"  {feature}: {type(exc).__name__}: {exc}")

    print(f"Transformed data written: {output_file}")
    print(f"Shape: {result.shape}")


if __name__ == "__main__":
    args = sys.argv[1:]
    export_json = args[0] if args else "selected_feature_splits.json"
    output_file = args[1] if len(args) > 1 else None
    main(export_json, output_file)
