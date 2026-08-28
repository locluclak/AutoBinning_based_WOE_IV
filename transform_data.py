# python transform_data.py selected_feature_splits.json transformed.csv
# Export format inferred from output file extension (.csv or .parquet)

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from core.io_utils import load_data, write_data
from core.woe_stats import create_woe_df


def transform_feature(feature, x, y, splits, specialvalue=None, min_bin_size=0.05, consider_missing=True):
    feature_special = None
    if specialvalue is not None and (x == specialvalue).mean() >= min_bin_size:
        feature_special = specialvalue

    woe_df = create_woe_df(x, y, splits, missing_first=consider_missing, specialvalue=feature_special)
    woe_map = dict(zip(woe_df['Bin'], woe_df['WOE']))

    result = pd.DataFrame(index=x.index)
    result[f"{feature}_WOE"] = np.nan

    bin_edges = [-np.inf] + list(splits) + [np.inf]
    numeric_mask = x.notna()
    if feature_special is not None:
        numeric_mask = numeric_mask & (x != feature_special)

    cat = pd.cut(x[numeric_mask], bins=bin_edges, right=True, include_lowest=True)

    bins_in_order = []
    if feature_special is not None:
        bins_in_order.append(f"Special {feature_special}")
    bins_in_order += list(cat.cat.categories.astype(str))
    has_missing = x.isna().any()
    if consider_missing and has_missing:
        bins_in_order.append("Missing")

    bin_number = {b: i + 1 for i, b in enumerate(bins_in_order)}

    for b in bins_in_order:
        col = f"{feature}_bin{bin_number[b]}" if b != "Missing" else f"{feature}_MISSING"
        result[col] = 0
        if b.startswith("Special"):
            sel = (x == feature_special)
        elif b == "Missing":
            sel = x.isna()
        else:
            sel = numeric_mask & (cat.astype(str) == b)
        result.loc[sel, col] = 1
        result.loc[sel, f"{feature}_WOE"] = woe_map.get(b, np.nan)

    return result


def build_transformed(df, features_config, label_name, specialvalue=None, min_bin_size=0.05):
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
            splits = list(map(float, cfg['splits']))
            part = cfg.get('part', 'considerMISSING')
            consider_missing = part == "considerMISSING"
            x = X_train[feature]
            tf = transform_feature(feature, x, y, splits, specialvalue=specialvalue,
                                   min_bin_size=min_bin_size, consider_missing=consider_missing)
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
    min_bin_size = config.get("min_bin_size", 0.05)

    output_file = output_file or config.get("output_file", "transformed.csv")

    df = load_data(input_file)
    if ignore_columns:
        df = df.drop(columns=[c for c in ignore_columns if c in df.columns])

    transformed, skipped = build_transformed(df, features_config, label_name,
                                             specialvalue=specialvalue, min_bin_size=min_bin_size)

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
