"""Generate a per-bin CSV report from a selected splits JSON file.

Usage:
    python generate_bins_csv.py [splits.json] [output.csv]

The JSON is the one exported by the HTML report (e.g. selected_categorical_splits.json):

    {
      "config": {"input_file": ..., "label_name": ..., "specialvalue": ..., ...},
      "features": {
        "FEATURE": {"option": ..., "part": "considerMISSING", "type": "categorical", "splits": [...]}
      }
    }

Each CSV row describes one bin of one feature.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from core.io_utils import load_data
from core.woe_stats import (cramer_v_type, create_woe_df, create_woe_df_categorical,
                            is_categorical, special_mask)

CSV_COLUMNS = [
    "Label name",
    "Feature type",
    "Feature name",
    "Column name",
    "Bin",
    "Number customer",
    "Contribution",
    "Conversion rate",
    "Event_cnt",
    "Non_event_cnt",
    "Dist_event",
    "Dist_non_event",
    "ChiSquare",
    "P_value",
    "Pvalue_remark",
    "CramerV",
    "CramerV_remark",
    "WOE",
    "IV_bin",
    "IV_total of feature",
    "score",
]


def round6(value):
    return round(float(value), 6)


def normalize_special(jconfig):
    special = jconfig.get("special")
    if special is None:
        sv = jconfig.get("specialvalue")
        special = [sv] if sv is not None else []
    return [s for s in special if s != ""]


def chi2_stats(labels, target):
    cont = pd.crosstab(pd.Series(labels, name="bin"), pd.Series(target, name="target"))
    table = np.asarray(cont.values, dtype=float)
    n = table.sum()
    r, c = table.shape
    if n == 0 or r <= 1 or c <= 1:
        return 0.0, 1.0, 0.0
    chi2_stat, p_value, _, _ = chi2_contingency(table, correction=False)
    v_c = float(np.sqrt(chi2_stat / (n * min(r - 1, c - 1))))
    return float(chi2_stat), float(p_value), v_c


def chi2_continuous(x, y, splits, special):
    data = pd.DataFrame({"x": x, "target": y}).reset_index(drop=True)
    value_specials = [v for v in (special or []) if v != "MISSING"]
    keep = data["x"].notna()
    if value_specials:
        keep = keep & ~data["x"].isin(value_specials)
    labels = pd.Series(np.nan, index=data.index, dtype=object)
    cut = pd.cut(data.loc[keep, "x"], bins=[-np.inf] + list(splits) + [np.inf],
                 right=True, include_lowest=True)
    labels.loc[keep] = [str(b) for b in cut]
    for v in value_specials:
        labels.loc[data["x"] == v] = f"Special {v}"
    labels.loc[data["x"].isna()] = "Missing"
    return chi2_stats(labels, data["target"])


def chi2_categorical(x, y):
    data = pd.DataFrame({"x": x, "target": y}).reset_index(drop=True)
    labels = pd.Series([str(v) if pd.notna(v) else "Missing" for v in data["x"]],
                       index=data.index, dtype=object)
    return chi2_stats(labels, data["target"])


def continuous_counts(x, y, splits, missing_first, special):
    data = pd.DataFrame({"x": x, "target": y}).reset_index(drop=True)
    value_specials = [v for v in (special or []) if v != "MISSING"]
    keep_mask = data["x"].notna()
    if value_specials:
        keep_mask = keep_mask & ~data["x"].isin(value_specials)

    data_cut = data[keep_mask].copy()
    bin_edges = [-np.inf] + list(splits) + [np.inf]
    data_cut["bin"] = pd.cut(data_cut["x"], bins=bin_edges, right=True, include_lowest=True)

    bins_df = (data_cut.groupby("bin", observed=False)["target"]
        .agg(n_events=lambda s: (s == 1).sum(), n_non_events=lambda s: (s == 0).sum())
        .reset_index())
    bins_df["Bin"] = bins_df["bin"].astype(str)
    bins_df = bins_df[["Bin", "n_events", "n_non_events"]]

    special_df = None
    if value_specials:
        rows = []
        for v in value_specials:
            mask = data["x"] == v
            if not mask.any():
                continue
            rows.append({
                "Bin": f"Special {v}",
                "n_events": int((mask & (data["target"] == 1)).sum()),
                "n_non_events": int((mask & (data["target"] == 0)).sum()),
            })
        if rows:
            special_df = pd.DataFrame(rows)

    missing_df = None
    missing_mask = data["x"].isna()
    if missing_mask.any():
        missing_df = pd.DataFrame({
            "Bin": ["Missing"],
            "n_events": [int((missing_mask & (data["target"] == 1)).sum())],
            "n_non_events": [int((missing_mask & (data["target"] == 0)).sum())],
        })

    parts = []
    order = (missing_df, bins_df, special_df) if missing_first else (special_df, bins_df, missing_df)
    for d in order:
        if d is not None and len(d):
            parts.append(d)
    if not parts:
        return pd.DataFrame(columns=["Bin", "n_events", "n_non_events"])
    return pd.concat(parts, ignore_index=True)


def categorical_counts(x, y, missing_first):
    data = pd.DataFrame({"x": x, "target": y})
    non_null = data["x"].notna()
    cats = list(pd.unique(data.loc[non_null, "x"]))

    if non_null.any():
        grouped = (data.loc[non_null].groupby("x", sort=False)["target"]
            .agg(n_events=lambda s: (s == 1).sum(), n_non_events=lambda s: (s == 0).sum())
            .reindex(cats)
            .reset_index()
            .rename(columns={"x": "bin"}))
        grouped["Bin"] = grouped["bin"].astype(str)
        grouped = grouped[["Bin", "n_events", "n_non_events"]]
    else:
        grouped = pd.DataFrame(columns=["Bin", "n_events", "n_non_events"])

    missing_mask = data["x"].isna()
    if missing_mask.any():
        missing_row = pd.DataFrame({
            "Bin": ["Missing"],
            "n_events": [(missing_mask & (data["target"] == 1)).sum()],
            "n_non_events": [(missing_mask & (data["target"] == 0)).sum()],
        })
        if missing_first:
            grouped = pd.concat([missing_row, grouped], ignore_index=True)
        else:
            grouped = pd.concat([grouped, missing_row], ignore_index=True)
    return grouped


def process_feature(name, cfg, x, y, special_values, label_name):
    ftype = cfg.get("type") or ("categorical" if is_categorical(x) else "continuous")
    categorical = ftype == "categorical"

    part = cfg.get("part", "considerMISSING")
    consider = str(part).lower().startswith("consider")
    missing_first = consider
    splits = list(cfg.get("splits") or [])

    value_specials = [v for v in special_values if v != "MISSING"]
    if consider:
        x_use, y_use = x, y
        special_use = special_values
    else:
        keep = ~special_mask(x, special=special_values)
        x_use, y_use = x[keep], y[keep]
        special_use = []

    if categorical:
        woe_df = create_woe_df_categorical(x_use, y_use, missing_first=missing_first)
        counts = categorical_counts(x_use, y_use, missing_first)
        chi2_stat, p_value, v_c = chi2_categorical(x_use, y_use)
    else:
        woe_df = create_woe_df(x_use, y_use, splits, missing_first=missing_first, special=special_use)
        counts = continuous_counts(x_use, y_use, splits, missing_first, special_use)
        chi2_stat, p_value, v_c = chi2_continuous(x_use, y_use, splits, special_use)

    merged = woe_df.merge(counts, on="Bin", how="left")

    total_events = int(merged["n_events"].sum())
    total_non_events = int(merged["n_non_events"].sum())
    total_obs = total_events + total_non_events
    iv_total = float(woe_df["IV_total"].iloc[0])
    remark = cramer_v_type(v_c)

    rows = []
    for _, row in merged.iterrows():
        event_cnt = int(row["n_events"])
        non_event_cnt = int(row["n_non_events"])
        n_obs = event_cnt + non_event_cnt
        contribution = n_obs / total_obs * 100 if total_obs else 0.0
        conversion = event_cnt / n_obs * 100 if n_obs else 0.0
        dist_event = event_cnt / total_events * 100 if total_events else 0.0
        dist_non_event = non_event_cnt / total_non_events * 100 if total_non_events else 0.0
        woe = float(row["WOE"])
        iv = float(row["IV_detail"])
        score = np.sign(woe) * v_c * (iv / iv_total) * 100 if iv_total else 0.0

        rows.append({
            "Label name": label_name,
            "Feature type": ftype,
            "Feature name": name,
            "Column name": name,
            "Bin": row["Bin"],
            "Number customer": n_obs,
            "Contribution": round6(contribution),
            "Conversion rate": round6(conversion),
            "Event_cnt": event_cnt,
            "Non_event_cnt": non_event_cnt,
            "Dist_event": round6(dist_event),
            "Dist_non_event": round6(dist_non_event),
            "ChiSquare": round6(chi2_stat),
            "P_value": round6(p_value),
            "Pvalue_remark": "YES" if p_value <= 0.05 else "NO",
            "CramerV": round6(v_c),
            "CramerV_remark": remark,
            "WOE": round6(woe),
            "IV_bin": round6(iv),
            "IV_total of feature": round6(iv_total),
            "score": round6(score),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description="Generate per-bin CSV report from splits JSON.")
    parser.add_argument("splits_json", nargs="?", default="selected_categorical_splits.json")
    parser.add_argument("output_csv", nargs="?", default="bins_report.csv")
    args = parser.parse_args()

    with open(args.splits_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    jconfig = payload.get("config", {})
    input_file = jconfig.get("input_file", "segmentation.csv")
    label_name = jconfig.get("label_name", "LABEL")
    ignore_columns = jconfig.get("ignore_column", [])
    special_values = normalize_special(jconfig)

    df = load_data(input_file)
    df = df.drop(columns=[c for c in ignore_columns if c in df.columns])
    if label_name not in df.columns:
        sys.exit(f"Label column '{label_name}' not found in '{input_file}'.")

    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)

    all_rows = []
    for feature, cfg in payload.get("features", {}).items():
        if feature not in X_train.columns:
            print(f"SKIP: {feature} -> not in data columns", file=sys.stderr)
            continue
        try:
            rows = process_feature(feature, cfg, X_train[feature], y, special_values, label_name)
            all_rows.extend(rows)
            print(f"OK: {feature} ({len(rows)} bins)")
        except Exception as exc:
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}", file=sys.stderr)

    if not all_rows:
        sys.exit("No rows produced.")

    out_df = pd.DataFrame(all_rows, columns=CSV_COLUMNS)
    out_df.to_csv(args.output_csv, index=False, encoding="utf-8-sig")
    print(f"CSV report: {args.output_csv} ({len(out_df)} rows)")


if __name__ == "__main__":
    main()