"""
Chấm điểm khách hàng bằng cách sử dụng báo cáo CSV theo từng nhóm (bin)
được tạo từ file generate_bins_csv.py.

Cách dùng:
python generate_score_csv.py raw_data.csv bins_report.csv [output.csv] --reserve SNAPSHOT --id CUSTOMER_CDE

Đối với mỗi hàng trong file dữ liệu thô, 
từng đặc trưng (feature) có trong file CSV nhóm sẽ được ánh xạ vào nhóm mà 
giá trị của nó thuộc về (chia khoảng đối với đặc trưng liên tục, 
khớp chính xác đối với đặc trưng phân loại, cộng thêm các nhóm "Missing" / "Special "). 
Cột "score" được xuất ra là tổng điểm của tất cả các nhóm đang hoạt động.

Các cột đầu ra: , các cột được giữ lại (reserved columns), các cột đặc trưng ban đầu, score.
"""
import argparse
import re
import sys

import numpy as np
import pandas as pd

from core.io_utils import load_data, write_data

BIN_COLUMNS = ["Feature name", "Feature type", "Bin", "score"]

INTERVAL_RE = re.compile(r"^([\[(])(.+),\s*(.+)([\])])$")


def parse_edge(text):
    text = str(text).strip()
    lowered = text.lower()
    if lowered in ("-inf", "-infinity"):
        return float("-inf")
    if lowered in ("inf", "infinity"):
        return float("inf")
    try:
        return float(text)
    except ValueError:
        return None


def parse_interval(name):
    match = INTERVAL_RE.match(str(name))
    if not match:
        return None
    left = parse_edge(match.group(2))
    right = parse_edge(match.group(3))
    if left is None or right is None:
        return None
    return left, match.group(1) == "(", right, match.group(4) == ")"


class FeatureScorer:
    """Maps raw feature values to the score of the bin they fall into."""

    def __init__(self, feature, rows):
        self.feature = feature
        self.ftype = str(rows["Feature type"].iloc[0]).strip().lower()
        self.missing_score = 0.0
        self.special_scores = {}
        self.label_scores = {}
        self.intervals = []

        for _, row in rows.iterrows():
            score = float(row["score"]) if pd.notna(row["score"]) else 0.0
            name = str(row["Bin"])
            if name == "Missing":
                self.missing_score = score
            elif name.startswith("Special "):
                self.special_scores[name[len("Special "):]] = score
            else:
                interval = None if self.ftype == "categorical" else parse_interval(name)
                if interval is not None:
                    self.intervals.append(interval + (score,))
                else:
                    self.label_scores[name] = score

    def _special_mask(self, values):
        mask = pd.Series(False, index=values.index, dtype=bool)
        for label, _ in self.special_scores.items():
            part = values.eq(label)
            try:
                numeric = float(label)
            except (TypeError, ValueError):
                numeric = None
            if numeric is not None:
                part = part | pd.to_numeric(values, errors="coerce").eq(numeric)
            mask = mask | part.fillna(False)
        return mask

    def score(self, values):
        values = values.reset_index(drop=True)
        scores = np.zeros(len(values), dtype=float)
        is_null = pd.isna(values)

        if self.missing_score:
            scores[is_null.values] = self.missing_score

        special_mask = self._special_mask(values) if self.special_scores else None
        if special_mask is not None and special_mask.any():
            for label, s in self.special_scores.items():
                part = values.eq(label)
                try:
                    numeric = float(label)
                except (TypeError, ValueError):
                    numeric = None
                if numeric is not None:
                    part = part | pd.to_numeric(values, errors="coerce").eq(numeric)
                part = part.fillna(False)
                scores[part.values & ~is_null.values] = s

        remaining = ~is_null.values
        if special_mask is not None:
            remaining &= ~special_mask.values

        if self.intervals and remaining.any():
            idx = np.flatnonzero(remaining)
            cut_values = pd.to_numeric(values[remaining], errors="coerce")
            valid = ~cut_values.isna()
            if valid.any():
                edges = sorted({edge for iv in self.intervals for edge in (iv[0], iv[2])})
                right_closed = not any(iv[3] for iv in self.intervals)
                binned = pd.cut(cut_values[valid], bins=edges, right=right_closed,
                                include_lowest=True)
                key_to_score = {
                    (iv[0], iv[2], "right" if right_closed else "left"): iv[4]
                    for iv in self.intervals
                }
                bin_scores = np.array([
                    key_to_score.get((interval.left, interval.right, interval.closed), 0.0)
                    for interval in binned
                ])
                target = idx[valid.values]
                scores[target] = bin_scores

        if not self.intervals and remaining.any():
            lookup = {}
            for u in pd.unique(values[remaining]):
                if pd.isna(u):
                    continue
                key = str(u)
                if key in self.label_scores:
                    lookup[u] = self.label_scores[key]
                    continue
                try:
                    numeric = float(u)
                except (TypeError, ValueError):
                    continue
                if numeric in self.label_scores:
                    lookup[u] = self.label_scores[numeric]
            if lookup:
                scores[remaining] = values[remaining].map(lookup).fillna(0.0).values

        return scores


def main():
    parser = argparse.ArgumentParser(description="Score customers from a bin CSV report.")
    parser.add_argument("input_file", help="raw data file (csv/parquet/xlsx/json)")
    parser.add_argument("bin_csv", help="bin info CSV from generate_bins_csv.py")
    parser.add_argument("output_csv", nargs="?", default="scored_customers.csv")
    parser.add_argument("--reserve", nargs="*", default=[],
                        help="columns to keep in the output (in addition to the id)")
    parser.add_argument("--id", default="CUSTOMER_CDE",
                        help="customer id column (default: CUSTOMER_CDE)")
    args = parser.parse_args()

    bins = pd.read_csv(args.bin_csv)
    missing = [c for c in BIN_COLUMNS if c not in bins.columns]
    if missing:
        sys.exit(f"Bin CSV is missing required column(s): {', '.join(missing)}")

    df = load_data(args.input_file)
    if args.id not in df.columns:
        sys.exit(f"Id column '{args.id}' not found in '{args.input_file}'.")

    all_features = list(pd.unique(bins["Feature name"]))
    feature_names = [f for f in all_features if f in df.columns]
    for f in all_features:
        if f not in df.columns:
            print(f"SKIP: {f} -> not in data columns", file=sys.stderr)

    reserve = []
    for c in args.reserve:
        if c not in df.columns:
            print(f"SKIP reserve: {c} -> not in data columns", file=sys.stderr)
        elif c == args.id or c in feature_names:
            print(f"SKIP reserve: {c} -> already included", file=sys.stderr)
        else:
            reserve.append(c)

    total = np.zeros(len(df), dtype=float)
    for feature in feature_names:
        scorer = FeatureScorer(feature, bins[bins["Feature name"] == feature])
        total += scorer.score(df[feature])
        print(f"OK: {feature} ({len(scorer.intervals)} intervals, "
              f"{len(scorer.label_scores)} labels, {len(scorer.special_scores)} specials)")

    out = pd.DataFrame({args.id: df[args.id].values})
    for c in reserve:
        out[c] = df[c].values
    for f in feature_names:
        out[f] = df[f].values
    out["score"] = np.round(total, 6)

    write_data(out, args.output_csv)
    print(f"Scored CSV: {args.output_csv} ({len(out)} rows, {len(feature_names)} features)")


if __name__ == "__main__":
    main()