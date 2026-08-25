# python reconstruct_report.py selected_feature_splits.json output.html

import json
from pathlib import Path

from core.io_utils import load_data
from reconstruct.reconstruct import build_report


def main(export_json: str, output_file: str = None):
    with open(export_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    features_config = payload.get("features", {})

    input_file = config.get("input_file", "")
    label_name = config.get("label_name", "LABEL")
    ignore_columns = config.get("ignore_column", [])

    output_file = output_file or config.get("output_file", "reconstructed_report.html")

    df = load_data(input_file)
    if ignore_columns:
        df = df.drop(columns=[c for c in ignore_columns if c in df.columns])

    html = build_report(df, features_config, label_name)
    Path(output_file).write_text(html, encoding="utf-8")
    print(f"HTML report: {output_file}")


if __name__ == "__main__":
    import sys
    args = sys.argv[1:]
    export_json = args[0] if args else "selected_feature_splits.json"
    output_file = args[1] if len(args) > 1 else None
    main(export_json, output_file)