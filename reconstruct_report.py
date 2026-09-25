# python reconstruct_report.py selected_feature_splits.json [selected2.json] [output.html]
#
# With one JSON, the report is a single stacked report.
# With two JSONs, the report is split into a left/right comparison where the
# same feature appears on the same row.

import json
import sys
from pathlib import Path

from core.io_utils import load_data
from reconstruct.reconstruct import build_report


def normalize_special(config):
    special = config.get("special")
    if special is None:
        sv = config.get("specialvalue")
        special = [sv] if sv is not None else []
    return [s for s in special if s != ""]


def parse_args(argv):
    """Return (json_paths, output_file or None)."""
    args = list(argv)
    output_file = None
    if len(args) >= 2 and args[-1].lower().endswith(".html"):
        output_file = args.pop()
    return args, output_file


def main(export_jsons: list, output_file: str = None):
    versions = []
    first_config = {}
    for path in export_jsons:
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        config = payload.get("config", {})
        if not first_config:
            first_config = config
        versions.append({
            "label": Path(path).stem,
            "config": payload.get("features", {}),
            "special": normalize_special(config),
        })

    input_file = first_config.get("input_file", "")
    label_name = first_config.get("label_name", "LABEL")
    ignore_columns = first_config.get("ignore_column", [])
    min_bin_size = first_config.get("min_bin_size", 0.05)

    output_file = output_file or first_config.get("output_file", "reconstructed_report.html")

    df = load_data(input_file)
    if ignore_columns:
        df = df.drop(columns=[c for c in ignore_columns if c in df.columns])

    html = build_report(df, versions, label_name, min_bin_size=min_bin_size)
    Path(output_file).write_text(html, encoding="utf-8")
    print(f"HTML report: {output_file} ({len(versions)} version(s))")


if __name__ == "__main__":
    jsons, out = parse_args(sys.argv[1:])
    if not jsons:
        print("Usage: python reconstruct_report.py selected1.json [selected2.json] [output.html]",
              file=sys.stderr)
        sys.exit(1)
    main(jsons, out)