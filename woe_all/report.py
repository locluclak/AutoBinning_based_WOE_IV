import json
from html import escape

import matplotlib.pyplot as plt
import pandas as pd

from core.config_loader import CONFIG, IGNORE_COLUMN
from core.export_script import EXPORT_SCRIPT
from core.woe_stats import create_woe_df, figure_to_base64
from .html_tables import display_woe_tables
from .optimization import calculate_feature
from .optimization import calculate_feature
from .plotting import plot_options


def build_feature_html(feature, X_train, y):
    results = {}
    for consider in (False, True):
        results.update(calculate_feature(feature, X_train, y, considerMISSING=consider))

    parts = {}
    for option_name, result in results.items():
        parts.setdefault(result['part'], {})[result['option']] = result

    plot_options(results, feature)
    fig = plt.gcf()
    plot_html = figure_to_base64(fig)

    _, tables_html = display_woe_tables(results=results, create_woe_df_func=create_woe_df, render=False)

    status_rows = []
    for idx, (part, part_results) in enumerate(parts.items()):
        option_cells = []
        for option_name, result in part_results.items():
            splits = list(map(float, result['splits']))
            status = result['model'].status
            option_cells.append(
                f"<td>"
                f"<label class=\"option-choice\">"
                f"<input type=\"radio\" name=\"{escape(str(feature))}\" value=\"{idx}\" "
                f"data-option=\"{escape(option_name)}\" "
                f"data-splits=\"{escape(json.dumps(splits))}\" "
                f"data-status=\"{escape(str(status))}\">{escape(option_name)}</label>"
                f"<div class=\"option-status {escape(str(status).lower())}\">Status: {escape(str(status))}</div>"
                f"<div class=\"option-splits\">Splits: {escape(str(splits))}</div>"
                f"</td>"
            )
        checked = " checked" if idx == 0 else ""
        status_rows.append(
            f"<tr>"
            f"<th class=\"part-label\">{escape(part)}</th>"
            f"{''.join(option_cells)}"
            f"</tr>"
        )

    status_html = f"""
    <table class="status-table" data-feature="{escape(str(feature))}">
        <thead>
            <tr><th>Part</th><th>Free optimization</th><th>Monotonic</th><th>U-shape / heuristic</th></tr>
        </thead>
        <tbody>{''.join(status_rows)}</tbody>
    </table>
    """

    return f"""
    <section class="feature">
        <h2>{escape(str(feature))}</h2>
        {status_html}
        <h3>WOE Trend &amp; Bin Count Comparison</h3>
        <img class="plot" src="data:image/png;base64,{plot_html}" alt="WOE plots for {escape(str(feature))}">
        <h3>WOE Tables</h3>
        {tables_html}
    </section>
    <hr class="feature-separator">
    """


def build_report(df:pd.DataFrame, label_name = "LABEL"):
    df = df.drop(columns=[c for c in IGNORE_COLUMN if c in df.columns])
    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)
    features = [column for column in X_train.columns if column != label_name and not str(column).startswith(label_name)]

    sections = []
    skipped = []

    for feature in features:
        try:
            sections.append(build_feature_html(feature, X_train, y))
            print(f"OK: {feature}")
        except Exception as exc:
            skipped.append((feature, exc))
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")

    skipped_html = ""
    if skipped:
        skipped_rows = "".join(
            f"<tr><td>{escape(str(feature))}</td><td>{escape(type(exc).__name__)}</td><td>{escape(str(exc))}</td></tr>"
            for feature, exc in skipped
        )
        skipped_html = f"""
        <section class="skipped">
            <h2>Skipped features</h2>
            <table class="status-table">
                <thead><tr><th>Feature</th><th>Error</th><th>Message</th></tr></thead>
                <tbody>{skipped_rows}</tbody>
            </table>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WOE Report - All Features</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h1 {{ margin-bottom: 8px; }}
h1.title {{ text-align: center; font-size: 40px; color: #111; margin: 16px 0 4px; }}
h2 {{ margin-top: 0; }}
h3 {{ margin-top: 24px; }}
.plot {{ display: block; max-width: 100%; height: auto; }}
table, th, td {{ border: 1px solid #ccc; border-collapse: collapse; }}
th, td {{ padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; }}
.status-table th, .status-table td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table th {{ background: #f3f3f3; }}
.status-table .part-label {{ background: #e9ecef; font-weight: bold; width: 180px; }}
.status-table .option-choice {{ display: flex; align-items: center; gap: 6px; }}
.status-table .option-status {{ color: #555; font-size: 0.9em; margin-top: 4px; font-weight: bold; }}
.status-table .option-status.optimal {{ color: #16a34a; }}
.status-table .option-status.infeasible {{ color: #dc2626; }}
.status-table .option-splits {{ color: #777; font-size: 0.85em; }}
.woe-table-container {{ display: flex; flex-direction: column; gap: 20px; }}
.woe-row {{ display: flex; align-items: flex-start; gap: 12px; }}
.woe-row-label {{ font-weight: bold; background: #e9ecef; padding: 8px; border-radius: 4px; min-width: 160px; flex: 0 0 auto; }}
.woe-row-columns {{ display: flex; flex-direction: row; gap: 15px; width: 100%; overflow-x: auto; }}
.woe-block {{ flex: 1 1 0; min-width: 320px; overflow-x: auto; }}
.woe-table-container table {{ border-collapse: separate; table-layout: auto; width: 100%; }}
.woe-table-container td, .woe-table-container th {{ white-space: nowrap; }}
.skipped {{ margin-top: 40px; }}
.feature-separator {{ border: 0; border-top: 3px solid #999; margin: 36px 0; }}
</style>
</head>
<body>
<div style="position: sticky; top: 0; background: #fff; padding: 10px; border-bottom: 2px solid #ccc; z-index: 1000;">
    <button onclick="exportConfig()" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
        Export Selected Splits (JSON)
    </button>
</div>
<h1 class="title">{escape(str(label_name))}</h1>
<p>Features are processed independently. A feature that raises an exception is skipped.</p>
{''.join(sections)}
{skipped_html}
<script>
const APP_CONFIG = {json.dumps(CONFIG, ensure_ascii=False)};
{EXPORT_SCRIPT}
</script>
</body>
</html>
"""