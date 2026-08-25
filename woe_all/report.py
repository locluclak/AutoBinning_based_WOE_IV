import json
from html import escape

import matplotlib.pyplot as plt
import pandas as pd

from core.config_loader import CONFIG, IGNORE_COLUMN
from core.export_script import EXPORT_SCRIPT
from core.woe_stats import create_woe_df, figure_to_base64
from .html_tables import display_woe_tables
from .optimization import calculate_feature
from .plotting import plot_three_options


def build_feature_html(feature, X_train, y):
    results, x, y_clean = calculate_feature(feature, X_train, y)

    plot_three_options(results, feature, x, y_clean)
    fig = plt.gcf()
    plot_html = figure_to_base64(fig)

    _, tables_html = display_woe_tables(results=results, x=x, y_clean=y_clean, create_woe_df_func=create_woe_df, render=False)

    status_rows = []
    for idx, (option_name, result) in enumerate(results.items()):
        splits = list(map(float, result['splits']))
        checked = " checked" if idx == 0 else ""
        status_rows.append(
            f"<tr>"
            f"<td><input type=\"radio\" name=\"{escape(str(feature))}\" value=\"{idx}\" "
            f"data-option=\"{escape(option_name)}\" "
            f"data-splits=\"{escape(json.dumps(splits))}\"{checked}></td>"
            f"<td>{escape(option_name)}</td>"
            f"<td>{escape(str(result['model'].status))}</td>"
            f"<td>{escape(str(splits))}</td>"
            f"</tr>"
        )

    status_html = f"""
    <table class="status-table" data-feature="{escape(str(feature))}">
        <thead>
            <tr><th>Select</th><th>Option</th><th>Status</th><th>Optimal splits</th></tr>
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
h2 {{ margin-top: 0; }}
h3 {{ margin-top: 24px; }}
.plot {{ display: block; max-width: 100%; height: auto; }}
table, th, td {{ border: 1px solid #ccc; border-collapse: collapse; }}
th, td {{ padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; }}
.status-table th, .status-table td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table th {{ background: #f3f3f3; }}
.skipped {{ margin-top: 40px; }}
</style>
</head>
<body>
<div style="position: sticky; top: 0; background: #fff; padding: 10px; border-bottom: 2px solid #ccc; z-index: 1000;">
    <button onclick="exportConfig()" style="padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer;">
        Export Selected Splits (JSON)
    </button>
</div>
<h1>WOE Report - All Features</h1>
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