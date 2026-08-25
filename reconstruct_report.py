import base64
import io
import json
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_woe_df(x, y, splits):
    """
    Create WOE / IV table from manually specified splits.

    Parameters
    ----------
    x : pd.Series
        Feature values.
    y : pd.Series
        Binary target: 0 = non-event, 1 = event.
    splits : array-like
        Bin split points.

    Returns
    -------
    pd.DataFrame
        Bin, prob_n_obs, pct_event, pct_non_event, WOE, IV_detail, IV_total
    """
    data = pd.DataFrame({'x': x, 'target': y})
    bin_edges = [-np.inf] + list(splits) + [np.inf]
    data['bin'] = pd.cut(data['x'], bins=bin_edges, right=True, include_lowest=True)

    grouped = (data.groupby('bin', observed=False)['target']
        .agg(n_events=lambda s: (s == 1).sum(), n_non_events=lambda s: (s == 0).sum())
        .reset_index())

    missing_mask = data['x'].isna()

    if missing_mask.any():
        missing_row = pd.DataFrame({
            'bin': ['Missing'],
            'n_events': [(missing_mask & (data['target'] == 1)).sum()],
            'n_non_events': [(missing_mask & (data['target'] == 0)).sum()]
        })
        grouped = pd.concat([grouped, missing_row], ignore_index=True)

    total_obs = len(data)
    total_events = (data['target'] == 1).sum()
    total_non_events = (data['target'] == 0).sum()

    grouped['prob_n_obs'] = (grouped['n_events'] + grouped['n_non_events']) / total_obs
    grouped['pct_event'] = grouped['n_events'] / total_events if total_events > 0 else 0
    grouped['pct_non_event'] = grouped['n_non_events'] / total_non_events if total_non_events > 0 else 0

    eps = 1e-6
    pct_event_adj = np.where(grouped['pct_event'] == 0, eps, grouped['pct_event'])
    pct_non_event_adj = np.where(grouped['pct_non_event'] == 0, eps, grouped['pct_non_event'])

    grouped['WOE'] = np.log(pct_event_adj / pct_non_event_adj)
    grouped['IV_detail'] = (pct_event_adj - pct_non_event_adj) * grouped['WOE']
    grouped['IV_total'] = grouped['IV_detail'].sum()
    grouped['Bin'] = grouped['bin'].astype(str)

    return grouped[['Bin', 'prob_n_obs', 'pct_event', 'pct_non_event', 'WOE', 'IV_detail', 'IV_total']].copy()


def get_bin_stats(x, y, splits):
    bin_edges = [-np.inf] + list(splits) + [np.inf]
    bins = pd.cut(x, bins=bin_edges, right=True, include_lowest=True)
    stats = (pd.DataFrame({'bin': bins, 'target': y})
        .groupby('bin', observed=False)
        .size()
        .reset_index(name='n_obs')
    )
    stats['prob_n_obs'] = stats['n_obs'] / len(x)
    return stats


def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def plot_feature(feature, x, y_clean, splits):
    fig, ax = plt.subplots(figsize=(7, 6))
    woe_df = create_woe_df(x, y_clean, splits)
    woe_plot = woe_df[woe_df['Bin'] != 'Missing'].copy()
    bin_stats = get_bin_stats(x=x, y=y_clean, splits=splits)

    n_bins = len(bin_stats)
    x_pos = np.arange(n_bins)

    ax.plot(x_pos, woe_plot['WOE'].values, marker='o', linewidth=2, label='WOE')
    ax.axhline(0, linestyle='--', linewidth=1)
    ax.set_ylabel('WOE')
    ax.set_xlabel('Bin')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(bin_stats['bin'].astype(str), rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    ax_count = ax.twinx()
    ax_count.bar(x_pos, bin_stats['n_obs'].values, alpha=0.25, width=0.6, label='Bin Count')
    ax_count.set_ylabel('Bin Count')

    ax.set_title(f'{feature}', fontsize=12)

    handles_woe, labels_woe = ax.get_legend_handles_labels()
    handles_count, labels_count = ax_count.get_legend_handles_labels()
    ax.legend(handles_woe + handles_count, labels_woe + labels_count, loc='best', fontsize=8)

    fig.tight_layout()
    return figure_to_base64(fig)


def woe_table_html(woe_df):
    formats = {
        "prob_n_obs": "{:.2%}",
        "pct_event": "{:.2%}",
        "pct_non_event": "{:.2%}",
        "WOE": "{:.2f}",
        "IV_detail": "{:.2f}",
        "IV_total": "{:.2f}",
    }
    valid_formats = {k: v for k, v in formats.items() if k in woe_df.columns}
    return woe_df.style.format(valid_formats).to_html()


def build_feature_html(feature, x, y_clean, splits, option):
    plot_html = plot_feature(feature, x, y_clean, splits)
    woe_df = create_woe_df(x, y_clean, splits)
    tables_html = woe_table_html(woe_df)
    iv_total = woe_df['IV_total'].iloc[0]

    return f"""
    <section class="feature">
        <h2>{escape(str(feature))} <span class="iv">IV: {iv_total:.4f}</span></h2>
        <p class="option">Selected option: {escape(option)}</p>
        <p class="splits">Splits: {escape(str(splits))}</p>
        <img class="plot" src="data:image/png;base64,{plot_html}" alt="WOE plot for {escape(str(feature))}">
        <h3>WOE Table</h3>
        {tables_html}
    </section>
    """


def build_report(df, features_config, label_name):
    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)

    sections = []
    skipped = []

    for feature, cfg in features_config.items():
        if feature == label_name or str(feature).startswith(label_name):
            continue
        if feature not in X_train.columns:
            skipped.append((feature, "Missing in data"))
            continue
        try:
            splits = list(map(float, cfg['splits']))
            option = cfg.get('option', '')
            x = X_train[feature]
            sections.append(build_feature_html(feature, x, y, splits, option))
            print(f"OK: {feature}")
        except Exception as exc:
            skipped.append((feature, exc))
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")

    skipped_html = ""
    if skipped:
        skipped_rows = "".join(
            f"<tr><td>{escape(str(feature))}</td><td>{escape(str(msg))}</td></tr>"
            for feature, msg in skipped
        )
        skipped_html = f"""
        <section class="skipped">
            <h2>Skipped features</h2>
            <table class="status-table">
                <thead><tr><th>Feature</th><th>Message</th></tr></thead>
                <tbody>{skipped_rows}</tbody>
            </table>
        </section>
        """

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WOE Report - Reconstructed</title>
<style>
body {{ font-family: Arial, sans-serif; margin: 24px; color: #222; }}
h2 {{ margin-top: 0; }}
h3 {{ margin-top: 24px; }}
.plot {{ display: block; max-width: 40%; height: auto; }}
.iv {{ color: #007bff; font-weight: normal; font-size: 0.8em; }}
.option {{ color: #555; }}
.splits {{ color: #777; font-size: 0.9em; }}
table, th, td {{ border: 1px solid #ccc; border-collapse: collapse; }}
th, td {{ padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; }}
.status-table th {{ background: #f3f3f3; }}
.skipped {{ margin-top: 40px; }}
</style>
</head>
<body>
<h1>WOE Report - Reconstructed</h1>
<p>Reconstructed from exported splits. No OptimalBinning used.</p>
{''.join(sections)}
{skipped_html}
</body>
</html>
"""


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)
    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(file_path)
        case ".parquet":
            return pd.read_parquet(file_path)
        case ".xlsx" | ".xls":
            return pd.read_excel(file_path)
        case ".json":
            return pd.read_json(file_path)
        case _:
            raise ValueError(f"Unsupported file format: {path.suffix}")


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