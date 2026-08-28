from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.woe_stats import create_woe_df, figure_to_base64, get_bin_stats


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
h1.title {{ text-align: center; font-size: 40px; color: #111; margin: 16px 0 4px; }}
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
<h1 class="title">{escape(str(label_name))}</h1>
<p>Reconstructed from exported splits. No OptimalBinning used.</p>
{''.join(sections)}
{skipped_html}
</body>
</html>
"""