from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.woe_stats import (add_score_column, chi2_cramers, chi2_cramers_categorical,
                            cramer_v_color, cramer_v_type, create_woe_df,
                            create_woe_df_categorical, figure_to_base64, format_p_value,
                            get_bin_stats, get_bin_stats_categorical, is_categorical, special_mask)


def woe_table_html(woe_df, v_c):
    formats = {
        "prob_n_obs": "{:.2%}",
        "pct_event": "{:.2%}",
        "pct_non_event": "{:.2%}",
        "WOE": "{:.4f}",
        "IV_detail": "{:.4f}",
        "IV_total": "{:.4f}",
        "score": "{:.4f}",
    }
    display_df = add_score_column(woe_df, v_c).drop(columns=["IV_total"])
    valid_formats = {k: v for k, v in formats.items() if k in display_df.columns}
    return display_df.style.format(valid_formats).to_html()


def plot_feature(feature, x, y_clean, splits, special=None, missing_first=False, categorical=False):
    fig, ax = plt.subplots(figsize=(7, 6))
    if categorical:
        woe_df = create_woe_df_categorical(x, y_clean, missing_first=missing_first)
        bin_stats = get_bin_stats_categorical(x=x, y=y_clean, missing_first=missing_first)
    else:
        woe_df = create_woe_df(x, y_clean, splits, missing_first=missing_first, special=special)
        bin_stats = get_bin_stats(x=x, y=y_clean, splits=splits, missing_first=missing_first, special=special)
    if missing_first:
        woe_plot = woe_df.copy()
    else:
        woe_plot = woe_df[woe_df['Bin'] != 'Missing'].copy()
        bin_stats = bin_stats[bin_stats['bin'] != 'Missing'].copy()

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


def build_feature_html(feature, x, y_clean, splits, option, special=None, consider_special=False,
                       categorical=False, part=""):
    plot_html = plot_feature(feature, x, y_clean, splits, special=special,
                             missing_first=consider_special, categorical=categorical)
    if categorical:
        woe_df = create_woe_df_categorical(x, y_clean, missing_first=consider_special)
    else:
        woe_df = create_woe_df(x, y_clean, splits, missing_first=consider_special, special=special)

    if categorical:
        p_value, v_c = chi2_cramers_categorical(x, y_clean)
    else:
        p_value, v_c = chi2_cramers(x, y_clean, splits, special=special)
    tables_html = woe_table_html(woe_df, v_c)
    iv_total = woe_df['IV_total'].iloc[0]

    p_color = "#16a34a" if p_value <= 0.05 else "#dc2626"
    p_html = f'<span style="color:{p_color}">p-value: {format_p_value(p_value)}</span>'
    v_color = cramer_v_color(v_c)
    v_html = (
        f'<span style="color:{v_color}; font-size: 0.9em;">'
        f"Cramer's V: {v_c:.4f} - Type: {cramer_v_type(v_c)}</span>"
    )

    if categorical:
        option_html = f'<p class="option">Selected version: {escape(part)}</p>'
        splits_html = f'<p class="splits">Groups: {escape(str(splits))}</p>'
    else:
        option_html = f'<p class="option">Selected option: {escape(option)}</p>'
        splits_html = f'<p class="splits">Splits: {escape(str(splits))}</p>'

    return f"""
    <section class="feature">
        <h2>{escape(str(feature))} <span class="iv">IV: {iv_total:.4f}</span> {p_html}</h2>
        <p class="cramers">{v_html}</p>
        {option_html}
        {splits_html}
        <img class="plot" src="data:image/png;base64,{plot_html}" alt="WOE plot for {escape(str(feature))}">
        <h3>WOE Table</h3>
        {tables_html}
    </section>
    """


def build_report(df, features_config, label_name, special=None, min_bin_size=0.05):
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
            cfg_type = cfg.get('type')
            series = X_train[feature]
            part = cfg.get('part', 'considerSPECIAL')
            consider_special = part in ("considerSPECIAL", "considerMISSING")
            x_full = X_train[feature]
            if cfg_type == 'categorical' or (cfg_type is None and is_categorical(series)):
                categories = [str(c) for c in cfg.get('splits', [])]
                if not categories:
                    categories = [str(c) for c in pd.unique(series.dropna())]
                option = cfg.get('option', '')
                if consider_special:
                    x = x_full
                    y_clean = y
                else:
                    mask = ~special_mask(x_full, special=special)
                    x = x_full[mask]
                    y_clean = y[mask]
                sections.append(build_feature_html(feature, x, y_clean, categories, option,
                                                   consider_special=consider_special,
                                                   categorical=True, part=part))
            else:
                splits = list(map(float, cfg['splits']))
                option = cfg.get('option', '')
                if consider_special:
                    x = x_full
                    y_clean = y
                    feature_special = list(special) if special else []
                else:
                    mask = ~special_mask(x_full, special=special)
                    x = x_full[mask]
                    y_clean = y[mask]
                    feature_special = []
                sections.append(build_feature_html(feature, x, y_clean, splits, option,
                                                   special=feature_special,
                                                   consider_special=consider_special))
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
.cramers {{ margin: 4px 0; font-size: 0.9em; }}
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