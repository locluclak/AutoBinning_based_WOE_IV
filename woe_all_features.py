import base64
import io
from html import escape
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from optbinning import OptimalBinning

INPUT_FILE = "export_ftr_loan_1.csv"
OUTPUT_FILE = "woe_all_features2.html"
label_name = "LABEL_IS_CASA_50M_ACTUAL_BAL_LCL"
MIN_BIN = 4
MAX_BIN = 10
SPECIALVALUE = 0
MIN_DIFF_WOE = 0.01

def create_woe_df(x, y, splits):
    """
    Create WOE / IV table from manually specified or optimal splits.

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
        Bin, prob_n_obs, WOE, IV_detail, IV_total
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
    """
    Return observation count and proportion for each bin.
    Missing values are excluded because x used for optimization
    has already removed missing values.
    """
    bin_edges = [-np.inf] + list(splits) + [np.inf]

    bins = pd.cut(x, bins=bin_edges, right=True, include_lowest=True)

    stats = (pd.DataFrame({'bin': bins, 'target': y})
        .groupby('bin', observed=False)
        .size()
        .reset_index(name='n_obs')
    )
    stats['prob_n_obs'] = stats['n_obs'] / len(x)
    return stats

def plot_three_options(results, feature, x, y):
    """
    Compare 3 optimal binning strategies.

    Each subplot:
        - Left Y-axis  : WOE
        - Right Y-axis : Bin Count
        - X-axis       : Bin
    """

    fig, axes = plt.subplots(1, 3, figsize=(21, 6))

    for ax_woe, (option_name, result) in zip(axes, results.items()):
        splits = result['splits']
        woe_df = create_woe_df(x, y, splits)
        woe_plot = woe_df[woe_df['Bin'] != 'Missing'].copy()

        bin_stats = get_bin_stats(x=x, y=y, splits=splits)

        n_bins = len(bin_stats)
        x_pos = np.arange(n_bins)

        ax_woe.plot(x_pos, woe_plot['WOE'].values, marker='o', linewidth=2, label='WOE')
        ax_woe.axhline(0, linestyle='--', linewidth=1)

        ax_woe.set_ylabel('WOE')
        ax_woe.set_xlabel('Bin')

        ax_woe.set_xticks(x_pos)
        ax_woe.set_xticklabels(bin_stats['bin'].astype(str), rotation=45, ha='right')
        ax_woe.grid(axis='y', alpha=0.3)

        ax_count = ax_woe.twinx()
        ax_count.bar(x_pos, bin_stats['n_obs'].values, alpha=0.25, width=0.6, label='Bin Count')
        ax_count.set_ylabel('Bin Count')

        ax_woe.set_title(option_name, fontsize=12)

        handles_woe, labels_woe = ax_woe.get_legend_handles_labels()
        handles_count, labels_count = ax_count.get_legend_handles_labels()

        ax_woe.legend(handles_woe + handles_count, labels_woe + labels_count, loc='best', fontsize=8)

    fig.suptitle(f'{feature} - WOE Trend & Bin Count Comparison', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()

def display_woe_tables(results: dict, x, y_clean, create_woe_df_func, formats: dict = None, render: bool = True):
    """Generates and displays side-by-side formatted HTML WOE tables from results.
    """
    if formats is None:
        formats = {
            "prob_n_obs": "{:.2%}",
            "pct_event": "{:.2%}",
            "pct_non_event": "{:.2%}",
            "WOE": "{:.2f}",
            "IV_detail": "{:.2f}",
            "IV_total": "{:.2f}",
        }

    print("WOE TABLES")
    woe_tables = {}
    html_blocks = []

    for idx, (option_name, result) in enumerate(results.items()):
        splits = result["splits"]
        woe_df = create_woe_df_func(x, y_clean, splits)
        woe_tables[option_name] = woe_df

        valid_formats = {k: v for k, v in formats.items() if k in woe_df.columns}
        styler = woe_df.style.format(valid_formats)

        if idx > 0:
            styler.hide(axis="index")

        styled_html = styler.to_html()

        block = f"""
        <div style="flex: 1; min-width: 1;">
            <h4 style="margin-bottom: 8px;">{option_name}</h4>
            {styled_html}
        </div>
        """
        html_blocks.append(block)

    flex_container = f"""
    <div style="display: flex; flex-direction: row; gap: 15px; width: 100%; overflow-x: auto;">
        {''.join(html_blocks)}
    </div>
    """

    if render:
        pass
        # display(HTML(flex_container))

    return woe_tables, flex_container

def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def calculate_feature(feature, X_train, y):
    x_original = X_train[feature].copy()

    zero_pct = (x_original == SPECIALVALUE).mean()

    if zero_pct < 0.05:
        mask_nonmissing = x_original.notna()
    else:
        mask_nonmissing = x_original.notna() & (x_original != SPECIALVALUE)

    x = x_original.loc[mask_nonmissing]
    y_clean = y.loc[mask_nonmissing]

    n_old = len(X_train)
    n_new = len(x)

    if n_new == 0:
        raise ValueError("Feature has no usable observations.")

    min_bin_size = 0.05 #* n_old / n_new
    max_bin_size = min(0.50 * n_old / n_new, 1)

    p_bar = np.mean(y_clean)
    min_event_rate_diff = 2 * p_bar * (1 - p_bar) * np.tanh(MIN_DIFF_WOE / 2)

    options = {
        "1. Free optimization": "auto",
        "2. Monotonic": "auto_asc_desc",
        "3. U-shape / heuristic": "auto_heuristic"
    }

    results = {}

    for option_name, trend in options.items():
        optb = OptimalBinning(name=feature, dtype="numerical", min_n_bins=MIN_BIN, max_n_bins=MAX_BIN, min_bin_size=min_bin_size, max_bin_size=max_bin_size, min_event_rate_diff=min_event_rate_diff, monotonic_trend=trend)
        optb.fit(x, y_clean)
        results[option_name] = {
            "model": optb,
            "splits": optb.splits,
            "binning_table": optb.binning_table
        }

    return results, x, y_clean


def build_feature_html(feature, X_train, y):
    results, x, y_clean = calculate_feature(feature, X_train, y)

    plot_three_options(results, feature, x, y_clean)
    fig = plt.gcf()
    plot_html = figure_to_base64(fig)

    _, tables_html = display_woe_tables(results=results, x=x, y_clean=y_clean, create_woe_df_func=create_woe_df, render=False)

    status_rows = []
    for option_name, result in results.items():
        status_rows.append(
            f"<tr><td>{escape(option_name)}</td><td>{escape(str(result['model'].status))}</td><td>{escape(str(list(map(float, result['splits']))))}</td></tr>"
        )

    status_html = f"""
    <table class="status-table">
        <thead>
            <tr><th>Option</th><th>Status</th><th>Optimal splits</th></tr>
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


def build_report(df, label_name = "LABEL"):
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
.feature {{ margin-bottom: 56px; padding-bottom: 40px; border-bottom: 1px solid #ddd; }}
.plot {{ display: block; max-width: 100%; height: auto; }}
.status-table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; }}
.status-table th, .status-table td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table th {{ background: #f3f3f3; }}
.skipped {{ margin-top: 40px; }}
</style>
</head>
<body>
<h1>WOE Report - All Features</h1>
<p>Features are processed independently. A feature that raises an exception is skipped.</p>
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
            raise ValueError(f"Định dạng file không được hỗ trợ: {path.suffix}")
    
def main():
    df = load_data(INPUT_FILE)
    html = build_report(df, label_name)
    Path(OUTPUT_FILE).write_text(html, encoding="utf-8")
    print(f"HTML report: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
