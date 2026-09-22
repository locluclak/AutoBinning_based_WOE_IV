import matplotlib.pyplot as plt
import numpy as np

from core.woe_stats import create_woe_df, create_woe_df_categorical, get_bin_stats, get_bin_stats_categorical


def _plot_single(result, option_name):
    """Plot WOE (left axis) and Bin Count (right axis) for a single option."""
    x = result['x']
    y = result['y']
    splits = result['splits']
    missing_first = result.get('missing_first', False)
    special = result.get('special')
    if result.get('categorical'):
        woe_df = create_woe_df_categorical(x, y, missing_first=missing_first)
        bin_stats = get_bin_stats_categorical(x=x, y=y, missing_first=missing_first)
    else:
        woe_df = create_woe_df(x, y, splits, missing_first=missing_first, special=special)
        bin_stats = get_bin_stats(x=x, y=y, splits=splits, missing_first=missing_first, special=special)

    n_bins = len(bin_stats)
    x_pos = np.arange(n_bins)

    fig, ax_woe = plt.subplots(figsize=(7, 6))

    ax_woe.plot(x_pos, woe_df['WOE'].values, marker='o', linewidth=2, label='WOE')
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

    return fig


def plot_options(results, feature):
    """
    Compare all optimal binning strategies, grouped by part.

    Returns one figure PER option so the report can lay out the options in a
    row and the exported (selection-only) HTML can keep a single figure.

    Each figure:
        - Left Y-axis  : WOE
        - Right Y-axis : Bin Count
        - X-axis       : Bin

    Returns
    -------
    list of (part_label, [(option_name, fig), ...])
    """
    parts = {}
    for option_key, result in results.items():
        parts.setdefault(result.get('part', ''), []).append((option_key, result))

    out = []
    for part, items in parts.items():
        figs = []
        for option_key, result in items:
            option_name = result.get('option', option_key)
            figs.append((option_name, _plot_single(result, option_name)))
        out.append((part, figs))

    return out