import matplotlib.pyplot as plt
import numpy as np

from core.woe_stats import create_woe_df, create_woe_df_categorical, get_bin_stats, get_bin_stats_categorical


def plot_options(results, feature):
    """
    Compare all optimal binning strategies (one subplot per option).

    Each subplot:
        - Left Y-axis  : WOE
        - Right Y-axis : Bin Count
        - X-axis       : Bin
    """

    n = len(results)
    cols = 3
    rows = max(int(np.ceil(n / cols)), 1)

    fig, axes = plt.subplots(rows, cols, figsize=(7 * cols, 6 * rows))
    axes = np.atleast_2d(axes).reshape(-1)

    for ax_woe, (option_name, result) in zip(axes, results.items()):
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

    for ax in axes[n:]:
        ax.axis('off')

    fig.suptitle(f'{feature} - WOE Trend & Bin Count Comparison', fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.show()