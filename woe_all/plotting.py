import matplotlib.pyplot as plt
import numpy as np

from core.woe_stats import create_woe_df, get_bin_stats


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