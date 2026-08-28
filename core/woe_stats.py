import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_woe_df(x, y, splits, missing_first=False, specialvalue=None):
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

    if specialvalue is not None:
        special_mask = (data['x'] == specialvalue)
        data_cut = data[~special_mask].copy()
    else:
        special_mask = pd.Series(False, index=data.index)
        data_cut = data

    bin_edges = [-np.inf] + list(splits) + [np.inf]
    data_cut['bin'] = pd.cut(data_cut['x'], bins=bin_edges, right=True, include_lowest=True)

    grouped = (data_cut.groupby('bin', observed=False)['target']
        .agg(n_events=lambda s: (s == 1).sum(), n_non_events=lambda s: (s == 0).sum())
        .reset_index())

    missing_mask = data['x'].isna()

    if missing_mask.any():
        missing_row = pd.DataFrame({
            'bin': ['Missing'],
            'n_events': [(missing_mask & (data['target'] == 1)).sum()],
            'n_non_events': [(missing_mask & (data['target'] == 0)).sum()]
        })
        if missing_first:
            grouped = pd.concat([missing_row, grouped], ignore_index=True)
        else:
            grouped = pd.concat([grouped, missing_row], ignore_index=True)

    if special_mask.any():
        special_row = pd.DataFrame({
            'bin': [f'Special {specialvalue}'],
            'n_events': [(special_mask & (data['target'] == 1)).sum()],
            'n_non_events': [(special_mask & (data['target'] == 0)).sum()]
        })
        if missing_first and missing_mask.any():
            grouped = pd.concat([grouped.iloc[:1], special_row, grouped.iloc[1:]], ignore_index=True)
        else:
            grouped = pd.concat([special_row, grouped], ignore_index=True)

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


def get_bin_stats(x, y, splits, missing_first=False, specialvalue=None):
    """
    Return observation count and proportion for each bin.
    Missing values are excluded because x used for optimization
    has already removed missing values.
    """
    bin_edges = [-np.inf] + list(splits) + [np.inf]

    if specialvalue is not None:
        special_mask = (x == specialvalue)
        bins = pd.cut(x[~special_mask], bins=bin_edges, right=True, include_lowest=True)
        stats = (pd.DataFrame({'bin': bins, 'target': y[~special_mask]})
            .groupby('bin', observed=False)
            .size()
            .reset_index(name='n_obs')
        )
    else:
        special_mask = pd.Series(False, index=x.index)
        bins = pd.cut(x, bins=bin_edges, right=True, include_lowest=True)
        stats = (pd.DataFrame({'bin': bins, 'target': y})
            .groupby('bin', observed=False)
            .size()
            .reset_index(name='n_obs')
        )
    stats['prob_n_obs'] = stats['n_obs'] / len(x)

    missing_mask = x.isna()
    if missing_first and missing_mask.any():
        n_missing = int(missing_mask.sum())
        missing_row = pd.DataFrame({
            'bin': ['Missing'],
            'n_obs': [n_missing],
            'prob_n_obs': [n_missing / len(x)]
        })
        stats = pd.concat([missing_row, stats], ignore_index=True)

    if special_mask.any():
        n_special = int(special_mask.sum())
        special_row = pd.DataFrame({
            'bin': [f'Special {specialvalue}'],
            'n_obs': [n_special],
            'prob_n_obs': [n_special / len(x)]
        })
        if missing_first and missing_mask.any():
            stats = pd.concat([stats.iloc[:1], special_row, stats.iloc[1:]], ignore_index=True)
        else:
            stats = pd.concat([special_row, stats], ignore_index=True)

    return stats


def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")