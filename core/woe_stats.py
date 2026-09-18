import base64
import io

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def is_categorical(series):
    """Detect whether a feature should be treated as categorical (no OptimalBinning)."""
    if pd.api.types.is_categorical_dtype(series.dtype):
        return True
    if pd.api.types.is_bool_dtype(series.dtype):
        return True
    if pd.api.types.is_object_dtype(series.dtype):
        return True
    if pd.api.types.is_string_dtype(series.dtype):
        return True
    return False


def _normalize_special(special=None, specialvalue=None):
    """Return a list of special groups.

    'MISSING' inside the list denotes the missing-value group; the other
    elements are raw feature values (e.g. 0, 'SME') matched by equality.
    """
    if special is not None:
        return list(special)
    if specialvalue is not None:
        return [specialvalue]
    return []


def special_mask(series, special=None, specialvalue=None):
    """Boolean mask of rows belonging to a special group (missing or listed value)."""
    special_values = _normalize_special(special, specialvalue)
    mask = pd.Series(False, index=series.index)
    for v in special_values:
        if v == "MISSING":
            mask = mask | series.isna()
        else:
            try:
                mask = mask | (series == v)
            except Exception:
                continue
    return mask


def _finalize_woe(grouped, total_obs, total_events, total_non_events):
    grouped = grouped.copy()

    grouped['prob_n_obs'] = (grouped['n_events'] + grouped['n_non_events']) / total_obs
    grouped['pct_event'] = grouped['n_events'] / total_events if total_events > 0 else 0
    grouped['pct_non_event'] = grouped['n_non_events'] / total_non_events if total_non_events > 0 else 0

    n_obs = grouped['n_events'] + grouped['n_non_events']
    grouped['conversion_rate'] = np.where(n_obs > 0, grouped['n_events'] / n_obs, np.nan)

    eps = 1e-6
    pct_event_adj = np.where(grouped['pct_event'] == 0, eps, grouped['pct_event'])
    pct_non_event_adj = np.where(grouped['pct_non_event'] == 0, eps, grouped['pct_non_event'])

    grouped['WOE'] = np.log(pct_event_adj / pct_non_event_adj)
    grouped['IV_detail'] = (pct_event_adj - pct_non_event_adj) * grouped['WOE']
    grouped['IV_total'] = grouped['IV_detail'].sum()
    grouped['Bin'] = grouped['bin'].astype(str)

    return grouped[['Bin', 'prob_n_obs', 'pct_event', 'pct_non_event', 'conversion_rate',
                    'WOE', 'IV_detail', 'IV_total']].copy()


def create_woe_df(x, y, splits, missing_first=False, specialvalue=None, special=None):
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
    missing_first : bool
        If True, the Missing group is placed first (before the cut bins).
    special : list
        Special groups rendered as their own bins: 'MISSING' for the missing
        group and/or raw values (e.g. 0, 'SME') labelled 'Special {value}'.
        Rows in these groups are excluded from the cut bins.

    Returns
    -------
    pd.DataFrame
        Bin, prob_n_obs, WOE, IV_detail, IV_total
    """

    data = pd.DataFrame({'x': x, 'target': y}).reset_index(drop=True)

    special_values = _normalize_special(special, specialvalue)
    value_specials = [v for v in special_values if v != 'MISSING']

    keep_mask = data['x'].notna()
    if value_specials:
        keep_mask = keep_mask & ~data['x'].isin(value_specials)

    data_cut = data[keep_mask].copy()
    bin_edges = [-np.inf] + list(splits) + [np.inf]
    data_cut['bin'] = pd.cut(data_cut['x'], bins=bin_edges, right=True, include_lowest=True)

    bins_df = (data_cut.groupby('bin', observed=False)['target']
        .agg(n_events=lambda s: (s == 1).sum(), n_non_events=lambda s: (s == 0).sum())
        .reset_index())

    special_df = None
    if value_specials:
        special_rows = []
        for v in value_specials:
            mask = data['x'] == v
            if not mask.any():
                continue
            special_rows.append({
                'bin': f'Special {v}',
                'n_events': int((mask & (data['target'] == 1)).sum()),
                'n_non_events': int((mask & (data['target'] == 0)).sum()),
            })
        if special_rows:
            special_df = pd.DataFrame(special_rows)

    missing_df = None
    missing_mask = data['x'].isna()
    if missing_mask.any():
        missing_df = pd.DataFrame({
            'bin': ['Missing'],
            'n_events': [int((missing_mask & (data['target'] == 1)).sum())],
            'n_non_events': [int((missing_mask & (data['target'] == 0)).sum())],
        })

    if missing_first:
        grouped = pd.concat(
            [d for d in (missing_df, bins_df, special_df) if d is not None and len(d)],
            ignore_index=True)
    else:
        grouped = pd.concat(
            [d for d in (special_df, bins_df, missing_df) if d is not None and len(d)],
            ignore_index=True)

    total_obs = len(data)
    total_events = (data['target'] == 1).sum()
    total_non_events = (data['target'] == 0).sum()

    return _finalize_woe(grouped, total_obs, total_events, total_non_events)


def create_woe_df_categorical(x, y, missing_first=False):
    """
    Create WOE / IV table for a categorical feature: each group (category value)
    is treated as one bin. No optimal binning is required.

    Parameters
    ----------
    x : pd.Series
        Feature values (string / object / bool dtype).
    y : pd.Series
        Binary target: 0 = non-event, 1 = event.
    missing_first : bool
        If True, the Missing group is placed first.

    Returns
    -------
    pd.DataFrame
        Bin, prob_n_obs, WOE, IV_detail, IV_total
    """

    data = pd.DataFrame({'x': x, 'target': y})

    non_null = data['x'].notna()
    cats = list(pd.unique(data.loc[non_null, 'x']))

    if non_null.any():
        grouped = (data.loc[non_null].groupby('x', sort=False)['target']
            .agg(n_events=lambda s: (s == 1).sum(), n_non_events=lambda s: (s == 0).sum())
            .reindex(cats)
            .reset_index()
            .rename(columns={'x': 'bin'}))
    else:
        grouped = pd.DataFrame(columns=['bin', 'n_events', 'n_non_events'])

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

    total_obs = len(data)
    total_events = (data['target'] == 1).sum()
    total_non_events = (data['target'] == 0).sum()

    woe_df = _finalize_woe(grouped, total_obs, total_events, total_non_events)
    return _sort_categorical_bins(woe_df, missing_first=missing_first)


def _sort_categorical_bins(df, prob_col='prob_n_obs', bin_col='Bin', missing_first=False):
    """Sort categorical bins by prop_n_obs ascending; keep Missing in its fixed position.

    Used so WOE tables and plots order rare categories first consistently,
    while the Missing group stays first (missing_first=True) or last.
    """
    if df is None or df.empty:
        return df

    has_missing = (df[bin_col] == 'Missing').any()
    if not has_missing:
        return df.sort_values(prob_col, kind='stable').reset_index(drop=True)

    if missing_first:
        head = df[df[bin_col] == 'Missing']
        tail = df[df[bin_col] != 'Missing'].sort_values(prob_col, kind='stable')
    else:
        head = df[df[bin_col] != 'Missing'].sort_values(prob_col, kind='stable')
        tail = df[df[bin_col] == 'Missing']
    return pd.concat([head, tail], ignore_index=True)


def create_missing_woe_df(x, y):
    """
    Create a WOE / IV table with only two groups: Missing and Non-Missing.
    Used as additional evidence when deciding how to treat missing values.
    """
    data = pd.DataFrame({'x': x, 'target': y})
    missing_mask = data['x'].isna()

    rows = []
    for name, mask in (("Non-Missing", ~missing_mask), ("Missing", missing_mask)):
        rows.append({
            "bin": name,
            "n_events": int((mask & (data['target'] == 1)).sum()),
            "n_non_events": int((mask & (data['target'] == 0)).sum()),
        })
    grouped = pd.DataFrame(rows)

    total_obs = len(data)
    total_events = (data['target'] == 1).sum()
    total_non_events = (data['target'] == 0).sum()

    return _finalize_woe(grouped, total_obs, total_events, total_non_events)


CRAMER_TYPES = [
    ("Strong", 0.5),
    ("Good", 0.3),
    ("Medium", 0.1),
    ("Weak", 0.0),
]

CRAMER_COLORS = {
    "Strong": "#007bff",
    "Good": "#16a34a",
    "Medium": "#f59e0b",
    "Weak": "#dc2626",
}


def chi2_contingency_test(bin_labels, target):
    """Chi-square test of independence between a binned feature and a binary target.

    Parameters
    ----------
    bin_labels : pd.Series
        Bin label for each observation.
    target : pd.Series
        Binary target aligned with bin_labels.

    Returns
    -------
    (p_value, cramers_v)
    """
    from scipy.stats import chi2_contingency

    cont = pd.crosstab(pd.Series(bin_labels, name="bin"), pd.Series(target, name="target"))
    table = np.asarray(cont.values, dtype=float)
    n = table.sum()
    r, c = table.shape
    if n == 0 or r <= 1 or c <= 1:
        return 1.0, 0.0
    chi2_stat, p_value, _, _ = chi2_contingency(table, correction=False)
    v_c = float(np.sqrt(chi2_stat / (n * min(r - 1, c - 1))))
    return float(p_value), v_c


def chi2_cramers(x, y, splits, specialvalue=None, special=None):
    """Chi-square test of a numeric feature (binned by splits) vs binary target.

    Missing values and the special values form their own groups.

    Returns
    -------
    (p_value, cramers_v)
    """
    data = pd.DataFrame({"x": x, "target": y}).reset_index(drop=True)
    special_values = _normalize_special(special, specialvalue)
    value_specials = [v for v in special_values if v != 'MISSING']

    keep = data["x"].notna()
    if value_specials:
        keep = keep & ~data["x"].isin(value_specials)

    bin_edges = [-np.inf] + list(splits) + [np.inf]
    labels = pd.Series(np.nan, index=data.index, dtype=object)
    cut = pd.cut(data.loc[keep, "x"], bins=bin_edges, right=True, include_lowest=True)
    labels.loc[keep] = [str(b) for b in cut]

    for v in value_specials:
        labels.loc[data["x"] == v] = f"Special {v}"
    labels.loc[data["x"].isna()] = "Missing"
    return chi2_contingency_test(labels, data["target"])


def chi2_cramers_categorical(x, y):
    """Chi-square test of a categorical feature vs binary target.

    Missing values form their own group.

    Returns
    -------
    (p_value, cramers_v)
    """
    data = pd.DataFrame({"x": x, "target": y}).reset_index(drop=True)
    labels = pd.Series([str(v) if pd.notna(v) else "Missing" for v in data["x"]],
                       index=data.index, dtype=object)
    return chi2_contingency_test(labels, data["target"])


def cramer_v_type(v_c):
    """Classify Cramer's V into Strong / Good / Medium / Weak."""
    for name, threshold in CRAMER_TYPES:
        if v_c >= threshold:
            return name
    return "Weak"


def cramer_v_color(v_c):
    """Return the text color associated with a Cramer's V value."""
    return CRAMER_COLORS[cramer_v_type(v_c)]


def format_p_value(p_value):
    """Format a p-value: fixed decimals for common values, scientific below 0.001."""
    if p_value < 0.001:
        return f"{p_value:.2e}"
    return f"{p_value:.4f}"


def add_score_column(woe_df, v_c):
    """Add 'score' column: score = sign(WOE) * V_c * (IV_bin / IV_total) * 100."""
    df = woe_df.copy()
    iv_total = df["IV_total"].iloc[0]
    if iv_total == 0:
        df["score"] = 0.0
    else:
        df["score"] = np.sign(df["WOE"]) * v_c * df["IV_detail"] / iv_total * 100
    return df


def get_bin_stats(x, y, splits, missing_first=False, specialvalue=None, special=None):
    """
    Return observation count and proportion for each bin (including special
    groups rendered as their own bins, mirroring create_woe_df ordering).
    """
    special_values = _normalize_special(special, specialvalue)
    value_specials = [v for v in special_values if v != 'MISSING']

    keep = x.notna()
    if value_specials:
        keep = keep & ~x.isin(value_specials)

    bin_edges = [-np.inf] + list(splits) + [np.inf]
    bins = pd.cut(x[keep], bins=bin_edges, right=True, include_lowest=True)
    stats = (pd.DataFrame({'bin': bins, 'target': y[keep]})
        .groupby('bin', observed=False)
        .size()
        .reset_index(name='n_obs'))
    stats['prob_n_obs'] = stats['n_obs'] / len(x)

    special_stats = None
    if value_specials:
        special_rows = []
        for v in value_specials:
            mask = x == v
            if not mask.any():
                continue
            special_rows.append({
                'bin': f'Special {v}',
                'n_obs': int(mask.sum()),
                'prob_n_obs': mask.sum() / len(x),
            })
        if special_rows:
            special_stats = pd.DataFrame(special_rows)

    missing_stats = None
    missing_mask = x.isna()
    if missing_mask.any():
        missing_stats = pd.DataFrame({
            'bin': ['Missing'],
            'n_obs': [int(missing_mask.sum())],
            'prob_n_obs': [missing_mask.sum() / len(x)],
        })

    if missing_first:
        stats = pd.concat(
            [d for d in (missing_stats, stats, special_stats) if d is not None and len(d)],
            ignore_index=True)
    else:
        stats = pd.concat(
            [d for d in (special_stats, stats, missing_stats) if d is not None and len(d)],
            ignore_index=True)

    return stats


def get_bin_stats_categorical(x, y=None, missing_first=False):
    """
    Return observation count and proportion for each category group.
    Missing values form their own 'Missing' group when present.
    """
    data = pd.DataFrame({'x': x})

    non_null = data['x'].notna()
    cats = list(pd.unique(data.loc[non_null, 'x']))

    if non_null.any():
        stats = (data.loc[non_null].groupby('x', sort=False)
            .size()
            .reindex(cats)
            .reset_index(name='n_obs')
            .rename(columns={'x': 'bin'}))
    else:
        stats = pd.DataFrame(columns=['bin', 'n_obs'])

    stats['prob_n_obs'] = stats['n_obs'] / len(data)

    missing_mask = data['x'].isna()
    if missing_first and missing_mask.any():
        n_missing = int(missing_mask.sum())
        missing_row = pd.DataFrame({
            'bin': ['Missing'],
            'n_obs': [n_missing],
            'prob_n_obs': [n_missing / len(data)]
        })
        stats = pd.concat([missing_row, stats], ignore_index=True)

    return _sort_categorical_bins(stats, prob_col='prob_n_obs', bin_col='bin',
                                  missing_first=missing_first)


def figure_to_base64(fig):
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buffer.getvalue()).decode("ascii")