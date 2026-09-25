from html import escape

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from core.woe_stats import (add_score_column, chi2_contingency_test, chi2_cramers,
                            chi2_cramers_categorical, cramer_v_color, cramer_v_type,
                            create_missing_woe_df, create_woe_df, create_woe_df_categorical,
                            figure_to_base64, format_p_value, get_bin_stats,
                            get_bin_stats_categorical, is_categorical, special_mask)

DEFAULT_FORMATS = {
    "prob_n_obs": "{:.2%}",
    "pct_event": "{:.2%}",
    "pct_non_event": "{:.2%}",
    "conversion_rate": "{:.2%}",
    "WOE": "{:.2f}",
    "IV_detail": "{:.2f}",
    "IV_total": "{:.2f}",
    "score": "{:.4f}",
}

PAGE_CSS = """
body { font-family: Arial, sans-serif; margin: 24px; color: #222; }
h1.title { text-align: center; font-size: 40px; color: #111; margin: 16px 0 4px; }
h2 { margin-top: 0; }
.feature h2, .feature-card h2 { font-size: 28px; }
h3 { margin-top: 24px; }
.plot { display: block; max-width: 100%; height: auto; }
.plot-part { margin-bottom: 20px; }
.plot-row { display: flex; flex-wrap: wrap; gap: 15px; }
.plot-figure { flex: 1 1 0; min-width: 320px; }
.feature-meta { margin: 4px 0; font-size: 0.9em; }
.iv-total { color: #555; font-size: 0.9em; font-weight: normal; }
.iv-label { color: #555; font-weight: normal; }
.p-value { font-size: 0.9em; }
.stat-label { font-weight: normal; }
.cramers { margin: 4px 0; font-size: 0.9em; }
.option { color: #555; }
.splits { color: #777; font-size: 0.9em; }
table, th, td { border: 1px solid #ccc; border-collapse: collapse; }
th, td { padding: 6px 8px; text-align: left; vertical-align: top; }
.status-table { border-collapse: collapse; margin: 12px 0 20px; width: 100%; }
.status-table th { background: #f3f3f3; }
.woe-table-container { display: flex; flex-direction: column; gap: 20px; }
.woe-row { display: flex; flex-direction: column; gap: 8px; }
.woe-row-label { font-weight: bold; background: #e9ecef; padding: 8px; border-radius: 4px; align-self: flex-start; }
.woe-row-columns { display: flex; flex-direction: row; gap: 15px; width: 100%; overflow-x: auto; }
.woe-block { flex: 1 1 0; min-width: 320px; overflow-x: auto; }
.woe-table-container table { border-collapse: separate; table-layout: auto; width: 100%; }
.woe-table-container td, .woe-table-container th { white-space: nowrap; }
.skipped { margin-top: 40px; }
.feature-separator { border: 0; border-top: 3px solid #999; margin: 36px 0; }
.summary { margin: 16px 0 24px; }
.summary-metrics { display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }
.metric { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px 18px; text-align: center; min-width: 120px; }
.metric .value { font-size: 26px; font-weight: bold; color: #111; }
.metric .label { color: #555; font-size: 0.85em; }
.summary-tables { display: flex; flex-direction: column; gap: 12px; }
.summary-tables details { background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 8px 12px; }
.summary-tables summary { font-weight: bold; cursor: pointer; color: #333; }
.summary-table { border-collapse: collapse; table-layout: fixed; width: 100%; background: #f8f9fa; border: 1px solid #dee2e6; }
.summary-table th { border: 1px solid #dee2e6; padding: 8px 12px; background: #e9ecef; font-weight: bold; color: #333; }
.summary-table td { border: 1px solid #dee2e6; padding: 8px 12px; vertical-align: top; white-space: normal; line-height: 1.5; }
.compare-wrap { display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin: 8px 0 24px; }
.compare-col { min-width: 0; }
.version-title { font-size: 20px; color: #111; border-bottom: 2px solid #ccc; padding-bottom: 6px; }
.compare-table { margin-top: 8px; }
.compare-row { display: grid; grid-template-columns: 1fr 1fr; gap: 12px; border-bottom: 3px solid #999; padding: 8px 0 24px; }
.compare-cell { min-width: 0; vertical-align: top; }
.compare-empty { color: #999; font-style: italic; border: 1px dashed #ccc; padding: 24px; text-align: center; background: #f8f9fa; }
"""


def woe_table_html(woe_df, v_c):
    display_df = add_score_column(woe_df, v_c).drop(columns=["IV_total"], errors="ignore")
    valid_formats = {k: v for k, v in DEFAULT_FORMATS.items() if k in display_df.columns}
    return display_df.style.format(valid_formats).to_html()


def build_missing_table_html(x, y):
    labels = pd.Series(["Missing" if pd.isna(v) else "Non-Missing" for v in x],
                       index=x.index, dtype=object)
    p_value, v_c = chi2_contingency_test(labels, pd.Series(y, index=x.index))
    missing_woe_df = add_score_column(create_missing_woe_df(x, y), v_c)
    iv_total = missing_woe_df["IV_total"].iloc[0] if "IV_total" in missing_woe_df.columns else None
    display_df = missing_woe_df.drop(columns=["IV_total"], errors="ignore")
    valid_formats = {k: v for k, v in DEFAULT_FORMATS.items() if k in display_df.columns}
    table_html = display_df.style.format(valid_formats).to_html()

    meta_parts = []
    if iv_total is not None:
        iv_color = "#16a34a" if iv_total >= 0.2 else "#dc2626"
        meta_parts.append(
            f'<span class="iv-total">'
            f'<span class="iv-label">IV total:</span> '
            f'<b class="iv-value" style="color:{iv_color}">{iv_total:.2f}</b>'
            f'</span>'
        )
    p_color = "#16a34a" if p_value <= 0.05 else "#dc2626"
    meta_parts.append(
        f'<span class="p-value" style="color:{p_color}">'
        f'<span class="stat-label">p-value:</span> '
        f'<b>{format_p_value(p_value)}</b>'
        f'</span>'
    )
    title = " ".join(meta_parts)

    v_color = cramer_v_color(v_c)
    stats_html = (
        f'<div style="margin-bottom: 2px; color:{v_color}; font-size: 0.9em;">'
        f'<span class="stat-label">Cramer&apos;s V:</span> <b>{v_c:.4f}</b>'
        f' &nbsp;-&nbsp; '
        f'<span class="stat-label">Type:</span> <b>{cramer_v_type(v_c)}</b>'
        f'</div>'
    )

    return f"""
    <div class="woe-block">
        <div style="margin-bottom: 6px;">{title}</div>
        {stats_html}
        {table_html}
    </div>
    """


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


def build_feature_card(feature, cfg, x_full, y, special):
    """Build the feature card (WOE plot + tables + meta) for one version.

    Returns (html, meta) where meta is a dict for the summary section.
    """
    cfg_type = cfg.get('type')
    categorical = cfg_type == 'categorical' or (cfg_type is None and is_categorical(x_full))
    part = cfg.get('part', 'consider SPECIAL')
    consider_special = part.startswith("consider")
    option = cfg.get('option', '')

    if categorical:
        splits = [str(c) for c in cfg.get('splits', [])]
        if not splits:
            splits = [str(c) for c in pd.unique(x_full.dropna())]
        x = x_full if consider_special else x_full[~special_mask(x_full, special=special)]
        y_clean = y if consider_special else y[~special_mask(x_full, special=special)]
        woe_df = create_woe_df_categorical(x, y_clean, missing_first=consider_special)
        p_value, v_c = chi2_cramers_categorical(x, y_clean)
    else:
        splits = [float(s) for s in cfg.get('splits', [])]
        if consider_special:
            x, y_clean = x_full, y
            feature_special = list(special) if special else []
        else:
            mask = ~special_mask(x_full, special=special)
            x, y_clean = x_full[mask], y[mask]
            feature_special = []
        woe_df = create_woe_df(x, y_clean, splits, missing_first=consider_special, special=feature_special)
        p_value, v_c = chi2_cramers(x, y_clean, splits, special=feature_special)

    iv_total = float(woe_df['IV_total'].iloc[0])
    plot_b64 = plot_feature(feature, x, y_clean, splits, special=feature_special if not categorical else None,
                            missing_first=consider_special, categorical=categorical)
    missing_html = build_missing_table_html(x_full, y)
    woe_table = woe_table_html(woe_df, v_c)

    iv_color = "#16a34a" if iv_total >= 0.2 else "#dc2626"
    meta_html = (
        f'<div class="feature-meta">'
        f'<span class="iv-total"><span class="iv-label">IV total:</span> '
        f'<b class="iv-value" style="color:{iv_color}">{iv_total:.2f}</b></span>'
        f'<span class="p-value" style="color:{"#16a34a" if p_value <= 0.05 else "#dc2626"}">'
        f'<span class="stat-label">p-value:</span> <b>{format_p_value(p_value)}</b></span>'
        f'</div>'
    )
    v_color = cramer_v_color(v_c)
    stats_html = (
        f'<div class="cramers" style="color:{v_color}">'
        f'<span class="stat-label">Cramer&apos;s V:</span> <b>{v_c:.4f}</b>'
        f' &nbsp;-&nbsp; '
        f'<span class="stat-label">Type:</span> <b>{cramer_v_type(v_c)}</b>'
        f'</div>'
    )

    if categorical:
        option_html = f'<p class="option">Selected version: {escape(part)}</p>'
        splits_html = f'<p class="splits">Groups: {escape("[" + ", ".join(splits) + "]")}</p>'
    else:
        option_html = f'<p class="option">Selected option: {escape(option)}</p>'
        formatted_splits = "[" + ", ".join(f"{v:,}" for v in splits) + "]"
        splits_html = f'<p class="splits">Splits: {escape(formatted_splits)}</p>'

    html = f"""
    <div class="feature-card" data-feature="{escape(str(feature))}" data-type="{'categorical' if categorical else 'continuous'}">
        <h2>{escape(str(feature))}</h2>
        {meta_html}
        {stats_html}
        {option_html}
        {splits_html}
        <h3>WOE Trend &amp; Bin Count Comparison</h3>
        <div class="plot-row"><div class="plot-figure">
            <img class="plot" src="data:image/png;base64,{plot_b64}" alt="WOE plot for {escape(str(feature))}">
        </div></div>
        <h3>WOE Tables</h3>
        <div class="woe-table-container">
            <div class="woe-row">
                <div class="woe-row-label">Missing vs Non-Missing</div>
                <div class="woe-row-columns">{missing_html}</div>
            </div>
            <div class="woe-row">
                <div class="woe-row-label">{escape(part)}</div>
                <div class="woe-row-columns"><div class="woe-block">{woe_table}</div></div>
            </div>
        </div>
    </div>
    """

    ftype = "categorical" if categorical else "continuous"
    meta = {
        "feature": feature,
        "type": ftype,
        "iv_total": iv_total,
        "v_type": cramer_v_type(v_c),
        "has_optimal": True,
        "all_infeasible": False,
    }
    return html, meta


def build_summary_html(meta):
    """Build the summary section shown at the top of the report (selected-report look)."""
    if not meta:
        return ""
    total = len(meta)
    continuous = [m for m in meta if m['type'] == 'continuous']
    categorical = [m for m in meta if m['type'] == 'categorical']

    def cells(items):
        if not items:
            return ""
        return "<br>".join(escape(str(m['feature'])) + "," for m in items)

    def summary_table(summary, headers, columns):
        header_cells = "".join(
            f"<th>{escape(header)} ({len(col)})</th>"
            for header, col in zip(headers, columns)
        )
        body_cells = "".join(f"<td>{cells(col)}</td>" for col in columns)
        return f"""
        <details>
            <summary>{escape(summary)}</summary>
            <table class="summary-table">
                <thead><tr>{header_cells}</tr></thead>
                <tbody><tr>{body_cells}</tr></tbody>
            </table>
        </details>
        """

    table_types = summary_table(
        "Continuous vs Categorical",
        ["Continuous features", "Categorical features"], [continuous, categorical])

    vt_labels = ("Strong", "Good", "Medium", "Weak")
    vtype_groups = [[m for m in meta if m['v_type'] == vt] for vt in vt_labels]
    table_vtypes = summary_table("Cramer's V type", list(vt_labels), vtype_groups)

    iv_headers = ["IV total \u2264 0.2", "0.2 < IV total \u2264 0.5", "0.5 < IV total \u2264 1", "IV total > 1"]
    iv_groups = [
        [m for m in meta if m['iv_total'] <= 0.2],
        [m for m in meta if 0.2 < m['iv_total'] <= 0.5],
        [m for m in meta if 0.5 < m['iv_total'] <= 1],
        [m for m in meta if m['iv_total'] > 1],
    ]
    table_iv = summary_table("IV total", iv_headers, iv_groups)

    metrics = (
        f'<div class="metric"><div class="value">{total}</div><div class="label">Total features</div></div>'
        f'<div class="metric"><div class="value">{sum(m.get("has_optimal", False) for m in meta)}</div>'
        f'<div class="label">With optimal solution</div></div>'
        f'<div class="metric"><div class="value">{sum(m.get("all_infeasible", False) for m in meta)}</div>'
        f'<div class="label">All INFEASIBLE</div></div>'
    )

    return f"""
    <section class="summary">
        <h2>Summary</h2>
        <div class="summary-metrics">{metrics}</div>
        <div class="summary-tables">{table_types}{table_vtypes}{table_iv}</div>
    </section>
    """


def build_skipped_html(skipped):
    if not skipped:
        return ""
    skipped_rows = "".join(
        f"<tr><td>{escape(str(feature))}</td><td>{escape(str(msg))}</td></tr>"
        for feature, msg in skipped
    )
    return f"""
    <section class="skipped">
        <h2>Skipped features</h2>
        <table class="status-table">
            <thead><tr><th>Feature</th><th>Message</th></tr></thead>
            <tbody>{skipped_rows}</tbody>
        </table>
    </section>
    """


def build_report(df, versions, label_name, min_bin_size=0.05):
    """Build the reconstructed HTML report.

    versions : list of dicts with keys:
        label   - short name of the version (e.g. JSON file stem)
        config  - features_config (feature -> {type, part, option, splits})
        special - special values list

    One version  -> stacked feature sections (selected-report look).
    Two versions -> side-by-side comparison, same feature on the same row.
    """
    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)

    version_data = []
    for version in versions:
        cfg = version.get("config", {})
        special = version.get("special") or []
        results = {}
        meta = []
        skipped = []
        for feature, fcfg in cfg.items():
            if feature == label_name or str(feature).startswith(str(label_name)):
                continue
            if feature not in X_train.columns:
                skipped.append((feature, "Missing in data"))
                continue
            try:
                card, card_meta = build_feature_card(feature, fcfg, X_train[feature], y, special)
                results[feature] = card
                meta.append(card_meta)
                print(f"OK: {feature}")
            except Exception as exc:
                skipped.append((feature, exc))
                print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")
        version_data.append({
            "label": version.get("label", ""),
            "results": results,
            "meta": meta,
            "skipped": skipped,
        })

    if len(version_data) == 1:
        body = build_single_body(version_data[0])
    else:
        body = build_compare_body(version_data)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WOE Report - Reconstructed</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<h1 class="title">{escape(str(label_name))}</h1>
<p>Reconstructed from exported splits. No OptimalBinning used.</p>
{body}
</body>
</html>
"""


def build_single_body(version):
    sections = []
    for card in version["results"].values():
        sections.append(f"<section class=\"feature\">{card}<hr class=\"feature-separator\"></section>")
    return f"""
{build_summary_html(version["meta"])}
{''.join(sections)}
{build_skipped_html(version["skipped"])}
"""


def build_compare_body(version_data):
    left, right = version_data[0], version_data[1]

    ordered = []
    for v in version_data:
        for f in v["results"]:
            if f not in ordered:
                ordered.append(f)

    rows = []
    for feature in ordered:
        left_cell = left["results"].get(feature)
        right_cell = right["results"].get(feature)
        rows.append(f"""
        <div class="compare-row">
            <div class="compare-cell">{left_cell if left_cell is not None else empty_cell("Not present in this version")}</div>
            <div class="compare-cell">{right_cell if right_cell is not None else empty_cell("Not present in this version")}</div>
        </div>
        """)

    left_summary = (f'<h2 class="version-title">{escape(str(left["label"]))}</h2>'
                    + build_summary_html(left["meta"]))
    right_summary = (f'<h2 class="version-title">{escape(str(right["label"]))}</h2>'
                     + build_summary_html(right["meta"]))

    return f"""
<div class="compare-wrap">
    <div class="compare-col">{left_summary}</div>
    <div class="compare-col">{right_summary}</div>
</div>
<div class="compare-table">
{''.join(rows)}
</div>
{build_skipped_html(left["skipped"])}
{build_skipped_html(right["skipped"])}
"""


def empty_cell(text):
    return f'<div class="compare-empty">{escape(text)}</div>'