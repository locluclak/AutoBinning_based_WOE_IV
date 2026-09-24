import json
from html import escape

import pandas as pd

from core.config_loader import CONFIG, IGNORE_COLUMN
from core.export_script import EXPORT_SCRIPT
from core.woe_stats import (add_score_column, chi2_contingency_test, chi2_cramers,
                            chi2_cramers_categorical, cramer_v_color, cramer_v_type,
                            create_missing_woe_df, create_woe_df, create_woe_df_categorical,
                            figure_to_base64, format_p_value, is_categorical)
from .html_tables import (compute_result_stats, compute_result_woe_df, display_woe_tables,
                          option_meta_html, option_stats_html)
from .optimization import calculate_categorical_feature, calculate_feature
from .plotting import plot_options


MISSING_FORMATS = {
    "prob_n_obs": "{:.2%}",
    "pct_event": "{:.2%}",
    "pct_non_event": "{:.2%}",
    "conversion_rate": "{:.2%}",
    "WOE": "{:.2f}",
    "IV_detail": "{:.2f}",
    "IV_total": "{:.2f}",
    "score": "{:.4f}",
}


def build_missing_table_html(x, y):
    labels = pd.Series(["Missing" if pd.isna(v) else "Non-Missing" for v in x],
                       index=x.index, dtype=object)
    p_value, v_c = chi2_contingency_test(labels, pd.Series(y, index=x.index))
    missing_woe_df = add_score_column(create_missing_woe_df(x, y), v_c)
    iv_total = missing_woe_df["IV_total"].iloc[0] if "IV_total" in missing_woe_df.columns else None
    display_df = missing_woe_df.drop(columns=["IV_total"], errors="ignore")
    valid_formats = {k: v for k, v in MISSING_FORMATS.items() if k in display_df.columns}
    table_html = display_df.style.format(valid_formats).to_html()

    meta_parts = []
    if iv_total is not None:
        iv_total_text = MISSING_FORMATS.get("IV_total", "{:.2f}").format(iv_total)
        iv_color = "#16a34a" if iv_total >= 0.2 else "#dc2626"
        meta_parts.append(
            f'<span class="iv-total">'
            f'<span class="iv-label">IV total:</span> '
            f'<b class="iv-value" style="color:{iv_color}">{iv_total_text}</b>'
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
    <div class="woe-table-container">
        <div class="woe-row">
            <div class="woe-row-label">Missing vs Non-Missing</div>
            <div class="woe-row-columns">
                <div class="woe-block">
                    <div style="margin-bottom: 6px;">{title}</div>
                    {stats_html}
                    {table_html}
                </div>
            </div>
        </div>
    </div>
    """


def compute_feature_results(feature, X_train, y):
    """Compute all binning options/parts for a feature.

    Shared by the static HTML report and the Streamlit app.
    Returns (results, parts, option_names).
    """
    results = {}
    if is_categorical(X_train[feature]):
        for consider in (False, True):
            results.update(calculate_categorical_feature(feature, X_train, y, considerSPECIAL=consider))
    else:
        for consider in (False, True):
            results.update(calculate_feature(feature, X_train, y, considerSPECIAL=consider))

    parts = {}
    for option_name, result in results.items():
        parts.setdefault(result['part'], {})[result['option']] = result

    option_names = []
    for result in results.values():
        if result['option'] not in option_names:
            option_names.append(result['option'])

    return results, parts, option_names


def feature_summary(feature, X_train, y, results):
    """Return (feature_type, iv_total, cramer_type) used for filtering.

    iv_total / Cramer's V type come from the best (max IV) optimal option.
    """
    ftype = "categorical" if is_categorical(X_train[feature]) else "continuous"
    best = None
    for result in results.values():
        if result['model'].status not in ("OPTIMAL", "OK"):
            continue
        if result.get('categorical'):
            woe_df = create_woe_df_categorical(result['x'], result['y'],
                                               missing_first=result.get('missing_first', False))
            _, v_c = chi2_cramers_categorical(result['x'], result['y'])
        else:
            woe_df = create_woe_df(result['x'], result['y'], result['splits'],
                                   missing_first=result.get('missing_first', False),
                                   special=result.get('special'))
            _, v_c = chi2_cramers(result['x'], result['y'], result['splits'],
                                  special=result.get('special'))
        iv = float(woe_df['IV_total'].iloc[0])
        if best is None or iv > best[0]:
            best = (iv, v_c)
    if best is None:
        best = (0.0, 0.0)
    return ftype, best[0], cramer_v_type(best[1])


def build_summary_html(meta):
    """Build the summary section shown at the top of the HTML report.

    meta : list of dicts with keys feature/type/iv_total/v_type.
    """
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


def build_feature_html(feature, X_train, y, results=None, parts=None, option_names=None):
    if results is None:
        results, parts, option_names = compute_feature_results(feature, X_train, y)

    plot_sections = []
    for part, figures in plot_options(results, feature):
        figure_html = "".join(
            f'<div class="plot-figure" data-part="{escape(part)}" data-option="{escape(option_name)}">'
            f'<img class="plot" src="data:image/png;base64,{figure_to_base64(fig)}" '
            f'alt="WOE plot for {escape(str(feature))} - {escape(option_name)}">'
            f'</div>'
            for option_name, fig in figures
        )
        plot_sections.append(
            f'<div class="plot-part" data-part="{escape(part)}">'
            f'<div class="woe-group-title">{escape(part)}</div>'
            f'<div class="plot-row">{figure_html}</div>'
            f'</div>'
        )
    plot_html = "".join(plot_sections)

    _, tables_html = display_woe_tables(results=results, create_woe_df_func=create_woe_df, render=False)

    missing_html = build_missing_table_html(X_train[feature], y)

    ftype, iv_total, v_type = feature_summary(feature, X_train, y, results)

    statuses = [r["model"].status for r in results.values()]
    has_optimal = any(s in ("OPTIMAL", "OK") for s in statuses)
    all_infeasible = bool(statuses) and all(s == "INFEASIBLE" for s in statuses)

    status_rows = []
    for idx, (part, part_results) in enumerate(parts.items()):
        option_cells = []
        for option_name in option_names:
            result = part_results.get(option_name)
            if result is None:
                option_cells.append("<td></td>")
                continue
            splits = list(result['splits'])
            if not result.get('categorical'):
                splits = list(map(float, splits))
            ftype = "categorical" if result.get('categorical') else "continuous"
            status = result['model'].status

            woe_df = compute_result_woe_df(result, create_woe_df)
            stats = compute_result_stats(result)
            meta_html = option_meta_html(woe_df, stats)
            stats_html = option_stats_html(stats)

            option_cells.append(
                f"<td>"
                f"<label class=\"option-choice\">"
                f"<input type=\"radio\" name=\"{escape(str(feature))}\" value=\"{idx}\" "
                f"data-option=\"{escape(str(option_name))}\" "
                f"data-part=\"{escape(str(result['part']))}\" "
                f"data-type=\"{ftype}\" "
                f"data-splits=\"{escape(json.dumps(splits))}\" "
                f"data-status=\"{escape(str(status))}\">{escape(option_name)}</label>"
                f"<div class=\"option-metrics\">{meta_html}{stats_html}</div>"
                f"<div class=\"option-status {escape(str(status).lower())}\">Status: {escape(str(status))}</div>"
                f"<div class=\"option-splits\">Splits: {escape(str(splits))}</div>"
                f"</td>"
            )
        checked = " checked" if idx == 0 else ""
        status_rows.append(
            f"<tr>"
            f"<th class=\"part-label\">{escape(part)}</th>"
            f"{''.join(option_cells)}"
            f"</tr>"
        )

    option_headers = "".join(f"<th>{escape(o)}</th>" for o in option_names)
    status_html = f"""
    <table class="status-table" data-feature="{escape(str(feature))}">
        <thead>
            <tr><th>Part</th>{option_headers}</tr>
        </thead>
        <tbody>{''.join(status_rows)}</tbody>
    </table>
    """

    return f"""
    <section class="feature" data-feature="{escape(str(feature))}" data-type="{ftype}" data-iv="{iv_total:.4f}" data-vtype="{v_type}"
        data-has-optimal="{str(has_optimal).lower()}" data-all-infeasible="{str(all_infeasible).lower()}">
        <h2>{escape(str(feature))}</h2>
        {status_html}
        <h3>WOE Trend &amp; Bin Count Comparison</h3>
        {plot_html}
        <h3>WOE Tables</h3>
        {missing_html}
        {tables_html}
        <hr class="feature-separator">
    </section>
    """


def build_report(df:pd.DataFrame, label_name = "LABEL"):
    df = df.drop(columns=[c for c in IGNORE_COLUMN if c in df.columns])
    X_train = df.drop(columns=[label_name])
    y = pd.Series(df[label_name].values, index=X_train.index)
    features = [column for column in X_train.columns if column != label_name and not str(column).startswith(label_name)]

    sections = []
    skipped = []
    feature_meta = []

    for feature in features:
        try:
            results, parts, option_names = compute_feature_results(feature, X_train, y)
            ftype, iv_total, v_type = feature_summary(feature, X_train, y, results)
            statuses = [r["model"].status for r in results.values()]
            feature_meta.append({
                "feature": feature,
                "type": ftype,
                "iv_total": iv_total,
                "v_type": v_type,
                "has_optimal": any(s in ("OPTIMAL", "OK") for s in statuses),
                "all_infeasible": bool(statuses) and all(s == "INFEASIBLE" for s in statuses),
            })
            sections.append(build_feature_html(feature, X_train, y, results, parts, option_names))
            print(f"OK: {feature}")
        except Exception as exc:
            skipped.append((feature, exc))
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")

    summary_html = build_summary_html(feature_meta)

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
h1.title {{ text-align: center; font-size: 40px; color: #111; margin: 16px 0 4px; }}
h2 {{ margin-top: 0; }}
.feature h2 {{ font-size: 28px; }}
h3 {{ margin-top: 24px; }}
.plot {{ display: block; max-width: 100%; height: auto; }}
.plot-part {{ margin-bottom: 20px; }}
.plot-row {{ display: flex; flex-wrap: wrap; gap: 15px; }}
.plot-figure {{ flex: 1 1 0; min-width: 320px; }}
.woe-group-title {{ margin: 8px 0 6px; font-weight: bold; background: #e9ecef; padding: 8px; border-radius: 4px; display: inline-block; }}
table, th, td {{ border: 1px solid #ccc; border-collapse: collapse; }}
th, td {{ padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table {{ border-collapse: collapse; margin: 12px 0 20px; width: 100%; }}
.status-table th, .status-table td {{ border: 1px solid #ccc; padding: 6px 8px; text-align: left; vertical-align: top; }}
.status-table th {{ background: #f3f3f3; }}
.status-table .part-label {{ background: #e9ecef; font-weight: bold; width: 180px; }}
.status-table .option-choice {{ display: flex; align-items: center; gap: 6px; }}
.status-table .option-status {{ color: #555; font-size: 0.9em; margin-top: 4px; font-weight: bold; }}
.status-table .option-metrics {{ color: #555; font-size: 0.9em; margin-top: 4px; }}
.status-table .option-status.optimal {{ color: #16a34a; }}
.status-table .option-status.infeasible {{ color: #dc2626; }}
.status-table .option-splits {{ color: #777; font-size: 0.85em; }}
.woe-table-container {{ display: flex; flex-direction: column; gap: 20px; }}
.woe-row {{ display: flex; flex-direction: column; gap: 8px; }}
.woe-row-label {{ font-weight: bold; background: #e9ecef; padding: 8px; border-radius: 4px; align-self: flex-start; }}
.woe-row-columns {{ display: flex; flex-direction: row; gap: 15px; width: 100%; overflow-x: auto; }}
.woe-block {{ flex: 1 1 0; min-width: 320px; overflow-x: auto; }}
.iv-total {{ color: #555; font-size: 0.9em; font-weight: normal; }}
.iv-label {{ color: #555; font-weight: normal; }}
.p-value {{ font-size: 0.9em; }}
.stat-label {{ font-weight: normal; }}
.woe-table-container table {{ border-collapse: separate; table-layout: auto; width: 100%; }}
.woe-table-container td, .woe-table-container th {{ white-space: nowrap; }}
.skipped {{ margin-top: 40px; }}
.feature-separator {{ border: 0; border-top: 3px solid #999; margin: 36px 0; }}
.filter-bar {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px 14px; display: flex; flex-wrap: wrap; gap: 14px; align-items: center; }}
.filter-group {{ display: flex; align-items: center; gap: 8px; flex-wrap: wrap; }}
.filter-label {{ font-weight: bold; color: #333; }}
.filter-bar input[type="number"] {{ width: 90px; padding: 4px 6px; border: 1px solid #ccc; border-radius: 4px; }}
.filter-bar input[type="text"] {{ width: 300px; padding: 4px 6px; border: 1px solid #ccc; border-radius: 4px; }}
.filter-bar label {{ display: inline-flex; align-items: center; gap: 4px; cursor: pointer; }}
#filter-count {{ font-weight: bold; color: #007bff; margin-left: auto; }}
.report-header {{ position: sticky; top: 0; background: #fff; padding: 8px 16px; border-bottom: 2px solid #ccc; box-shadow: 0 2px 8px rgba(0,0,0,0.15); z-index: 1000; display: flex; flex-direction: column; gap: 6px; }}
.export-btn {{ align-self: flex-start; padding: 8px 16px; background: #007bff; color: white; border: none; border-radius: 4px; cursor: pointer; font-size: 14px; }}
.export-btns {{ display: flex; gap: 10px; }}
.summary {{ margin: 16px 0 24px; }}
.summary-metrics {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 16px; }}
.metric {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 10px 18px; text-align: center; min-width: 120px; }}
.metric .value {{ font-size: 26px; font-weight: bold; color: #111; }}
.metric .label {{ color: #555; font-size: 0.85em; }}
.summary-tables {{ display: flex; flex-direction: column; gap: 12px; }}
.summary-tables details {{ background: #f8f9fa; border: 1px solid #dee2e6; border-radius: 6px; padding: 8px 12px; }}
.summary-tables summary {{ font-weight: bold; cursor: pointer; color: #333; }}
.summary-table {{ border-collapse: collapse; table-layout: fixed; width: 100%; background: #f8f9fa; border: 1px solid #dee2e6; }}
.summary-table th {{ border: 1px solid #dee2e6; padding: 8px 12px; background: #e9ecef; font-weight: bold; color: #333; }}
.summary-table td {{ border: 1px solid #dee2e6; padding: 8px 12px; vertical-align: top; white-space: normal; line-height: 1.5; }}
</style>
</head>
<body>
<div class="report-header">
    <div class="export-btns">
        <button class="export-btn" onclick="exportConfig()">Export Selected Splits (JSON)</button>
        <button class="export-btn" onclick="exportSelectedHTML()">Export Selected Report (HTML)</button>
    </div>
    <div class="filter-bar">
        <span class="filter-group">
            <span class="filter-label">Type:</span>
            <label><input type="checkbox" class="filter-type" value="continuous" checked>Continuous</label>
            <label><input type="checkbox" class="filter-type" value="categorical" checked>Categorical</label>
        </span>
        <span class="filter-group">
            <span class="filter-label">Min IV total:</span>
            <input type="number" id="filter-iv" min="0" step="0.05" value="0">
        </span>
        <span class="filter-group">
            <span class="filter-label">Cramer's V type:</span>
            <label><input type="checkbox" class="filter-vtype" value="Strong" checked>Strong</label>
            <label><input type="checkbox" class="filter-vtype" value="Good" checked>Good</label>
            <label><input type="checkbox" class="filter-vtype" value="Medium" checked>Medium</label>
            <label><input type="checkbox" class="filter-vtype" value="Weak" checked>Weak</label>
        </span>
        <span class="filter-group">
            <span class="filter-label">Status:</span>
            <label><input type="checkbox" class="filter-status" value="optimal" checked>At least optimal</label>
            <label><input type="checkbox" class="filter-status" value="infeasible" checked>All INFEASIBLE</label>
        </span>
        <span class="filter-group">
            <span class="filter-label">Feature names:</span>
            <input type="text" id="filter-names" placeholder="feature_1, feature_2, ...">
        </span>
        <span id="filter-count">Showing all</span>
    </div>
</div>
<h1 class="title">{escape(str(label_name))}</h1>
<p>Features are processed independently. A feature that raises an exception is skipped.</p>
{summary_html}
{skipped_html}
{''.join(sections)}
<script>
const APP_CONFIG = {json.dumps(CONFIG, ensure_ascii=False)};
{EXPORT_SCRIPT}
function applyFilters() {{
    const types = Array.from(document.querySelectorAll('.filter-type:checked')).map(c => c.value);
    const vtypes = Array.from(document.querySelectorAll('.filter-vtype:checked')).map(c => c.value);
    const statuses = Array.from(document.querySelectorAll('.filter-status:checked')).map(c => c.value);
    const minIv = parseFloat(document.getElementById('filter-iv').value) || 0;
    const rawNames = document.getElementById('filter-names').value;
    const names = rawNames.split(',').map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
    const sections = document.querySelectorAll('section.feature');
    let shown = 0;
    sections.forEach(s => {{
        const nameOk = names.length === 0 || names.includes(s.dataset.feature.toLowerCase());
        let statusOk = true;
        if (statuses.length === 0) {{
            statusOk = false;
        }} else if (!(statuses.includes('optimal') && statuses.includes('infeasible'))) {{
            statusOk = (statuses.includes('optimal') && s.dataset.hasOptimal === 'true') ||
                       (statuses.includes('infeasible') && s.dataset.allInfeasible === 'true');
        }}
        const ok = nameOk && statusOk && types.includes(s.dataset.type) && vtypes.includes(s.dataset.vtype) && parseFloat(s.dataset.iv) >= minIv;
        s.style.display = ok ? '' : 'none';
        if (ok) shown++;
    }});
    document.getElementById('filter-count').textContent = `Showing ${{shown}} / ${{sections.length}}`;
}}
document.querySelectorAll('.filter-type, .filter-vtype, .filter-status').forEach(el => el.addEventListener('change', applyFilters));
document.getElementById('filter-iv').addEventListener('input', applyFilters);
document.getElementById('filter-names').addEventListener('input', applyFilters);
applyFilters();
</script>
</body>
</html>
"""