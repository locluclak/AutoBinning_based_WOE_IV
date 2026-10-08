"""Generate a final HTML report from a splits JSON and its per-bin CSV.

Usage:
    python export_final_bininfo.py [splits.json] [bin_info.csv] [output.html] [--label NAME]

The report mirrors the single-JSON (version 1) layout of reconstruct_report.py,
but it is built entirely from the exported splits JSON and the bin-info CSV
produced by generate_bins_csv.py, without loading the raw dataset.

Differences from the original report:
    - no WOE trend / bin count comparison plot
    - no binning-option tag above the WOE table rows
    - the WOE table keeps only Bin, Number of customer and score
    - the summary shows only the Total features metric
    - --label NAME overrides the label name shown in the report title
"""
import json
import sys
from html import escape
from pathlib import Path

import pandas as pd

from core.woe_stats import SPECIAL_OPTION, cramer_v_color, format_p_value
from reconstruct.reconstruct import PAGE_CSS, build_search_bar, build_skipped_html


def build_summary_html(meta):
    if not meta:
        return ""
    total = len(meta)
    continuous = [m for m in meta if m["type"] == "continuous"]
    categorical = [m for m in meta if m["type"] == "categorical"]

    def cells(items):
        if not items:
            return ""
        ordered = sorted(items, key=lambda m: str(m["feature"]).lower())
        return "<br>".join(escape(str(m["feature"])) + "," for m in ordered)

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
    vtype_groups = [[m for m in meta if m["v_type"] == vt] for vt in vt_labels]
    table_vtypes = summary_table("Cramer's V type", list(vt_labels), vtype_groups)

    iv_headers = ["IV total \u2264 0.2", "0.2 < IV total \u2264 0.5", "0.5 < IV total \u2264 1", "IV total > 1"]
    iv_groups = [
        [m for m in meta if m["iv_total"] <= 0.2],
        [m for m in meta if 0.2 < m["iv_total"] <= 0.5],
        [m for m in meta if 0.5 < m["iv_total"] <= 1],
        [m for m in meta if m["iv_total"] > 1],
    ]
    table_iv = summary_table("IV total", iv_headers, iv_groups)

    important = [m for m in meta if m.get("important")]
    table_important = summary_table("Important features", ["Important features"], [important])

    metrics = (
        f'<div class="metric"><div class="value">{total}</div>'
        f'<div class="label">Total features</div></div>'
    )

    return f"""
    <section class="summary">
        <h2>Summary</h2>
        <div class="summary-metrics">{metrics}</div>
        <div class="summary-tables">{table_types}{table_vtypes}{table_iv}{table_important}</div>
    </section>
    """


def parse_args(argv):
    args = list(argv)
    output_file = None
    label_name = None
    positional = []
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == "--label":
            if i + 1 < len(args):
                label_name = args[i + 1]
                i += 2
                continue
        elif arg.startswith("--label="):
            label_name = arg.split("=", 1)[1]
            i += 1
            continue
        positional.append(arg)
        i += 1
    args = positional
    if args and args[-1].lower().endswith(".html"):
        output_file = args.pop()
    splits_json = args[0] if args else "selected_feature_splits.json"
    bin_info_csv = args[1] if len(args) > 1 else "bins_report.csv"
    return splits_json, bin_info_csv, output_file, label_name


def woe_table_html(rows):
    display_df = (rows[["Bin", "Number customer", "score"]]
                  .rename(columns={"Number customer": "Number of customer"}).copy())
    return display_df.style.format({"score": "{:.4f}"}).to_html()


def feature_iv_total(rows):
    if "IV_total of feature" in rows.columns:
        return float(rows["IV_total of feature"].iloc[0])
    for col in ("IV_bin", "IV"):
        if col in rows.columns:
            return float(pd.to_numeric(rows[col], errors="coerce").sum())
    return 0.0


def build_feature_card(feature, cfg, rows, important=False):
    ftype = str(rows["Feature type"].iloc[0])
    categorical = ftype == "categorical" or str(cfg.get("type", "")).lower() == "categorical"
    part = cfg.get("part", "consider SPECIAL")
    option = cfg.get("option", "")
    special_case = option == SPECIAL_OPTION or part == SPECIAL_OPTION

    iv_total = feature_iv_total(rows)
    p_value = float(rows["P_value"].iloc[0])
    v_c = float(rows["CramerV"].iloc[0])
    v_type = str(rows["CramerV_remark"].iloc[0])

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
        f'<span class="stat-label">Type:</span> <b>{escape(v_type)}</b>'
        f'</div>'
    )

    if special_case:
        option_html = f'<p class="option">Selected option: {escape(SPECIAL_OPTION)}</p>'
        splits_html = ""
    elif categorical:
        option_html = f'<p class="option">Selected version: {escape(part)}</p>'
        splits = [str(c) for c in (cfg.get("splits") or [])]
        if not splits:
            splits = [str(b) for b in rows["Bin"] if str(b) != "Missing"]
        splits_html = f'<p class="splits">Groups: {escape("[" + ", ".join(splits) + "]")}</p>'
    else:
        option_html = f'<p class="option">Selected option: {escape(option)}</p>'
        splits = [float(s) for s in (cfg.get("splits") or [])]
        formatted_splits = "[" + ", ".join(f"{v:,}" for v in splits) + "]"
        splits_html = f'<p class="splits">Splits: {escape(formatted_splits)}</p>'

    woe_table = woe_table_html(rows)

    html = f"""
    <div class="feature-card" data-feature="{escape(str(feature))}" data-type="{ftype}">
        <h2>{escape(str(feature))}</h2>
        {meta_html}
        {stats_html}
        {option_html}
        {splits_html}
        <h3>WOE Tables</h3>
        <div class="woe-table-container">
            <div class="woe-row">
                <div class="woe-row-columns"><div class="woe-block">{woe_table}</div></div>
            </div>
        </div>
    </div>
    """

    meta = {
        "feature": feature,
        "type": ftype,
        "iv_total": iv_total,
        "v_type": v_type,
        "has_optimal": True,
        "all_infeasible": False,
        "important": important,
    }
    return html, meta


def build_single_body(version):
    sections = []
    for feature, card in version["results"].items():
        sections.append(
            f'<section class="feature" data-feature="{escape(str(feature))}">'
            f'{card}<hr class="feature-separator"></section>'
        )
    return f"""
{build_summary_html(version["meta"])}
{''.join(sections)}
{build_skipped_html(version["skipped"])}
"""


def build_report(splits_json, bin_info_csv, label_name=None):
    with open(splits_json, "r", encoding="utf-8") as f:
        payload = json.load(f)

    config = payload.get("config", {})
    label_name = label_name or config.get("label_name", "LABEL")
    important_set = {str(f).lower() for f in (config.get("important_features") or [])}

    bin_info = pd.read_csv(bin_info_csv, encoding="utf-8-sig")
    bin_info["Feature name"] = bin_info["Feature name"].astype(str)
    bin_info["score"] = pd.to_numeric(bin_info["score"], errors="coerce")
    bin_info["Number customer"] = (
        pd.to_numeric(bin_info["Number customer"], errors="coerce").fillna(0).astype(int)
    )

    results = {}
    meta = []
    skipped = []
    for feature, fcfg in payload.get("features", {}).items():
        if feature == label_name or str(feature).startswith(str(label_name)):
            continue
        rows = bin_info[bin_info["Feature name"] == str(feature)]
        if rows.empty:
            skipped.append((feature, "Missing in bin_info.csv"))
            continue
        try:
            card, card_meta = build_feature_card(
                feature, fcfg, rows, important=str(feature).lower() in important_set)
            results[feature] = card
            meta.append(card_meta)
            print(f"OK: {feature}")
        except Exception as exc:
            skipped.append((feature, exc))
            print(f"SKIP: {feature} -> {type(exc).__name__}: {exc}")

    version_data = [{
        "label": Path(splits_json).stem,
        "results": results,
        "meta": meta,
        "skipped": skipped,
    }]
    body = build_single_body(version_data[0])

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>WOE Report - Reconstructed</title>
<style>{PAGE_CSS}</style>
</head>
<body>
<div class="report-header">
    <h1 class="title">{escape(str(label_name))}</h1>
    {build_search_bar(version_data)}
</div>
<p>Reconstructed from exported splits. No OptimalBinning used.</p>
{body}
<script>
function getSearchNames(input) {{
    return input.value.split(',').map(s => s.trim().toLowerCase()).filter(s => s.length > 0);
}}

function applySearch() {{
    const inputs = Array.from(document.querySelectorAll('.search-input'));
    const targets = document.querySelectorAll('.feature, .compare-row');
    let shown = 0;
    targets.forEach(t => {{
        const feature = (t.dataset.feature || '').toLowerCase();
        let ok = inputs.length === 0;
        if (inputs.length > 0) {{
            ok = inputs.some(input => {{
                const names = getSearchNames(input);
                return names.length === 0 || names.some(n => feature.startsWith(n));
            }});
        }}
        t.style.display = ok ? '' : 'none';
        if (ok) shown++;
    }});
    document.getElementById('filter-count').textContent = `Showing ${{shown}} / ${{targets.length}}`;
}}

document.querySelectorAll('.search-input').forEach(input =>
    input.addEventListener('keydown', (event) => {{
        if (event.key === 'Enter') applySearch();
    }}));
applySearch();
</script>
</body>
</html>
"""


def main(argv):
    splits_json, bin_info_csv, output_file, label_name = parse_args(argv)
    output_file = output_file or "final_bininfo_report.html"
    html = build_report(splits_json, bin_info_csv, label_name)
    Path(output_file).write_text(html, encoding="utf-8")
    print(f"HTML report: {output_file}")


if __name__ == "__main__":
    main(sys.argv[1:])