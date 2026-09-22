import numpy as np

from html import escape

from core.woe_stats import (add_score_column, chi2_cramers, chi2_cramers_categorical,
                            cramer_v_color, cramer_v_type, format_p_value)


def display_woe_tables(results: dict, create_woe_df_func, formats: dict = None, render: bool = True):
    """Generates and displays side-by-side formatted HTML WOE tables from results.
    """
    from core.woe_stats import create_woe_df_categorical

    if formats is None:
        formats = {
            "prob_n_obs": "{:.2%}",
            "pct_event": "{:.2%}",
            "pct_non_event": "{:.2%}",
            "conversion_rate": "{:.2%}",
            "WOE": "{:.2f}",
            "IV_detail": "{:.2f}",
            "IV_total": "{:.2f}",
            "score": "{:.4f}",
        }

    print("WOE TABLES")
    woe_tables = {}
    html_blocks = []

    for idx, (option_name, result) in enumerate(results.items()):
        splits = result["splits"]
        missing_first = result.get("missing_first", False)
        special = result.get("special")
        if result.get("categorical"):
            woe_df = create_woe_df_categorical(result["x"], result["y"], missing_first=missing_first)
        else:
            woe_df = create_woe_df_func(result["x"], result["y"], splits, missing_first=missing_first, special=special)

        is_optimal = result["model"].status in ("OPTIMAL", "OK")

        stats = None
        if is_optimal:
            if result.get("categorical"):
                stats = chi2_cramers_categorical(result["x"], result["y"])
            else:
                stats = chi2_cramers(result["x"], result["y"], splits, special=special)
            woe_df = add_score_column(woe_df, stats[1])

        woe_tables[option_name] = woe_df

        iv_total = woe_df["IV_total"].iloc[0] if "IV_total" in woe_df.columns else None
        display_df = woe_df.drop(columns=["IV_total"], errors="ignore")

        valid_formats = {k: v for k, v in formats.items() if k in display_df.columns}
        styler = display_df.style.format(valid_formats)

        if idx > 0:
            styler.hide(axis="index")

        styled_html = styler.to_html()

        title = result["option"]

        meta_parts = []
        if iv_total is not None:
            iv_total_text = formats.get("IV_total", "{:.2f}").format(iv_total)
            iv_color = "#16a34a" if iv_total >= 0.2 else "#dc2626"
            meta_parts.append(
                f'<span class="iv-total">'
                f'<span class="iv-label">IV total:</span> '
                f'<b class="iv-value" style="color:{iv_color}">{iv_total_text}</b>'
                f'</span>'
            )
        if stats is not None:
            p_color = "#16a34a" if stats[0] <= 0.05 else "#dc2626"
            meta_parts.append(
                f'<span class="p-value" style="color:{p_color}">'
                f'<span class="stat-label">p-value:</span> '
                f'<b>{format_p_value(stats[0])}</b>'
                f'</span>'
            )
        meta_html = " ".join(meta_parts)

        stats_html = ""
        if stats is not None:
            v_c = stats[1]
            v_color = cramer_v_color(v_c)
            stats_html = (
                f'<div style="margin-bottom: 2px; color:{v_color}; font-size: 0.9em;">'
                f'<span class="stat-label">Cramer&apos;s V:</span> <b>{v_c:.4f}</b>'
                f' &nbsp;-&nbsp; '
                f'<span class="stat-label">Type:</span> <b>{cramer_v_type(v_c)}</b>'
                f'</div>'
            )

        block = f"""
        <div class="woe-block" data-part="{escape(result.get('part', ''))}" data-option="{escape(title)}">
            <h4 style="margin-bottom: 4px;">{title}</h4>
            {meta_html}
            {stats_html}
            {styled_html}
        </div>
        """
        html_blocks.append(block)

    parts = {}
    for option_name, result in results.items():
        parts.setdefault(result.get("part", ""), []).append(option_name)

    part_rows = []
    for part, option_names in parts.items():
        blocks = [html_blocks[list(results).index(n)] for n in option_names]
        part_rows.append(
            f"<div class=\"woe-row\" data-part=\"{escape(part)}\">"
            f"<div class=\"woe-row-label\">{part}</div>"
            f"<div class=\"woe-row-columns\">"
            f"{''.join(blocks)}"
            f"</div>"
            f"</div>"
        )

    flex_container = f"""
    <div class="woe-table-container">
        {''.join(part_rows)}
    </div>
    """

    if render:
        pass
        # display(HTML(flex_container))

    return woe_tables, flex_container