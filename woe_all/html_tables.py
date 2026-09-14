def display_woe_tables(results: dict, create_woe_df_func, formats: dict = None, render: bool = True):
    """Generates and displays side-by-side formatted HTML WOE tables from results.
    """
    from core.woe_stats import create_woe_df_categorical

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
        missing_first = result.get("missing_first", False)
        specialvalue = result.get("specialvalue")
        if result.get("categorical"):
            woe_df = create_woe_df_categorical(result["x"], result["y"], missing_first=missing_first)
        else:
            woe_df = create_woe_df_func(result["x"], result["y"], splits, missing_first=missing_first, specialvalue=specialvalue)
        woe_tables[option_name] = woe_df

        iv_total = woe_df["IV_total"].iloc[0] if "IV_total" in woe_df.columns else None
        display_df = woe_df.drop(columns=["IV_total"], errors="ignore")

        valid_formats = {k: v for k, v in formats.items() if k in display_df.columns}
        styler = display_df.style.format(valid_formats)

        if idx > 0:
            styler.hide(axis="index")

        styled_html = styler.to_html()

        if iv_total is not None:
            iv_total_text = formats.get("IV_total", "{:.2f}").format(iv_total)
            title = f"{result['option']} <span class=\"iv-total\">IV total: {iv_total_text}</span>"
        else:
            title = result["option"]

        block = f"""
        <div class="woe-block">
            <h4 style="margin-bottom: 8px;">{title}</h4>
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
            f"<div class=\"woe-row\">"
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