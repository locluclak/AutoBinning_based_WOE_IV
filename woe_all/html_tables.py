def display_woe_tables(results: dict, create_woe_df_func, formats: dict = None, render: bool = True):
    """Generates and displays side-by-side formatted HTML WOE tables from results.
    """
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
        woe_df = create_woe_df_func(result["x"], result["y"], splits, missing_first=missing_first)
        woe_tables[option_name] = woe_df

        valid_formats = {k: v for k, v in formats.items() if k in woe_df.columns}
        styler = woe_df.style.format(valid_formats)

        if idx > 0:
            styler.hide(axis="index")

        styled_html = styler.to_html()

        block = f"""
        <div class="woe-block">
            <h4 style="margin-bottom: 8px;">{option_name}</h4>
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