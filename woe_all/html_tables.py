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
        woe_df = create_woe_df_func(result["x"], result["y"], splits)
        woe_tables[option_name] = woe_df

        valid_formats = {k: v for k, v in formats.items() if k in woe_df.columns}
        styler = woe_df.style.format(valid_formats)

        if idx > 0:
            styler.hide(axis="index")

        styled_html = styler.to_html()

        block = f"""
        <div style="flex: 1; min-width: 1;">
            <h4 style="margin-bottom: 8px;">{option_name}</h4>
            {styled_html}
        </div>
        """
        html_blocks.append(block)

    flex_container = f"""
    <div style="display: flex; flex-direction: row; gap: 15px; width: 100%; overflow-x: auto;">
        {''.join(html_blocks)}
    </div>
    """

    if render:
        pass
        # display(HTML(flex_container))

    return woe_tables, flex_container