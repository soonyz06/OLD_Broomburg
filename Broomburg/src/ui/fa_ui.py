import tkinter as tk
from tkinter import ttk, font
from src.utils.financial import format_subsets_df
from src.ui.shell import Shell
from src.ui.ui import add_dict_frame, build_table
from src.utils.df import get_latest, transpose_df

highlights = ["Revenue", "Gross Profit", "Operating Income", "Pretax Income", "Net Income Common",
              "CFFO", "CFFI", "CFFF", "OCF", "CapEx", "FCF",
              "Total Assets", "Total Liabilities", "Total Equity", "Common Equity", "PPE", "Inventory", "NCAV",
              "Beta", "EVE", "EVB", "EVS", "PE", "PB", "PS", "EPS", "ROTA", "ROIC", "ROCE", "ROE", "D_E",
              "RevGrowth", "GrossMargin", "OpMargin",
              "CFGrowth", "CXChange", "FCFMargin",
              "EquityGrowth", "TAGrowth", "ICGrowth"]

def create_FA(parent, func, summary, background, dfs, unit_label):
    if parent is None:
        return
    if func in ["DES"]:
        create_overview_tab(parent, summary, background, dfs, unit_label)
    if func in ["FA", "IS", "BS", "CF", "RV"]:
        create_financials_tab(parent, dfs, func, unit_label)
    if func in ["CMD"]:
        create_terminal_tab(parent, summary, background, dfs, unit_label)
        
def create_overview_tab(tab_overview, summary, background, dfs, unit_label):
    tabs = ttk.Notebook(tab_overview)
    tabs.pack(expand=True, fill="both", padx=5, pady=5)
    
    for symbol in summary.keys():
        data = dfs[symbol]
        data = {k: data[k] for k in [list(data)[-1], *list(data)[:-1]]}
        container = ttk.Frame(tabs)
        container.pack(expand=True, fill="both")
        add_dict_frame(container, "Summary", summary[symbol])
        add_dict_frame(container, "Background", background[symbol])
        add_display_tables(container, data, subsets=None, highlights=highlights)
        tabs.add(container, text=symbol.upper()+f" ({unit_label})")

def create_financials_tab(tab_dfs, dfs, func, unit_label):
    domains = {"IS": "income_statement", "BS": "balance_sheet", "CF": "cash_flow", "RV": "relative_valuation"}
    subsets = {"Operating Expense": ["SG&A", "R&D", "Others"],
               "Operating Income": ["Other Income", "Net Interest"],
               "Pretax Income": ["After-Tax Adj"],
               "Net Income": ["Other Stockholders"],
               "Current Assets": ["Cash & Short Inv", "Inventory"],
               "Non-Current Assets": ["PPE"],
               "Current Liabilities": ["Short Term Debt"],
               "Non-Current Liabilities": ["Long Term Debt"],
               "Total Equity": ["Common Equity"]}
    
    tabs = ttk.Notebook(tab_dfs)
    tabs.pack(expand=True, fill="both", padx=5, pady=5)
    for key, data in dfs.items():
        if func != "FA":
            domain = domains[func]
            data = {domain: data[domain]}
        df_tabs = add_display_tables(tabs, data, subsets)
        tabs.add(df_tabs, text=key.upper()+f" ({unit_label})")
    return tabs
    
def create_terminal_tab(tab_calc, summary, background, dfs, unit_label):
    tabs = ttk.Notebook(tab_calc)
    tabs.pack(expand=True, fill="both", padx=5, pady=5)

    for key in summary.keys():
        container = ttk.Frame(tabs)
        container.pack(expand=True, fill="both")

        onload_data = onload_calculations(summary[key], background[key], dfs[key])
        cleaned_data = {k.replace(" ", "_"): v for k, v in onload_data.items()}
        shell = Shell(container)
        shell.add_to_env(summary[key])
        shell.add_to_env(dfs[key])
        shell.add_to_env(cleaned_data)
        shell.add_to_env(background[key])
        tabs.add(container, text=key.upper()+f" ({unit_label})")
    tab_calc.columnconfigure(0, weight=1)
            
def onload_calculations(summary, background, dfs):
    onload_data = {}
    onload_data["Price"] = summary['Price']
    onload_data["MC"] = summary['MC']
    IS = get_latest(dfs, "income_statement", highlights)
    BS = get_latest(dfs, "balance_sheet", ["Net Debt"]+highlights)
    CF = get_latest(dfs, "cash_flow", highlights)
    onload_data = {**onload_data, **IS, **BS, **CF}
    return onload_data

def add_display_tables(parent, dfs, subsets=None, highlights=None):
    df_tabs = ttk.Notebook(parent)
    df_tabs.pack(expand=True, fill="both", padx=5, pady=5)
    
    for domain, df in dfs.items():
        f = ttk.Frame(df_tabs)
        df_tabs.add(f, text=domain.title())
        df = format_subsets_df(df, subsets, highlights)
        df = transpose_df(df, "date")
        
        tree, vsb, hsb, default_font = build_table(f, df)

        for i, col in enumerate(df.columns):
            tree.heading(col, text=col.title())
            if i == 0:
                tree.column(col, anchor="w", stretch=False)
            else:
                tree.column(col, anchor="e", stretch=False)

        for row in df.iter_rows():
            tree.insert("", "end", values=row, tags=("default_row",))
    return df_tabs


