import tkinter as tk
from tkinter import ttk
from src.ui.ui import add_display_table, handle_link

def create_Wti(parent, screeners):
    notebook = ttk.Notebook(parent)
    notebook.pack(fill="both", expand=True)

    for name, df in screeners.items():
        tab = ttk.Frame(notebook)
        notebook.add(tab, text=name.replace("_", " ").title())
        add_display_table(tab, df=df, title="", to_pct = ["pctchange"])
    return notebook

def create_News(parent, df):
    container = ttk.Frame(parent)
    container.pack(expand=True, fill="both")
    add_display_table(container, df=df.select(["date", "symbol", "source", "headline"]), title="News",
                      on_select=lambda idx: handle_link(df, idx))
    return container

def create_Vol(parent, df):
    container = ttk.Frame(parent)
    container.pack(expand=True, fill="both")
    add_display_table(container, df=df, title="", to_pct = df.columns[1:])
    return container    
    
