from yahooquery import Ticker
import tkinter as tk
from tkinter import ttk, font
import re
from ttkthemes import ThemedTk
import webbrowser
from src.config.config import registry
                
def fmt(x, mode=",", mult=1):
    if x is None:
        return ""
    if isinstance(x, (int, float)):
        x = x*mult
        if mode==",":
            return f"{x:,}"
        elif mode=="%":
            return f"{(x*100):.2f}%"
        elif mode=="m":
            return f"{x:.1f}"
        else:
            return f"{x}{mode}"
    return str(x)

def reset_parent(parent):
    if parent is None:
        return
    for widget in parent.winfo_children():
        widget.destroy()
        
def build_root():
    print("[INFO]Loading interface...")
    root = ThemedTk(theme="classic")
    root.title("Broomburg")
    root.geometry("1000x650")
    root.state('zoomed')
    root.lift()               
    root.attributes('-topmost', True)  
    root.after(0, lambda: root.attributes('-topmost', False))  
    style = ttk.Style(root)
    root.configure(bg=style.lookup("TFrame", "background"))
    apply_entry_styles(root)
    default_font = font.nametofont("TkDefaultFont")
    default_font.configure(family="Inter", size=11)
    root.focus_force()
    input_frame = ttk.Frame(root, padding="10 10 10 0")
    input_frame.pack(fill="x")
    return root, input_frame

def add_dict_frame(parent, title, data):
    frame = ttk.LabelFrame(parent, text=title, padding=10)
    for i, (k, v) in enumerate(data.items()):
        if isinstance(v, float):
            v = f"{v:,.2f}"
        elif isinstance(v, int):
            v = f"{v:,.0f}"
        ttk.Label(frame, text=f"{k}:").grid(row=i, column=0, sticky="nw", padx=5)
        ttk.Label(frame, text=v).grid(row=i, column=1, sticky="ne", padx=5)
    return frame.pack(fill="x", anchor="n")
    
def create_Label(parent, func, texts=("Left", "Top Right", "Bottom Right"), size=48):
    if parent is None:
        return
    big_font = font.Font(family="Helvetica", size=size, weight="bold")
    small_font = font.Font(family="Helvetica", size=int(size/2.5), weight="bold")
    container = tk.Frame(parent, background="black")
    container.place(relx=0.5, rely=0.5, anchor="center")
    container.grid_columnconfigure(0, weight=2)  
    container.grid_columnconfigure(1, weight=3)  
    container.grid_rowconfigure(0, weight=1)
    container.grid_rowconfigure(1, weight=1)

    left_label = tk.Label(container, text=texts[0], font=big_font, anchor="e", fg="white", bg="black")
    left_label.grid(row=0, column=0, rowspan=2, sticky="nsew", padx=10)
    top_label = tk.Label(container, text=texts[1], font=small_font, anchor="sw", fg="white", bg="black")
    top_label.grid(row=0, column=1, sticky="nsew", padx=10)
    bottom_label = tk.Label(container, text=texts[2], font=small_font, anchor="nw", fg="white", bg="black")
    bottom_label.grid(row=1, column=1, sticky="nsew", padx=10)

    value = float(texts[2].replace("%", ""))  # parse as float
    if value < 0:
        left_label.configure(fg="red")
        top_label.configure(fg="red")
        bottom_label.configure(fg="red")
    elif value > 0:
        left_label.configure(fg="green")
        top_label.configure(fg="green")
        bottom_label.configure(fg="green")
    return left_label, top_label, bottom_label

def apply_entry_styles(root):
    style = ttk.Style(root)
    base_bg = style.lookup("TEntry", "fieldbackground") or "SystemWindow"
    base_fg = style.lookup("TEntry", "foreground") or "black"
    
    processing_fg = "blue"
    style.configure("Processing.TEntry", foreground=processing_fg, fieldbackground=base_bg)

def apply_theme(root, theme):
    root.set_theme(theme)
    style = ttk.Style(root)
    theme_bg = style.lookup("TFrame", "background") or "SystemButtonFace"
    root.configure(bg=theme_bg)
    apply_entry_styles(root)

def get_child(parent, child_type):
    for child in parent.winfo_children():
        if isinstance(child, child_type):
            return child
    return None

def bind_descendants(widget, sequence, func):
    focusable = (
       ttk.Treeview, ttk.Notebook, ttk.Frame
    )
    if isinstance(widget, focusable):
        widget.bind(sequence, func)

    for child in widget.winfo_children():
        bind_descendants(child, sequence, func)

def build_table(parent, df, fontsize=16):
    style = ttk.Style()
    style.configure("Treeview", rowheight=fontsize + 8)
    
    tree = ttk.Treeview(parent, columns=df.columns, show="headings")
    vsb = ttk.Scrollbar(parent, orient="vertical", command=tree.yview)
    hsb = ttk.Scrollbar(parent, orient="horizontal", command=tree.xview)
    tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)
    tree.grid(row=0, column=0, sticky="nsew")
    vsb.grid(row=0, column=1, sticky="ns")
    hsb.grid(row=1, column=0, sticky="ew")
    parent.grid_rowconfigure(0, weight=1)
    parent.grid_columnconfigure(0, weight=1)
    default_font = font.Font(family="Helvetica Neue", size=fontsize)
    tree.tag_configure("default_row", font=default_font)
    return tree, vsb, hsb, default_font

def autosize_table(tree, df, fnt):
    text_height = fnt.metrics("ascent") + fnt.metrics("descent")
    row_height = text_height + 4
    style = ttk.Style(tree)
    style.configure("Treeview", rowheight=row_height)
    
    for col in df.columns:
        header_text = col
        max_width = fnt.measure(header_text)
        for val in df[col]:
            cell_text = str(val)
            cell_width = fnt.measure(cell_text)
            if cell_width > max_width:
                max_width = cell_width
        tree.column(col, width=max_width + 10)

def add_display_table(parent, df, title="Data Viewer", to_pct=None, on_select=None): ##auto int vs float wihtout to_pct *check all
    tree, vsb, hsb, default_font = build_table(parent, df, 16)
    if to_pct is not None:
        to_pct = [x.lower() for x in to_pct]
    to_pct = set(to_pct or [])
    # add stable index column once
    df = df.with_row_count("orig_index")
    original_df = df.clone()
    sort_states = {col: 2 for col in df.columns}

    item_to_orig_index = {}

    def populate(dataframe):
        tree.delete(*tree.get_children())
        item_to_orig_index.clear()
        for row in dataframe.iter_rows(named=True):
            clean_row = []
            row_tags = []
            for col_name, x in row.items():
                if col_name == "orig_index":
                    continue  # don't display the helper column
                if col_name.lower() in to_pct and isinstance(x, (int, float)):
                    formatted = fmt(x/100, "%")
                else:
                    formatted = fmt(x, ",")
                clean_row.append(formatted)

            iid = tree.insert("", "end", values=clean_row,
                              tags=tuple(row_tags) or ("default_row",))
            item_to_orig_index[iid] = row["orig_index"]
        autosize_table(tree, dataframe.drop("orig_index"), default_font)

    tree.tag_configure("negative", font=default_font, foreground="red")
    tree.tag_configure("positive", font=default_font, foreground="green")
    tree.tag_configure("default_row", font=default_font)
    populate(df)

    def cycle_sort(col):
        sort_states[col] = (sort_states[col] + 1) % 3
        state = sort_states[col]
        if state == 0:
            new_df = df.sort(col, descending=True)
        elif state == 1:
            new_df = df.sort(col, descending=False)
        else:
            new_df = original_df.clone()
        populate(new_df)

    for col in df.columns:
        if col == "orig_index":
            continue
        tree.heading(col, text=col,
                     command=lambda c=col: cycle_sort(c))
        dtype = df[col].dtype
        tree.column(col, anchor="e" if dtype.is_numeric() else "w", stretch=False)

    def on_enter(event):
        selected = tree.selection()
        if selected and on_select is not None:
            iid = selected[0]
            orig_index = item_to_orig_index.get(iid)
            if orig_index is not None:
                on_select(orig_index)

    tree.bind("<Return>", on_enter)
    return tree


def handle_link(df, index):
    link = df["link"][index]
    if link is None:
        return
    if link.startswith(("https://")):
        webbrowser.open_new(link)
    else:
        print(f"Blocked unsafe link: {link}")


###abstract out and in MATCH, di and redo both tables
