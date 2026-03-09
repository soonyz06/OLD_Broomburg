import tkinter as tk
from tkinter import ttk
import tkinter.font as tkFont

class Table(ttk.Frame):
    def __init__(self, parent, df, df_summary=None, title="Data Viewer",
                 height=400, width=600, row_font_size=16,
                 on_row_select=None, to_pct=None, heading_bg="#e0e0e0", row_bg="#ffffff"):
        super().__init__(parent, width=width, height=height)
        self.pack(fill="both", expand=True)

        self.df = df
        self.df_summary = df_summary if df_summary is not None else df
        self.on_row_select = on_row_select
        self.to_pct = to_pct if to_pct is not None else []
        self.sort_states = {col: "original" for col in self.df_summary.columns}

        # Title
        label = ttk.Label(self, text=title, font=("TkDefaultFont", row_font_size, "bold"))
        label.pack(pady=(4, 2))

        # Scrollbars
        vsb = ttk.Scrollbar(self, orient="vertical")
        hsb = ttk.Scrollbar(self, orient="horizontal")
        vsb.pack(side="right", fill="y")
        hsb.pack(side="bottom", fill="x")

        # Fonts and styles
        row_font = tkFont.Font(family="TkDefaultFont", size=row_font_size)
        style = ttk.Style()
        table_style = "Custom.Treeview"
        style.configure(table_style, font=row_font, rowheight=int(row_font_size * 2))
        style.configure("Custom.Treeview.Heading",
                        font=("TkDefaultFont", row_font_size, "bold"),
                        background=heading_bg)
        style.map(table_style, background=[("selected", "#cce5ff")])

        # Treeview
        self.tree = ttk.Treeview(
            self,
            columns=list(self.df_summary.columns),
            show="headings",
            yscrollcommand=vsb.set,
            xscrollcommand=hsb.set,
            style=table_style
        )

        # Headings
        for col in self.df_summary.columns:
            self.tree.heading(col, text=col.title(), anchor="w",
                              command=lambda c=col: self.sort_column(c))
            self.tree.column(col, anchor="w")

        # Insert rows
        for idx, row in enumerate(self.df_summary.iter_rows(named=True)):
            values = []
            for col in self.df_summary.columns:
                val = row[col]
                if col in self.to_pct:
                    val = f"{float(val):.2f}%"
                values.append(val)
            self.tree.insert("", "end", iid=str(idx), values=values)

        # Auto column sizing
        for col in self.df_summary.columns:
            max_w = row_font.measure(col)
            for item in self.tree.get_children():
                w = row_font.measure(self.tree.set(item, col))
                if w > max_w:
                    max_w = w
            self.tree.column(col, width=int(max_w * 1.1))

        self.tree.pack(fill="both", expand=True)
        vsb.config(command=self.tree.yview)
        hsb.config(command=self.tree.xview)

        # Save original order
        self.original_rows = [(self.tree.item(item)["values"], item) for item in self.tree.get_children()]

        # Row select binding
        if self.on_row_select is not None:
            self.tree.bind("<Return>", self._on_row_select)

    def parse_val(self, iid, col):
        val = self.tree.set(iid, col)
        if col in self.to_pct:
            v = val.strip().replace(",", "")
            if v.endswith("%"):
                v = v[:-1]
            return float(v)
        return val

    def sort_column(self, col):
        state = self.sort_states[col]
        items = list(self.tree.get_children())

        if state == "original":
            items.sort(key=lambda iid: self.parse_val(iid, col))
            self.sort_states[col] = "desc"
        elif state == "desc":
            items = [iid for _, iid in self.original_rows]
            self.sort_states[col] = "asc"
        elif state == "asc":
            items.sort(key=lambda iid: self.parse_val(iid, col), reverse=True)
            self.sort_states[col] = "original"

        for index, iid in enumerate(items):
            self.tree.move(iid, "", index)

    def _on_row_select(self, event):
        selected = self.tree.selection()
        if not selected:
            return
        row_index = int(selected[0])
        self.on_row_select(self.df, row_index)
