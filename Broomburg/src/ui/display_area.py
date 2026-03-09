import tkinter as tk
from tkinter import ttk

class DisplayArea(ttk.Notebook):
    def __init__(self, root, **kwargs):
        super().__init__(root, padding="0 0 0 10", **kwargs)
        self.pack(expand=True, fill='both')

        root.bind("<Control-w>", self.close_current)
        root.bind("<Control-Shift-W>", self.close_all)
        for i in range(1, 10):
            root.bind(f"<Control-Key-{i}>", lambda e, idx=i-1: self.select_tab(idx))

        self._drag_data = {"tab": None, "index": None}
        self.bind("<ButtonPress-1>", self._on_tab_press, True)
        self.bind("<ButtonRelease-1>", self._on_tab_release, True)
        self.bind("<B1-Motion>", self._on_tab_motion, True)
        self.bind("<<NotebookTabChanged>>", self._on_tab_changed)

        root.bind("<Control-Shift-Prior>", self.move_tab_left)   # PgUp
        root.bind("<Control-Shift-Next>", self.move_tab_right)   # PgDn

    def close_current(self, event=None):
        current = self.select()
        if current:
            self.forget(current)

    def close_all(self, event=None):
        for tab_id in self.tabs():
            self.forget(tab_id)

    def _on_tab_press(self, event):
        x, y = event.x, event.y
        elem = self.identify(x, y)
        if "label" in elem:
            tab_id = self.index(f"@{x},{y}")
            self._drag_data["tab"] = self.tabs()[tab_id]
            self._drag_data["index"] = tab_id

    def _on_tab_motion(self, event):
        if self._drag_data["tab"] is None:
            return
        try:
            idx = self.index(f"@{event.x},{event.y}")
        except tk.TclError:
            return
        if idx != self._drag_data["index"]:
            tab_id = self._drag_data["tab"]
            self.insert(idx, tab_id)
            self._drag_data["index"] = idx

    def _on_tab_release(self, event):
        self._drag_data = {"tab": None, "index": None}

    def select_tab(self, idx):
        if idx < len(self.tabs()):
            self.select(idx)

    def move_tab_left(self, event=None):
        current = self.select()
        if current:
            idx = self.index(current)
            if idx > 0:
                self.insert(idx-1, current)
                self.select(idx-1)

    def move_tab_right(self, event=None):
        current = self.select()
        if current:
            idx = self.index(current)
            if idx < len(self.tabs()) - 1:
                self.insert(idx+1, current)
                self.select(idx+1)

    def _on_tab_changed(self, event=None):
        current = self.select()
        if current:
            frame = self.nametowidget(current)
            for child in frame.winfo_children():
                try:
                    child.focus_set()
                    break
                except tk.TclError:
                    continue



