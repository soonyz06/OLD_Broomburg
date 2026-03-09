import tkinter as tk
import tkinter.font as tkFont
from tkinter import ttk
from src.config.config import cache, registry
from src.utils.history_manager import HistoryManager
from src.utils.cli import parse_command
from src.ui.display_area import DisplayArea
from src.workers.workers import bind_workers
import threading

class CLIEntry(ttk.Entry):
    def __init__(self, root, parent):
        self.cli_var = tk.StringVar(value="")
        super().__init__(parent, textvariable=self.cli_var, width=40)

        self.history = HistoryManager()
        self.pack(side=tk.LEFT, fill="x", expand=True, padx=5)
        self.focus_set()
        self.configure(font=tkFont.Font(size=13))
        
        self.bind("<Return>", lambda event: self.CLI())
        self.bind("<Escape>", lambda event: self.reset_entry())
        self.bind("<Up>", lambda event: self.reset_entry(self.history.undo()))
        self.bind("<Down>", lambda event: self.reset_entry(self.history.redo()))
        parent.winfo_toplevel().bind_all("<Escape>", lambda event: self.focus_set())
        self.cli_var.trace_add("write", self._force_upper)

        self.root = root
        self.display_area = DisplayArea(parent.winfo_toplevel())
        bind_workers(self.root, self.display_area)

    def _force_upper(self, *args):
        val = self.cli_var.get()
        up = val.upper()
        if val != up:
            self.cli_var.set(up)

    def reset_entry(self, cmd=None):
        self.config(style="TEntry")
        self.delete(0, tk.END)
        if cmd:
            self.cli_var.set(cmd + " ")
        else:
            self.cli_var.set("")
        self.focus_set()
        self.icursor(tk.END)

    def process_entry(self, cmd=None):
        if cmd:
            self.config(style="Processing.TEntry")

    def CLI(self):
        cmd = self.cli_var.get().upper().strip()
        params, func, settings = parse_command(cmd)
        if not func: return
        
        self.history.do(cmd)
        self.process_entry(cmd)
        cmds = [c for c in cmd.split("<GO>")]
        
        def run_in_background(cmds):
            for cmd in cmds:
                params, func, _ = parse_command(cmd)
                worker_fn = registry.get(func)
                if worker_fn:
                    #try:
                    new_tab = ttk.Frame(self.display_area, padding="0 0 0 10")
                    result = worker_fn(cmd, new_tab)
                    #except Exception as e:
                       # print(f"[WARNING]{e}")
            self.winfo_toplevel().after(0, lambda: self.reset_entry())    
        threading.Thread(target=lambda: run_in_background(cmds), daemon=True).start()

