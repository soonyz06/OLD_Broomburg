import tkinter as tk
from tkinter import ttk, font
import re
import polars as pl
from src.utils.history_manager import HistoryManager
from src.utils.math import GrowingAnnuity, Perpetuity, FutureValue, PresentValue, PctChange


class FunctionWrapper:
    def __init__(self, func, signature):
        self.func = func
        self.signature = f"func: {signature}"

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __repr__(self):
        return self.signature

class Shell(tk.Text):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            wrap="word",
            font=("Consolas", 12),
            bg="white",
            fg="black",
            insertbackground="black",
            **kwargs
        )
        self.pack(expand=True, fill="both")

        def _help():
            self.insert("end", "\nOperations: +, -, *, /, **, =, ...")
            self.insert("end", "\nPress Tab for autocomplete.")
            self.mark_set("insert", "end")
            self.focus_set()

        def _clear():
            self.delete("1.0", "end")

        def _last_value(df: pl.DataFrame, v: str):
            return df.select(pl.col(v)).tail(1).item()

        self.env = {
            "help": FunctionWrapper(_help, "help()"),
            "clear": FunctionWrapper(_clear, "clear()"),
            "latest": FunctionWrapper(_last_value, "latest(df, col_name) -> val"),
            "GA": FunctionWrapper(GrowingAnnuity, "GA(E_yr, r, g, duration, yr_start=1) -> npv, cfs[]"),
            "TV": FunctionWrapper(Perpetuity, "TV(E_yr, r, g, yr_start=1) -> npv"),
            "FV": FunctionWrapper(FutureValue, "FV(E_0, g, duration) -> fv"),
            "PV": FunctionWrapper(PresentValue, "PV(FV, r, duration) -> pv"),
            "PCT_CHANGE": FunctionWrapper(PctChange, "PCT_CHANGE(new, old) -> diff")
        }

        self.history = HistoryManager(max_depth=50)
        self.history.do("")

        self._candidates = []
        self._candidate_index = 0
        
        self.bind("<Return>", self.on_enter)
        self.bind("<BackSpace>", self.on_backspace)
        self.bind("<Up>", self.on_up)
        self.bind("<Down>", self.on_down)
        self.bind("<Key>", self.on_key)
        self.bind("<Button-1>", self.on_click)
        self.bind("<Right>", self.on_right)
        self.bind("<Left>", self.on_left)
        self.bind("<Tab>", self.on_tab)
        self.focus_set()

        self.insert("end", "Welcome to the Shell!")
        self.insert("end", "\nType 'help()' or press Tab for more information.")
        self.insert("end", "\nUse the Arrow keys to navigate history.\n")
        self.insert_prompt(initial=True)  

    def insert_prompt(self, initial=False):
        if initial or self.compare("end-1c", "==", "1.0"):
            self.insert("end", ">>>")
        else:
            self.insert("end", "\n>>>")
        self.prompt_index = self.index("end-1c")
        self.see("end")
        self.mark_set("insert", "end")
        self.focus_set()

    def on_key(self, event):
        self._candidates = []
        self._candidate_index = 0
        if self.compare("insert", "<", self.prompt_index):
            self.mark_set("insert", "end")
            return "break"
        return None

    def on_click(self, event):
        index = self.index(f"@{event.x},{event.y}")
        if self.compare(index, "<", self.prompt_index):
            self.mark_set("insert", "end")
            return "break"
        return None

    def on_backspace(self, event):
        if self.compare("insert", "<=", self.prompt_index):
            return "break"
        return None

    def on_enter(self, event):
        line_index = self.index("insert linestart")
        line_text = self.get(line_index, line_index + " lineend")
        command = line_text[3:].lstrip() if line_text.startswith(">>>") else line_text
        if command.strip():
            self.history.do(command)
        try:
            result = eval(command, self.env)
            if result is not None:
                self.insert("end", f"\n{result}")
        except SyntaxError:
            try:
                exec(command, self.env)
            except Exception as e:
                self.insert("end", f"\nError: {e}")
        except Exception as e:
            self.insert("end", f"\nError: {e}")
        self.insert_prompt()
        return "break"

    def on_up(self, event):
        prev_cmd = self.history.undo()
        self.delete(self.prompt_index, "end")
        if prev_cmd is not None:
            self.insert("end", prev_cmd)
        return "break"

    def on_down(self, event):
        next_cmd = self.history.redo()
        self.delete(self.prompt_index, "end")
        if next_cmd is not None:
            self.insert("end", next_cmd)
        return "break"

    def on_right(self, event):
        line_start = self.index("insert linestart")
        line_end   = self.index(line_start + " lineend")

        if self.compare("insert", "==", line_end):
            self.mark_set("insert", f"{line_start}+3c")
            return "break"
        return None

    def on_left(self, event):
        line_start = self.index("insert linestart")
        line_end   = self.index(line_start + " lineend")

        prompt_pos = f"{line_start}+3c"
        if self.compare("insert", "==", prompt_pos):
            self.mark_set("insert", line_end)
            return "break"
        return None

    def get_candidates(self):
        return list(self.env.keys())

    def add_to_env(self, data):
        if isinstance(data, dict):
            self.env.update(data)
        else:
            raise TypeError("add_to_env expects a dict")

    def get_latest_word(self, text_before_cursor: str) -> str:
        if not text_before_cursor.strip():
            return ""
        if text_before_cursor.startswith(">>>"):
            text_before_cursor = text_before_cursor[3:]

        parts = re.split(r"[^A-Za-z0-9_]", text_before_cursor)
        last_token = parts[-1] if parts else ""
        return last_token

    def on_tab(self, event):
        line_index = self.index("insert linestart")
        line_text = self.get(line_index, line_index + " lineend")

        if line_text.startswith(">>>"):
            line_text = line_text[3:]

        cursor_index = self.index("insert")
        text_before_cursor = self.get(line_index, cursor_index)
        if text_before_cursor.startswith(">>>"):
            text_before_cursor = text_before_cursor[3:]

        latest_word = self.get_latest_word(text_before_cursor)
        prefix_lower = latest_word.lower()

        if not self._candidates:
            self._candidates = [c for c in self.env.keys() if c.lower().startswith(prefix_lower)]
            self._candidate_index = -1
            self._initial_input = latest_word

        if self._candidates:
            # Find the start index of the latest word relative to cursor
            word_start = f"{cursor_index} - {len(latest_word)}c"

            if self._candidate_index == -1:
                candidate = self._candidates[0]
                self.delete(word_start, cursor_index)
                self.insert(cursor_index, candidate)
                self._candidate_index = 0
            else:
                self._candidate_index += 1
                if self._candidate_index >= len(self._candidates):
                    self.delete(word_start, cursor_index)
                    self.insert(cursor_index, self._initial_input)
                    self._candidate_index = -1
                else:
                    candidate = self._candidates[self._candidate_index]
                    self.delete(word_start, cursor_index)
                    self.insert(cursor_index, candidate)
        return "break"
