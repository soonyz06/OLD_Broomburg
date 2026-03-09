from pathlib import Path
import polars as pl
import random
import threading
from concurrent.futures import ThreadPoolExecutor
from src.config.config import registry, SIZE
from src.utils.financial import preload_data, get_financial_data, get_relative_valuation_data, compute_fa_exponent, get_desc, get_tables, save_FA
from src.ui.ui import create_Label, apply_entry_styles, apply_theme, reset_parent, bind_descendants
from src.ui.fa_ui import create_FA
from src.ui.live_ui import create_News, create_Wti, create_Vol
from src.utils.cli import parse_command
from src.utils.io import fetch_price_data, fetch_screeners, fetch_etfs, fetch_vol, fetch_batch, fetch_background
from src.utils.file import open_file, filter_tickers
from src.utils.news import get_news_tape

def bind_workers(root, display_area): 
    @registry.register(["DES", "FA", "IS", "BS", "CF", "RV", "CMD", "SAVE"]) 
    def worker_FA(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        func_map = {"IS": ["income_statement"], "BS": ["balance_sheet"], "CF": ["cash_flow"]}
        domains = func_map.get(func, ["income_statement", "balance_sheet", "cash_flow"])
        summary, background, rv = None, None, None
        
        fa = get_financial_data(params, domains, **settings.get("f", {})) 
        exp = compute_fa_exponent(fa)

        if func == "DES":
            summary, background = get_desc(params, exp)
        print(summary, background)
        if domains == ["income_statement", "balance_sheet", "cash_flow"]:
            background = fetch_batch(params, Path.cwd() / "data" / "staging", "_back.parquet", fetch_background, SAVE=True)
            background = {row["symbol"]: {c: row[c] for c in background.columns if c != "symbol"} for row in background.iter_rows(named=True)}  
        rv = get_relative_valuation_data(fa, background)

        dfs = get_tables(domains, fa, rv, exp)
        UNITS = {12: "T", 9: "B", 6: "M", 3: "K", 0: ""}
        unit_label = UNITS[exp]
            
        if func == "SAVE":
            save_FA(summary, background, dfs, unit_label)
            return
        root.after(0, lambda: (
            create_FA(TAB, func, summary, background, dfs, unit_label), 
            display_area.add(TAB, text=func),
            display_area.select(TAB)
        ))

    @registry.register(["OPEN"]) 
    def worker_open(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        base_path = Path.cwd() / "data" / "output" / "fa_models"
        base_path.mkdir(exist_ok=True)

        for file_name in params:
            file_path = base_path / f"{file_name}_FA.xlsx"
            if file_path.exists():
                open_file(file_path)
                return

            worker_fn = registry.get("SAVE")
            if worker_fn:
                base_cmd = " ".join(params)
                result = worker_fn(base_cmd+" SAVE")
            if file_path.exists():
                open_file(file_path)
            else:
                print(f"[OPEN] still missing {file_path}, not retrying")
                    
    @registry.register(["VOL"])
    def worker_vol(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        df = fetch_vol(params, period="4y", interval="1d")
        
        root.after(0, lambda: (
            create_Vol(TAB, df),
            display_area.add(TAB, text=func),
            display_area.select(TAB)
        ))

    @registry.register(["N"]) #dedicated news wires instead of google news
    def worker_news(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        df = get_news_tape(params)

        root.after(0, lambda: (
            create_News(TAB, df),
            display_area.add(TAB, text=func),
            display_area.select(TAB)
        ))

    @registry.register(["WTI"])
    def worker_wti(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        screeners = fetch_screeners()
        etfs = fetch_etfs()
        wti_dict = {**screeners, **etfs}
        root.after(0, lambda: (
            create_Wti(TAB, wti_dict),
            display_area.add(TAB, text=func),
            display_area.select(TAB)
        ))
        

    """  
    @registry.register(["LIVE"]) #stream prices :)
    def worker_price(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        live = fetch_price_data([params[0]])
        price_data = live[params[0]]
        root.after(0, lambda: (
            create_Label(TAB, func, (params[0], f"{price_data['Price']:.2f}", f"{price_data['pctChange']:.2f}%")),
            display_area.add(TAB, text=func),
            display_area.select(TAB)
            ))
        
    @registry.register(["THEME"]) 
    def worker_theme(cmd, TAB=None): 
        params, func, settings = parse_command(cmd)
        if params and params[0] == "SYS":
            theme = params[1].lower() if len(params) > 1 else random.choice(root.get_themes())
            root.after(0, lambda: (apply_theme(root, theme)))

    @registry.register(["MATCH"])
    def worker_match(cmd, TAB=None):
        params, func, settings = parse_command(cmd)
        df = filter_tickers(col_filter=params, **settings.get("f", {}))
        if df is None:
            return

        symbols = df["Symbol"].to_list()
        worker_fn = registry.get("DES")
        if not worker_fn or not symbols:
            return
        buffer_size = 10
        state = {"symbols": symbols, "index": 0, "pointer": buffer_size}
        executor = ThreadPoolExecutor(max_workers=4) 

        def run_worker(symbol):
            try:
                worker_fn(symbol.upper() + " DES", TAB)
            except Exception as e:
                print(f"[WARNING] {symbol} failed due to {e}")

            root.after(0, lambda: (
                bind_descendants(TAB, "<Control-Left>", lambda e: change_tab(direction=-1, buffer_size=buffer_size, event=e)),
                bind_descendants(TAB, "<Control-Right>", lambda e: change_tab(direction=1, buffer_size=buffer_size, event=e)),
                TAB.focus_set()
            ))

        def change_tab(direction, buffer_size, event=None):
            n = len(state["symbols"])
            reset_parent(TAB)
        
            if n == 0:
                return
            if n == 1:
                state["index"] = 0
            else:
                if direction < 0:
                    state["index"] = max(state["index"] - 1, 0)
                else:
                    state["index"] = (state["index"] + 1) % n
                    if state["pointer"] != 0:
                        state["pointer"] = (state["index"] + buffer_size) if (state["index"] + buffer_size) < n else 0
                        executor.submit(preload_data, [state["symbols"][state["pointer"]]])
            threading.Thread(target=run_worker, args=(state["symbols"][state["index"]],), daemon=True).start()

        #first instance
        n = len(state["symbols"])
        first_symbol = state["symbols"][state["index"]]
        preload = [state["symbols"][j] for j in range(state["index"], min(state["index"] + buffer_size + 1, n))]
        executor.submit(preload_data, preload)
        threading.Thread(target=run_worker, args=(first_symbol,), daemon=True).start()
        """
        
        
    


        
