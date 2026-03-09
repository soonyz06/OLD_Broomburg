import os
import subprocess
import platform
import pandas as pd
import polars as pl
from pathlib import Path

def open_file(file_path):
    system = platform.system()
    if system == "Windows":
        os.startfile(file_path)
    elif system == "Darwin":  # macOS
        subprocess.call(["open", file_path])
    else:  # Linux/Unix
        subprocess.call(["xdg-open", file_path])
        
def write_to_xlsx(file_path, sheets, unit_label, percents=[]):
    print(sheets)
    with pd.ExcelWriter(file_path, engine="xlsxwriter") as writer:
        startrow, startcol = 1, 8
        for name, df in sheets.items():
            df.to_excel(writer, sheet_name=f"{name}({unit_label})", index=False,
                        startrow=startrow + 1, startcol=startcol, header=False)
            worksheet = writer.sheets[name]
            workbook = writer.book
            num_fmt     = workbook.add_format({"num_format": "#,##0", "align": "right"}) 
            percent_fmt = workbook.add_format({"num_format": "0.00%", "align": "right"})
            text_fmt    = workbook.add_format({"align": "left"})
            for idx, col in enumerate(df.columns):
                col_letter = startcol + idx
                max_len = max(df[col].astype(str).map(len).max(), len(str(col))) + 2
                if pd.api.types.is_numeric_dtype(df[col]):
                    fmt = percent_fmt if name in percents else num_fmt
                else:
                    fmt = text_fmt
                worksheet.write(startrow, col_letter, col, fmt)
                worksheet.set_column(col_letter, col_letter, max_len, fmt)
            startrow, startcol = 1, 1
    print(f"[INFO]Saved {file_path}")

def fetch_symbols():
    base_path = Path.cwd() / "data" / "fa_config"
    base_path.mkdir(exist_ok=True)
    file_name = "symbols.parquet"
    file_path = base_path / file_name
    if not file_path.exists():      
        lf = pl.scan_csv(base_path / "nasdaq_screener.csv") #load from nasdaq
        df = (lf.filter(pl.col("Symbol").is_not_null())
               .filter(pl.col("Market Cap").is_not_null())
               .filter(pl.col("Market Cap") > 0)
               .filter(pl.col("Symbol").str.contains(r"^[A-Za-z]+$"))
               .sort(by=["Market Cap"], descending=[True])
               .select(["Symbol"])
               .collect())
        df.write_parquet(file_path)
    else:
        df = pl.read_parquet(file_path)
    return df

def delete_symbols(to_remove=None):
    if not to_remove:
        return
    print(f"[INFO]Deleting {len(to_remove)} symbols")
    base_path = Path.cwd() / "data" / "fa_config"
    base_path.mkdir(exist_ok=True)
    file_name = "symbols.parquet"
    file_path = base_path / file_name
    if not file_path.exists():
        return
    df = pl.scan_parquet(file_path).filter(~pl.col("Symbol").is_in(to_remove)).collect()
    df.write_parquet(file_path)

def write_tickers_info(n_limit=100): ###rewrite/update MC if needed
    file_path = Path.cwd() / "data" / "fa_config" / "tickers_database.parquet"
    file_path.parent.mkdir(exist_ok=True, parents=True)

    all_symbols = fetch_symbols().lazy() #nasdaq
    found = pl.scan_parquet(file_path)
    found_symbols = (found.select("Symbol").unique().collect()["Symbol"].to_list())
    missing = (all_symbols.join(found, on="Symbol", how="anti").select("Symbol").unique(maintain_order=True).collect()["Symbol"].to_list())

    if n_limit:
        n_limit = n_limit - len(found_symbols)
        if n_limit <= 0:
            return

    if missing:
        print(f"{len(missing)} missing symbols")
        symbols = missing
        target = ["Symbol", "Market Cap", "country", "industryKey", "sectorKey"]
        rename = {"country": "Country", "industryKey": "Industry", "sectorKey": "Sector"}
        all_frames = []
        to_remove = []
        
        batches = [symbols[i : i + SIZE] for i in range(0, min(v for v in [n_limit, len(symbols)] if v is not None), SIZE)]
        for i, batch in tqdm(enumerate(batches), total=len(batches), desc="Batches"):
            if len(batch) > 1:
                time.sleep(int(SIZE * 0.25 * 2))
            ticker = Ticker(batch, asynchronous=True)
            pf = ticker.summary_profile
            live = ticker.price
            rows = []
            for symbol in pf.keys():
                try:
                    rows.append({"Symbol": symbol, "Market Cap": live[symbol]["marketCap"], **pf[symbol],})
                except Exception as e:
                    to_remove.append(symbol)
                    print(f"[Warning]Failed to write {symbol} due to {e}")
            if rows:
                lf = pl.DataFrame(rows).lazy()
                lf = lf.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in target]))
                df = lf.select(target).collect()
                if not df.is_empty():
                    all_frames.append(df)
        delete_symbols(to_remove)

        if all_frames:
            new_df = pl.concat(all_frames).rename(rename)
            combined = pl.concat([found.collect(), new_df])
            combined = combined.unique(subset=["Symbol"], maintain_order=True).sort(by="Market Cap", descending=True)
            combined.write_parquet(file_path)
            combined.write_csv(file_path.with_suffix(".csv"))

def filter_tickers(n_limit = None, mc_floor = 0, col_filter=["all"], **kwargs): ##redo by mc, ytd, style, factor exposures (cel and flr)
    file_path = Path.cwd() / "data" / "fa_config" / "tickers_database.parquet"
    if not file_path.exists():
        write_tickers_info()
    
    normalized_filter = [f.lower().replace(" ", "_") for f in col_filter]

    lf = pl.scan_parquet(file_path)
    conds = []
    known = {}
    for col in ["Country", "Sector", "Industry"]:
        unique_vals = (
            lf.select(pl.col(col).str.replace(" ", "_").str.to_lowercase())
            .unique()
            .collect()
            .to_series()
            .to_list()
        )
        known[col] = unique_vals

        filters = set(unique_vals) & set(normalized_filter) #atleast 1 valid
        if filters:  
            conds.append(pl.col(col).str.replace(" ", "_").str.to_lowercase().is_in(list(filters)))

    if "all" in col_filter:
        pass
    elif not conds:
        return None
    else:
        lf = lf.filter(pl.all_horizontal(conds))
        
    df = lf.filter(pl.col("Market Cap") > mc_floor)
    df = df.limit(n_limit).collect()
    ###save known_items? for autocomplete?
    return df

