import polars as pl
import pandas as pd
import json
import time
import datetime
from pathlib import Path
from src.utils.df import transpose_df, safe_concat, save_df, load_df, replace_metric
from src.utils.io import fetch_financial, fetch_batch, fetch_summary, fetch_background, fetch_symbols, fetch_dates
from src.utils.fx import adj_fx_table
from src.utils.file import write_to_xlsx, filter_tickers
from src.ui.ui import fmt
from src.utils.history import add_history_data, add_price_factors

def get_expression(df, x, op, dtype, key):
    missing = [col for col in x if (col not in df.columns) and isinstance(col, str)]
    x_filled = [(pl.col(name).fill_null(0) if name in df.columns else pl.lit(0)) for name in x]

    if op == "sum":
        expr = sum(x_filled)
    elif op == "diff":
        expr = x_filled[0] - sum(x_filled[1:])
        
    elif missing:
        expr = pl.lit(None)
    elif op=="":
        expr = pl.col(x[0])
    elif op == "proportion":
        expr = (pl.col(x[0]) * x[1]).round(6)
    elif op == "log":
        expr = pl.when(pl.col(x[0])>0).then(pl.col(x[0]).log1p()).otherwise(pl.lit(0)).round(6)
    elif op == "mult":
        expr = (pl.col(x[0]) * pl.col(x[1])).round(6)
    elif op == "bool":
        expr = (pl.col(x[0]).fill_null(0) > 0)
    elif op == "max":
        expr = pl.col(x[0]).clip(x[1], None)
    elif op == "min":
        expr = pl.col(x[0]).clip(None, x[1])
        
    elif op == "divide":
        num, denom = pl.col(x[0]), pl.col(x[1])
        expr = (
            pl
            .when((num<=0) & (denom<=0))
            .then(pl.lit(None))
            .when(denom ==0)
            .then(pl.lit(None))
            .otherwise((num / denom).round(6))
        )
    elif op == "change":
        num, denom = pl.col(x[0]), pl.col(x[0]).shift(1).over("symbol")
        expr = (
            pl
            .when(((num > 0) & (denom <= 0)) | ((num <= 0) & (denom > 0)))
            .then(pl.lit(None))
            .when(denom ==0)
            .then(pl.lit(None))
            .otherwise(((num / denom - 1)).round(6))
        )
    else:
        expr = pl.lit(None)

    if dtype == pl.Int64: expr = expr.cast(pl.Float64).round(0)
    return expr.cast(dtype).alias(key)
                    
def add_metrics(df, file_name, NEW=False):
    file_path = Path.cwd() / "data" / "fa_config" / f"{file_name}.json"
    try:
        keys = ["symbol", "date"]
        types = {"float": pl.Float64, "int": pl.Int64, "bool": pl.Int8}
        with open(file_path, "r") as f:
            metrics = json.load(f)
        
        for name, session_dict in metrics.items(): #{Section: {m, ..., m}}
            exprs = []
            if "hidden" not in name:
                keys.extend(list(session_dict.keys()))
            for key, val in session_dict.items():  # m = New_Metric: [[Inputs], Operation, Data_Type]
                inputs, operation, data_type = val[0], val[1], types[val[2]]
                expr = get_expression(df, inputs, operation, data_type, key)
                exprs.append(expr)
            df = df.with_columns(exprs) # Apply each section in one vectorized call
    except FileNotFoundError:
        print(f"[WARNING]{file_name}.json not found")
    return df if not NEW else df[keys]
        
def get_combined_financials(symbols, domains, freq):
    base_path = Path.cwd() / "data" / "raw"
    identifiers = ["symbol", "date", "currencyCode"]
    for i, domain in enumerate(domains):
        file_stem = f"_{domain}({freq[0].lower()}).parquet"   
        df = fetch_batch(symbols, base_path, file_stem, fetch_financial, {"domain": domain, "freq": freq}, SAVE=False)
        if not isinstance(df, pl.DataFrame):
            return None
    
        if i==0:
            combined_df = df
        else:
            combined_df = combined_df.join(df, on=identifiers, how="inner")
            combined_df = combined_df.drop([c for c in combined_df.columns if c.endswith("_right")])
    df = combined_df.select(identifiers+[c for c in combined_df.columns if c not in identifiers])
    df = replace_metric(df, "DilutedAverageShares", "BasicAverageShares")
    df = replace_metric(df, "DilutedAverageShares", "OrdinarySharesNumber")
    df = replace_metric(df, "CashFlowFromContinuingOperatingActivities", "OperatingCashFlow")
    df = replace_metric(df, "CashFlowFromContinuingInvestingActivities", "InvestingCashFlow")
    df = replace_metric(df, "CashFlowFromContinuingFinancingActivities", "FinancingCashFlow")
    df = replace_metric(df, "OperatingCashFlow", "CashFlowFromContinuingOperatingActivities")
    df = replace_metric(df, "CashCashEquivalentsAndShortTermInvestments", "CashAndCashEquivalents")
    return df

def get_exponent(min_abs_val):
    for exp in range(12, -1, -3):
        divisor = 10 ** exp
        if abs(min_abs_val)>=divisor:
            if exp>3:
                exp-=3
            return exp
        
def compute_fa_exponent(df): ##redo
    subset = ["NetIncome", "FreeCashFlow", "CommonStockEquity"]
    target = next((c for c in subset if c in df.columns), None)
    min_abs_val = df.group_by("symbol").agg(pl.col(target).abs().median()).sort(target).row(0)[1]
    exp = get_exponent(min_abs_val)
    return exp
        
def get_tables(domains, fa, rv, exp):
    divisor = 10 ** exp
    dfs = {}
    costs = ["COGS", "SG&A", "R&D", "Others", "Operating Expense", "Taxes", "After-Tax Adj"]
    rename_maps = {}
    cols_maps = {}
    for domain in domains:
        file_path = Path.cwd() / "data" / "fa_config" / f"{domain}_rename.json"
        with open(file_path, "r") as f:
            rename_maps[domain] = json.load(f)
            
        file_path = Path.cwd() / "data" / "fa_config" / f"{domain}_add.json"
        with open(file_path, "r") as f:
            metrics = json.load(f)

        cols = ["date"]
        for name, session_dict in metrics.items():
            if "hidden" not in name:
                cols.extend(session_dict.keys())
        cols_maps[domain] = cols
        
    symbols = fa["symbol"].unique(maintain_order=True).to_list()
    for symbol in symbols:
        symbol_fa = fa.filter(pl.col("symbol") == symbol)
        data = {}

        for domain, line_items in rename_maps.items():
            cols = cols_maps[domain]
            temp_fa = symbol_fa.with_columns([pl.lit(None).alias(c) for c in cols if c not in symbol_fa.columns])
            temp_fa = temp_fa.select(cols).rename(line_items)
            temp_fa = temp_fa.with_columns((pl.col(pl.Int64) / divisor).round(0).cast(pl.Int64)) 
            temp_fa = temp_fa.with_columns((pl.col(c)*-1).alias(c) for c in costs if c in temp_fa.columns)
            data[domain] = temp_fa
        if rv is not None:
            data["relative_valuation"] = rv.filter(pl.col("symbol") == symbol).drop("symbol")
        dfs[symbol] = data
    return dfs
    
def get_relative_valuation_data(fa, background):
    fa, history = add_history_data(fa, period="max", interval="1d", SAVE=True)
    rv = add_metrics(fa, "relative_valuation_add", NEW=True)
    rv = add_price_factors(rv, history)
    background = pl.DataFrame({"symbol": key, **val} for key, val in background.items()).drop("fullTimeEmployees")
    rv = rv.join(background, on="symbol", how="inner")
    return rv
        
def get_financial_data(symbols, domains=["income_statement", "cash_flow", "balance_sheet"], freq="annual", **kwargs): 
    base_path = Path.cwd() / "data" / "staging"
    file_stem = f"_combined({freq[0].lower()}).parquet"
    
    cached_df, missing_symbols = load_df(symbols, base_path, file_stem)
    if missing_symbols:
        fa = get_combined_financials(missing_symbols, domains, freq)
        fa = adj_fx_table(fa)
        fa = safe_concat([cached_df, fa]) if len(cached_df) else fa
    else:
        fa = cached_df
        
    if set(domains) == {"income_statement", "cash_flow", "balance_sheet"}:
        save_df(fa, base_path, file_stem)
    for domain in domains:
        fa = add_metrics(fa, f"{domain}_add")
    return fa

def get_desc(symbols, exp): 
    base_path = Path.cwd() / "data" / "staging"

    summary = fetch_batch(symbols, base_path, "_summ.parquet", fetch_summary, SAVE=False)
    background = fetch_batch(symbols, base_path, "_back.parquet", fetch_background, SAVE=True)
    des = summary.join(background, on="symbol", how="full").drop("symbol_right")
    des = pl.DataFrame(des) if not des.is_empty() else None
    
    background_cols = ["country", "industryKey", "sectorKey", "fullTimeEmployees"]
    divisor = 10**exp
    
    des = des.with_columns([(pl.col("Shares") / divisor).round(0).cast(pl.Int64).alias("Shares"),
                            (pl.col("MC") / divisor).round(0).cast(pl.Int64).alias("MC"),
                            (pl.col("Cash") / divisor).round(0).cast(pl.Int64).alias("Cash"),
                            (pl.col("Debt") / divisor).round(0).cast(pl.Int64).alias("Debt"),
                            (pl.col("EV") / divisor).round(0).cast(pl.Int64).alias("EV")])
    summary_cols = [c for c in des.columns if c not in ["symbol"] + background_cols]
    summary = {
        row["symbol"]: {c: row[c] for c in summary_cols}
        for row in des.iter_rows(named=True)
    }
    background = {
        row["symbol"]: {c: row[c] for c in background_cols}
        for row in des.iter_rows(named=True)
    }
    return summary, background

def preload_data(symbols, freq="a", **kwargs): ###
    fa = get_financial_data(symbols, freq=freq) 
    background = fetch_batch(symbols, Path.cwd() / "data" / "staging", "_back.parquet", fetch_background, SAVE=True)
    background = {
        row["symbol"]: {c: row[c] for c in background.columns if c != "symbol"}
        for row in background.iter_rows(named=True)
    }
    return fa, background
    
def save_FA(summary, background, dfs, unit_label):
    base_path = Path.cwd() / "data"/ "output" / "fa_models"
    base_path.mkdir(exist_ok=True)
    dfs = {
        symbol: {
            "summary": (
                pd.DataFrame.from_dict(summary[symbol], orient="index").reset_index()
                .pipe(lambda d: d.rename(columns=d.iloc[0]).drop(d.index[0]).reset_index(drop=True))
            ),
            **{
                domain: transpose_df(
                    df.with_columns(
                        pl.col('date').dt.year().alias('date')
                    ),
                    "date"
                ).fill_null(0).to_pandas()
                for domain, df in data.items()
            }
        }
        for symbol, data in dfs.items()
    }

    for symbol, data in dfs.items():
        file_name = f"{symbol}_FA.xlsx"
        file_path = base_path / file_name
        write_to_xlsx(file_path, data, unit_label, percents=["relative_valuation"]) 
        print(f"Saved {file_name}")

def format_subsets_df(df, subsets, highlights):
    if subsets is not None:
        for parent, children in subsets.items():
            for child in children:
                if child not in df.columns:
                    continue
                if parent in df.columns:
                    df = df.with_columns(pl.when(df[parent].is_null()).then(None).otherwise(df[child]).alias(child))
                if child in df.columns and df.select(pl.col(child).eq(0).all()).item(): #all zero
                    df = df.with_columns(pl.lit(None).alias(child))

    if highlights is not None:
        valid = [h for h in highlights if h in df.columns]
        df = df.select(["date"] + valid)

    df = df.with_columns([pl.when(pl.col(c) == -9999).then(None).otherwise(pl.col(c)).alias(c) for c in df.columns if df.schema[c] in (pl.Int64, pl.Float64)])
    mult_cols = [c for c in df.columns if c in ["EVE", "EVS", "EVB", "PE", "PS", "PB", "EPS", "Beta", "Close"]]
    float_cols = [c for c, dtype in df.schema.items()if c != "date" and dtype == pl.Float64 and c not in mult_cols]
    int_cols = [c for c, dtype in df.schema.items() if c != "date" and dtype == pl.Int64 and c not in mult_cols]
    df = df.with_columns(
        [pl.col(c).map_elements(lambda x: fmt(x, "m")).alias(c) for c in mult_cols] +
        [pl.col(c).map_elements(lambda x: fmt(x, "%")).alias(c) for c in float_cols] +
        [pl.col(c).map_elements(lambda x: fmt(x, ",")).alias(c) for c in int_cols]
    )
    df = df.with_columns([pl.col(c).cast(pl.Utf8) for c in df.columns]).fill_null("")
    return df

def get_highlights(df=None, domains=["income_statement", "cash_flow", "balance_sheet"]):
    cols = ["date", "symbol"]
    rename = []
    for domain in domains:
        file_path = Path.cwd() / "data" / "fa_config" / f"{domain}_add.json"
        with open(file_path, "r") as f:
            metrics = json.load(f)
                   
        for name, session_dict in metrics.items():
            if "hidden" not in name:
                cols.extend(session_dict.keys())
    if df is None:
        return cols
    df = df.with_columns([pl.lit(0).alias(c) for c in cols if c not in df.columns])
    df = df.select(list(dict.fromkeys(cols)))
    return df

def save_model_data(n_symbols, n_dates, identifiers):
    base_path = file_path = Path.cwd() / "data" / "staging"    
    t0 = time.perf_counter()
    symbols = fetch_symbols(n_symbols)
    dates = fetch_dates(n_dates)

    fa, background = preload_data(symbols) 
    df = pl.DataFrame({"symbol": symbols}).join(pl.DataFrame({"date": dates}), how="cross")
    fa = df.join_asof(fa, on="date", by="symbol", strategy="backward", check_sortedness=False)
    #fill null bs ?
    print(f"[INFO]Preloaded in {time.perf_counter() - t0:.2f} seconds")

    t1 = time.perf_counter()
    rv = get_relative_valuation_data(fa, background)
    print(f"[INFO]Retrieved RV in {time.perf_counter() - t1:.2f} seconds")
    
    t2 = time.perf_counter()
    fa = get_highlights(fa)
    fa_cols = [c for c in fa.columns if c not in rv.columns or c in identifiers]
    rv = rv.join(fa.select(fa_cols), on=identifiers, how="left")
    print(f"[INFO]Retrieved FA in {time.perf_counter() - t2:.2f} seconds")
    rv.write_parquet(base_path / f"PF(a).parquet")
    return rv

def load_model_data():
    base_path = file_path = Path.cwd() / "data" / "staging"
    rv = pl.read_parquet(base_path / f"PF(a).parquet")
    return rv

def get_model_data(feat, n_symbols, n_dates, domains=["income_statement", "cash_flow", "balance_sheet", "relative_valuation", "price_factor"], keys=["symbol"], LOAD=True):
    rv = load_model_data() if LOAD else save_model_data(n_symbols, n_dates+2, feat["id"])

    if domains !=None:
        cols = get_highlights(df=None, domains=domains)
        rv = rv.select(cols+feat["cat"])
    rv =  rv.drop_nulls(subset=keys).sort(keys, descending=True)

    all_dates = rv["date"].unique().sort().to_list()
    valid_dates = rv.drop_nulls(subset=feat["target"])["date"].unique().sort().to_list()[-n_dates:]
    dates = valid_dates + all_dates[all_dates.index(valid_dates[-1])+1:]
    rv = rv.filter(pl.col("date").is_in(dates))
    
    symbols = rv["symbol"].unique(maintain_order=True).to_list()[:n_symbols]
    dates = sorted(rv["date"].unique().sort(descending=True).to_list())
    rv = rv.filter((rv["symbol"].is_in(symbols)) & (rv["date"].is_in(dates)))
    print("All Dates: " + ", ".join(d.strftime("%Y-%m-%d") for d in dates))
    print("Train Dates: " + ", ".join(d.strftime("%Y-%m-%d") for d in valid_dates))
    print(f"Symbols: {len(symbols)}")

    feat["num"] = sorted(list(set(rv.columns) - set(feat["id"]+feat["cat"]+feat["target"])))
    rv = rv.select(feat["id"]+feat["num"]+feat["cat"]+feat["target"])
    print(rv[[c for c in rv.columns if c in ["date", "MC", "HML_5", "RevGrowth"]+feat["target"]]].filter(pl.col("date")==dates[0]).head(3))
    print(rv[[c for c in rv.columns if c in ["date", "MC", "HML_5", "RevGrowth"]+feat["target"]]].filter(pl.col("date")==dates[0]).tail(3))
    return rv, symbols, dates, feat

    












