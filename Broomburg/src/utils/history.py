import polars as pl
import json
import datetime
from datetime import timedelta
from dateutil.relativedelta import relativedelta
from pathlib import Path
from src.utils.df import save_df
from src.utils.io import fetch_batch, fetch_history
from src.utils.math import yang_zhang_vol

def get_history(symbols, period, interval="1d", SAVE=True): 
    base_path = Path.cwd() / "data" / "staging"
    file_stem = f"_history({interval}).parquet"
    history = fetch_batch(symbols, base_path, file_stem, fetch_history, {"period": period, "interval": interval}, SAVE=SAVE).sort(["symbol", "date"])
    history = history.with_columns((pl.col("close").log().diff().over("symbol")).alias("log_ret"))
    history = history.drop_nulls(subset="close")
    ##update history of each symbol 
    return history
    
def add_history_data(financials, period, interval="1d", SAVE=True):
    symbols = financials["symbol"].unique(maintain_order=True).to_list()
    identifiers = ["symbol", "date"]
    history = get_history(symbols, period, interval, SAVE=SAVE)
    financials = financials.sort(identifiers).drop([c for c in history.columns if c not in identifiers and c in financials.columns])
    df = financials.join_asof(history, on="date", by="symbol", strategy="backward", check_sortedness=False)
    return df, history

def get_offset(n, unit_type):
    if n==0: return timedelta(0)
    if unit_type == 'days':
        offset = timedelta(days=n)
    elif unit_type == 'weeks':
        offset = timedelta(weeks=n)
    elif unit_type == 'months':
        offset = relativedelta(months=n)
    elif unit_type == "years":
        n = int(n*12)
        offset = relativedelta(months=n)
    else:
        raise ValueError(f"Unsupported unit_type: {unit_type}")
    return offset
    
def get_price_change(history, date, params, sign=1, target="close", buffer=5): 
    n0, n1, unit_type, sign = params

    start = date + get_offset(n0, unit_type)  
    end  = date + get_offset(n1, unit_type)
    df = history.filter((pl.col("date") >= start) & (pl.col("date") <= end))
    if df.is_empty():
        return None

    first_date = df["date"].first()
    last_date  = df["date"].last()

    if abs((first_date - start).days) > buffer or abs((end - last_date).days) > buffer:
        return None
    
    old = df[target].first()
    new = df[target].last()
    if old is None or new is None:
        return None

    delta = ((new - old) / old) * sign
    return round(delta, 4)

def get_vol(history, date, params, YZ=False):
    n, unit_type = params[0], params[1]
    min_vol_obs = 120

    start = date + get_offset(-n, unit_type)  
    end  = date
    df = history.filter((pl.col("date")>=start) & (pl.col("date")<=end))
    if df.height < min_vol_obs:
        return None
    vol = yang_zhang_vol(df)["vol"].to_numpy()[-1] if YZ else df.select(pl.col("log_ret").std(ddof=1)).item() 
    return vol

def get_corr(history, benchmark, date, params):
    n, unit_type = params[0], params[1]
    min_corr_obs = 750

    start = date + get_offset(-n, unit_type)  
    end  = date
    asset = history.filter((pl.col("date")>=start) & (pl.col("date")<=end))
    bench = benchmark.filter((pl.col("date")>=start) & (pl.col("date")<=end))
    asset = asset.with_columns(pl.col("log_ret").rolling_sum(window_size=3).alias("asset_ret"))
    bench = bench.with_columns(pl.col("log_ret").rolling_sum(window_size=3).alias("bench_ret"))
    df = asset.join(bench, on="date", how="inner")
    if df.height < min_corr_obs:
        return None
    corr = df.select(pl.corr("asset_ret", "bench_ret")).item()  
    return corr

def get_beta(history, benchmark, date, params):
    n_vol, n_corr, unit_type, sign = params[0][0], params[0][1], params[1], params[2]

    vol_asset = get_vol(history, date, [n_vol, unit_type])
    vol_bench = get_vol(benchmark, date, [n_vol, unit_type])
    corr = get_corr(history, benchmark, date, [n_corr, unit_type])
    if vol_asset is None or vol_bench is None or corr is None:
        return None
    beta = corr*(vol_asset/vol_bench) #mathematically equivalent to OLS and Cov/Var

    w = 0.67 #shrinkage
    beta = w * beta + (1 - w) * 1 #industry median
    return round(beta, 4)*sign

def add_price_factors(df, history): #vectorise this date X, ###TA
    symbols = df["symbol"].unique(maintain_order=True).to_list()
    #history = get_history(symbols, period="max", interval="1d", SAVE=True)
    benchmark = get_history(["ACWI"], period="max", interval="1d", SAVE=True) 
    file_path = Path.cwd() / "data" / "fa_config" / f"price_factor_add.json"
    with open(file_path, "r") as f:
        price_config = json.load(f)
    
    rows = []
    for symbol in symbols:
        dates = df.filter(pl.col("symbol")==symbol)["date"].unique(maintain_order=True).to_list()
        sym_history = history.filter(pl.col("symbol")==symbol)

        for date in dates:
            row = {"symbol": symbol, "date": date}
            for func, metrics in price_config.items(): #{func: {m, ..., m}}
                for col_name, params in metrics.items(): #m = {Name: [n, unit_type, sign]}
                    if func == "price_change":
                        row[col_name] = get_price_change(sym_history, date, params)
                    elif func == "vol":
                        row[col_name] = get_vol(sym_history, date, params)
                    elif func == "beta":
                        row[col_name] = get_beta(sym_history, benchmark, date, params)
                    elif func == "performance":
                        row[col_name] = get_price_change(sym_history, date, params)
            rows.append(row)
    pf = pl.DataFrame(rows)
    df = df.join(pf, on=["symbol", "date"], how="full")
    df = df.drop(["symbol_right","date_right"])
    return df

