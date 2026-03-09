import polars as pl
from pathlib import Path
from dateutil.relativedelta import relativedelta
from yahooquery import Ticker

def fetch_fx(symbol, start, end):
    print(f"Fetching exchange rate ({symbol})")
    fx = Ticker(symbol)
    start = start - relativedelta(months=1)
    end   = end + relativedelta(months=1)
    df = pl.from_pandas(fx.history(start=start, end=end, interval="1d").reset_index()).select(["date", "close"])
    return df

def get_fx_data(currecy, start, end):
    if currecy == "USD":
        return None
    symbol = f"USD{currecy}=X"
    file_path = Path.cwd() / "data" / "fx_cache" / f"{symbol}.parquet"
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        df = pl.read_parquet(file_path)
        start_loaded = df["date"].min()
        end_loaded = df["date"].max()
    except FileNotFoundError:
        start_loaded=None
        end_loaded=None
        
    if start_loaded is None or end_loaded is None:
        df = fetch_fx(symbol, start, end)
    elif start<start_loaded or end>end_loaded:
        new_df = fetch_fx(symbol, start, end)
        df = pl.concat([df, new_df]).unique(subset=["date"], keep="last").sort("date")
    else:
        return df
    df.write_parquet(file_path)
    return df

def adj_fx_table(df):
    fx_df = {}
    results = []
    currencies = df["currencyCode"].unique()
    for currency in currencies:
        fx_df[currency] = get_fx_data(currency, df["date"].min(), df["date"].max())
        temp_df = df.filter(pl.col("currencyCode")==currency).sort("date")
        if fx_df[currency] is not None:
            fx_data = fx_df[currency].with_columns(pl.col("date").cast(df["date"].dtype))
            temp_df = temp_df.join_asof(fx_data, on="date", strategy="backward").rename({"close": "fx"})
        else:
            temp_df = temp_df.with_columns(pl.lit(1, dtype=pl.Float64).alias("fx"))
        results.append(temp_df)
    df = pl.concat(results).sort("date")
    
    df = df.with_columns(
        (pl.col(col) / pl.col("fx")).cast(pl.Float64).alias(col)
        for col in df.select(pl.selectors.numeric()).columns
        if not any(term in col.lower() for term in ["shares", "issuance", "market"]) and col != "fx"
    )
    return df
