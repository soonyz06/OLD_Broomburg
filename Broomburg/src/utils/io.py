import polars as pl
from yahooquery import Ticker, Screener
from pathlib import Path
import json
import time
import datetime
from src.utils.math import close_to_close_vol, yang_zhang_vol, garch_vol, calc_VaR
from src.config.config import SIZE
from src.utils.df import safe_concat, save_df, load_df, collapse_df
from src.utils.fx import adj_fx_table

def fetch_dates(n=3):
    today = datetime.date.today()
    year = today.year

    past_dates = []
    y = year
    while len(past_dates) < n and y > 0:
        d = datetime.date(year=y, month=4, day=3)
        if d <= today:
            past_dates.append(d)
        y -= 1
    past_dates = sorted(past_dates)
    return past_dates + [today]

def fetch_symbols(n):
    file_path = Path.cwd() / "data" / "fa_config" / "symbols.parquet" #5610
    df = pl.scan_parquet(file_path).limit(n).collect()
    df = df.filter(~pl.col("Symbol").is_in(["TEF", "TLK", "SQM", "LTM"]))
    symbols = df["Symbol"].to_list()
    return symbols

def fetch_history(symbols, period, interval):
    print(f"Fetching History({period},{interval}) -> {len(symbols)}")
    ticker = Ticker(symbols)
    retrieved = ticker.history(period=period, interval=interval)
    df = pl.from_pandas(retrieved.reset_index()).select(["symbol", "date", "open", "high", "low", "close", "volume", "adjclose"])
    df = df.with_columns([pl.col(c).round(2) for c in df.columns if c not in ["symbol", "date"]])
    return df if not df.is_empty() else None
    
def fetch_financial(symbols, domain, freq="annual"): 
    ticker = Ticker(symbols, asynchronous=True)
    print(f"Fetching {domain.title()}({freq[0]}) -> {len(symbols)}")
        
    if domain == "income_statement":
        retrieved = ticker.income_statement(frequency=freq, trailing=False)
        df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="NetIncome")
    elif domain == "cash_flow":
        retrieved = ticker.cash_flow(frequency=freq, trailing=False)
        df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="OperatingCashFlow")
    elif domain == "balance_sheet":
        retrieved = ticker.balance_sheet(frequency=freq)
        df = pl.from_pandas(retrieved.reset_index()).drop_nulls(subset="CommonStockEquity")
        df = df.with_columns(pl.col("OrdinarySharesNumber").forward_fill().over("symbol"))
    else:
        print("Invalid domain")
        return None
    
    df = df.with_columns(pl.col("asOfDate").dt.date().alias("date")).drop("asOfDate")
    df = df.select(["date"] + [c for c in df.columns if c != "date"])
    return df if not df.is_empty() else None

def fetch_summary(symbols):
    print(f"Fetching Summary -> {len(symbols)}")
    fill_cols = ["CurrentDebtAndCapitalLeaseObligation", "LongTermDebtAndCapitalLeaseObligation", "CashCashEquivalentsAndShortTermInvestments"]

    ticker = Ticker(symbols, asynchronous=True)
    live = ticker.price
    df = fetch_financial(symbols, "balance_sheet", freq="q")
    df = adj_fx_table(df)
    expr = [pl.col(c).fill_null(0) if c in df.columns else pl.lit(0).alias(c) for c in fill_cols]
    df = df.with_columns(expr)

    rows = []
    for symbol in symbols:
        latest = df.filter(pl.col("symbol") == symbol)[-1]
        price = live[symbol]["regularMarketPrice"]
        mc = live[symbol]["marketCap"]
        debt = latest["CurrentDebtAndCapitalLeaseObligation"][0] + latest["LongTermDebtAndCapitalLeaseObligation"][0]
        cash = latest["CashCashEquivalentsAndShortTermInvestments"][0]
        rows.append({
            "symbol": symbol,
            "Symbol": symbol,
            "Price": price,
            "Shares": round(mc/price),
            "MC": mc,
            "Cash": cash,
            "Debt": debt,
            "EV": mc + debt - cash
        })
    return pl.DataFrame(rows)    

def fetch_background(symbols):
    print(f"Fetching Background -> {len(symbols)}")
    background_cols = ["symbol", "country", "industryKey", "sectorKey", "fullTimeEmployees"]
    ticker = Ticker(symbols, asynchronous=True)
    pf = ticker.summary_profile
    valid_symbols = pf.keys()
    rows = []
    for symbol in valid_symbols:
        rows.append({"symbol": symbol, **pf[symbol]})
    df = pl.DataFrame(rows)
    df = df.with_columns(pl.lit(None).alias(c) for c in background_cols if c not in df.columns)
    return df.select(background_cols)

def fetch_batch(symbols, base_path, file_stem, function, params={}, SAVE=False):
    if SAVE:
        found_df, missing_symbols = load_df(symbols, base_path, file_stem)
    else:
        found_df, missing_symbols = pl.DataFrame(), symbols
        
    all_frames = []
    if missing_symbols:
        batches = [missing_symbols[i:i+SIZE] for i in range(0, len(missing_symbols), SIZE)]
        for i, batch in enumerate(batches):
            time.sleep(int(len(batch)*.5))
            try:
                df = function(batch, **params)
                if SAVE:
                    save_df(df, base_path, file_stem)
                if df is not None:
                    all_frames.append(df)
            except Exception as e:
                print(f"[WARNING]Batch {i} failed: {batch} due to {e}")

    if not found_df.is_empty():
        all_frames.append(found_df)
    return safe_concat(all_frames) if all_frames else None

def fetch_price_data(symbols, exp=None):
    print("[INFO]Getting live price data...")
    ticker = Ticker(symbols)
    live = {}
    for symbol in symbols:
        price_data = ticker.price[symbol]
        price = price_data["regularMarketPrice"]
        live[symbol] = {"Price": price}
        if exp is not None:
            divisor = 10 ** exp
            mc = price_data["marketCap"]
            live[symbol]["mc"] = mc / divisor
        else:
            #keys = ["preMarketChangePercent", "regularMarketChangePercent", "postMarketChangePercent"] ##include session data
            keys = ["regularMarketChangePercent"]
            delta = next((price_data[k] for k in keys if k in price_data), None)
            live[symbol]["pctChange"] = round(delta*100, 2)
    return live

def fetch_screeners(filters=['most_watched_tickers', 'most_actives', 'day_gainers', 'day_losers']):
    screeners = {}
    print(f"Fetching Live Screeners -> {len(filters)}")
    for f in filters:
        data = Screener().get_screeners(f, count=30)
        rows = [{"symbol": f"{item.get('symbol')} ({item.get('shortName', '')})", "price": item.get('regularMarketPrice'), "pctChange": round(item.get('regularMarketChangePercent'), 2)}
                for item in data[f]['quotes']]
        df = pl.DataFrame(rows)
        time.sleep(0.5)
        screeners[data[f].get("title", f)] = df
    return screeners

def fetch_etfs():
    file_path = Path.cwd() / "data" / "fa_config" / "etfs.json"
    file_path.parent.mkdir(exist_ok=True, parents=True)
    if not file_path.exists():
        return None
    with open(file_path, "r") as f:
        etfs_dict = json.load(f)

    result = {}
    for category, symbol_dict in etfs_dict.items():
        all_frames = []
        symbols = list(symbol_dict.keys())
        batches = [symbols[i:i+SIZE] for i in range(0, len(symbols), SIZE)]
        print(f"Fetching Live Indices -> {len(symbols)}")
        for i, batch in enumerate(batches):
            time.sleep(int(len(batch)*.2))
            try:            
                live = fetch_price_data(batch)
                rows = []
                for key, values in live.items():
                    rows.append({"symbol": f"{key} ({symbol_dict[key]})", **values})
                df = pl.DataFrame(rows)
                all_frames.append(df)
            except Exception as e:
                print(f"[WARNING]Batch {i} of description failed: {batch} due to {e}")
        result[category] = safe_concat(all_frames)
        time.sleep(0.5)
    return result

def fetch_vol(symbols, period="5y", interval="1d"):
    ticker = Ticker(symbols)
    df_all = pl.from_pandas(ticker.history(period=period, interval=interval).reset_index())

    rows = []
    for symbol in symbols:
        df = df_all.filter(pl.col("symbol") == symbol)
        df = df.with_columns((pl.col("close").log().diff()*100).alias("log_returns")).drop_nulls()
        row = {"Symbol": symbol}
        
        vol_df = close_to_close_vol(df)
        row["Close To Close Vol"] = vol_df["vol"].to_numpy()[-1]*100

        vol_df = yang_zhang_vol(df)
        row["Yang Zhang Vol"] = vol_df["vol"].to_numpy()[-1]*100

        vol, params = garch_vol(df)
        row["Forecasted Vol"] = vol

        alpha=0.01
        VaR = calc_VaR(vol, nu=params["nu"], mu=0, alpha=alpha)
        row[f"VaR (α={(1-alpha):.2f})"] = VaR

        rows.append(row)
    return pl.DataFrame(rows)
