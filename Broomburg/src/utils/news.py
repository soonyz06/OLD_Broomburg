import polars as pl
import urllib.parse
import feedparser
import datetime
import time
from pathlib import Path
from yahooquery import Ticker
from src.config.config import SIZE, sources
from src.utils.df import safe_concat

def fetch_google_news(symbols, before_date=None, source_filter=""):
    all_frames = []
    all_runs = []
    cutoff = before_date

    for symbol in symbols:
        for src in sources:
            source_filter = f"{src}"
            symbol = symbol.upper() #OR name
            date_filter = f" before:{cutoff.strftime('%Y-%m-%d')}" if cutoff else ""
            query = f'"{symbol}" AND {source_filter}{date_filter}'
            encoded_query = urllib.parse.quote(query)
            url = f'https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en'
            feed = feedparser.parse(url)

            rows = []
            for entry in feed.entries:
                date = datetime.datetime(*entry.get("published_parsed")[:6]).date()
                if cutoff and date >= cutoff:
                    continue
                rows.append({"date": date, "symbol": symbol, "source": src[1:-1], "headline": entry.get("title"), "link": entry.get("link")})
            if rows:
                all_frames.append(pl.DataFrame(rows))
            time.sleep(0.5)

        if all_frames:
            df = safe_concat(all_frames)
            all_runs.append(df)
            cutoff = df["date"].min() 
        else:
            break
    return safe_concat(all_runs).unique(subset="headline") if all_runs else pl.DataFrame()

def fetch_google_news_N(symbols, N=1, sources=sources):
    out_dir = Path.cwd() / "data" / "raw"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_runs = []
    for symbol in symbols:
        path = out_dir / f"{symbol.upper()}_news.parquet"
        if path.exists():
            cached = pl.read_parquet(path)
            print(f"Loaded {path}")
            
            cutoff = cached["date"].max()
            runs = []
            
            for _ in range(0):##
                df = fetch_google_news([symbol], before_date=cutoff)
                if df.is_empty():
                    break
                runs.append(df)
                cutoff = df["date"].min()
            fresh = safe_concat(runs).unique(subset="headline") if runs else pl.DataFrame()
            df = safe_concat([cached, fresh]).unique(subset="headline") #df = cached
        else:
            runs = []
            cutoff = None
            for _ in range(N):
                df = fetch_google_news([symbol], before_date=cutoff)
                if df.is_empty():
                    break
                runs.append(df)
                cutoff = df["date"].min()
            df = safe_concat(runs).unique(subset="headline") if runs else pl.DataFrame()
            
        if not df.is_empty():
            df.write_parquet(path)
            all_runs.append(df)
    return safe_concat(all_runs).unique(subset="headline") if all_runs else pl.DataFrame()

def get_news_tape(symbols):  ###specific rss feed, industry(filter_tickers), alt, 13F/D, insider
    all_frames = []        
    batches = [symbols[i:i+SIZE] for i in range(0, len(symbols), SIZE)]
    print(f"Fetching News -> {len(symbols)}")
    for i, batch in enumerate(batches):
        if len(batch)>1: time.sleep(int(SIZE*.2))
        try:
            news_df = fetch_google_news_N(symbols)
            sec_df = fetch_sec_filings(symbols)
            corp_df = fetch_corporate_events(symbols)
            df = safe_concat([news_df, sec_df, corp_df])
            if not df.is_empty():
                all_frames.append(df)
        except Exception as e:
            print(f"[WARNING]Batch {i} of financial failed: {batch} due to {e}")
                
    if all_frames is None:
        return None    
    df = safe_concat(all_frames).unique(subset=["date", "symbol", "headline"])
    df = df.sort(by="date", descending=True)
    return df

def fetch_sec_filings(symbols, n=100): 
    rows = []
    for symbol in symbols:
        url = f"https://www.sec.gov/cgi-bin/browse-edgar?action=getcompany&CIK={symbol}&type=&dateb=&owner=exclude&count={n}&output=atom"
        headers = {'User-Agent': 'Mozilla/5.0 (compatible; MyApp/1.0; +https://example.com)'}
        feed = feedparser.parse(url, request_headers=headers)

        for entry in feed.entries:
            date = datetime.datetime.fromisoformat(entry.get("updated")).date()
            title = entry.get("title")
            link = entry.get("link")
            rows.append({"date": date, "symbol": symbol, "source": "SEC", "headline": title, "link": link})
        time.sleep(0.5)
    return pl.DataFrame(rows)

def fetch_corporate_events(symbols):
    ticker = Ticker(symbols, asynchronous=True)
    df = pl.from_pandas(ticker.corporate_events.reset_index())
    df = df.select(["date", "symbol", "headline"])
    df = df.with_columns(pl.col("date").dt.date().alias("date"),
                         pl.lit("Corporate Event").alias("source"),
                         pl.lit(None).cast(pl.Utf8).alias("link"))
    df = df.select(["date", "symbol", "source", "headline", "link"])
    return df
