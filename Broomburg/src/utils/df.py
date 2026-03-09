import polars as pl

def check_df(symbol, base_path, file_stem):
    file_path = base_path / f"{symbol}{file_stem}"
    if not file_path.exists():
        return None
    try:
        df = pl.read_parquet(file_path)
    except:
        return None
    return df if not df.is_empty() else None

def save_df(df, base_path, file_stem):
    for symbol in df["symbol"].unique():
        file_path = base_path / f"{symbol}{file_stem}"
        cur_df = df.filter(pl.col("symbol") == symbol)
        cur_df.write_parquet(file_path)

def load_df(symbols, base_path, file_stem):
    found_dfs = []
    missing_symbols = []
    for symbol in symbols:
        df = check_df(symbol, base_path, file_stem)
        if df is not None:
            found_dfs.append(df)
        else:
            missing_symbols.append(symbol) ##schema aligned problem idk (float, int)
    found_df = safe_concat(found_dfs) if found_dfs else pl.DataFrame()
    return found_df, missing_symbols

def delete_df(symbols, base_path, file_stem):
    for symbol in symbols:
        file_path = base_path / f"{symbol}{file_stem}"
        try:
            file_path.unlink()
        except FileNotFoundError:
            pass

def get_latest(dfs, domain, variables):
    result = {}
    if domain in dfs:
        df = dfs[domain]
        variables = [v for v in variables if v in df.columns]
        for v in variables:
            result[v] = df.select(pl.col(v)).tail(1).item()
    return result

def transpose_df(df, header):
    if header in df.columns:
        try:
            df = df.with_columns(pl.col(header).cast(pl.String))
            df = df.transpose(include_header=True, header_name="", column_names=header) 
        except Exception as e:
            print(f"Could not transpose by {header}: {e}")
    return df

def safe_concat(dfs):
    dfs = [df for df in dfs if df is not None and not df.is_empty()]
    if not dfs:
        return pl.DataFrame()

    schemas = [df.schema for df in dfs]
    promoted = {}
    for schema in schemas:
        for col, dtype in schema.items():
            if col not in promoted:
                promoted[col] = dtype
            else:
                if promoted[col] == pl.Int64 and dtype == pl.Float64:
                    promoted[col] = pl.Float64
                elif promoted[col] == pl.Float64 and dtype == pl.Int64:
                    promoted[col] = pl.Float64

    aligned = []
    for df in dfs:
        df = df.select([
            (df[col].cast(promoted[col]) if col in df.columns
             else pl.lit(None, dtype=promoted[col]).alias(col))
            for col in promoted.keys()
        ])
        aligned.append(df)
    return pl.concat(aligned)

def collapse_df(df, n=4):
    df = df.sort("date")
    keywords = ("shares", "issuance", "market")

    def collapse_group(group):
        results = {"symbol": group["symbol"][0]}
        for col in group.columns:
            if col == "symbol":
                continue
            series = group[col]
            col_lower = col.lower()

            if any(k in col_lower for k in keywords):
                results[col] = series[-1]
            elif series.dtype in (pl.Float64, pl.Int64, pl.Int32, pl.Float32):
                if series.null_count() == len(series):
                    results[col] = None
                else:
                    results[col] = series.tail(n).fill_null(0).sum()
            else:
                results[col] = series[-1]
        return results

    collapsed = []
    for symbol, group in df.group_by("symbol", maintain_order=True):
        collapsed.append(collapse_group(group))
    return pl.DataFrame(collapsed)






















        






