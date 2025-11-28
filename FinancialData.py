import io
import sys
import logging
import argparse
from time import perf_counter
from pathlib import Path
from typing import List, Dict, Tuple

import pandas as pd
import yfinance as yf  # library to get data from Yahoo Finance

# --- Configuration & Logging --------------------------------------------------
LOG_PATH = Path(__file__).with_suffix(".log")


def configure_logging(log_path: Path = LOG_PATH) -> None:
    """Configure file + console logging."""
    logging.basicConfig(
        filename=str(log_path),
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s %(levelname)s: %(message)s",
    )
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(levelname)s: %(message)s"))
    logging.getLogger().addHandler(console_handler)
    logging.info("Logging initialized. Log file: %s", log_path)


# --- Helpers: fetch tickers --------------------------------------------------
def fetch_sp500_symbols(limit: int | None = None) -> List[str]:
    """
    Fetch S&P 500 symbols from public CSV mirrors.
    Returns a list of ticker symbols (periods replaced with '-').
    """
    urls = [
        "https://raw.githubusercontent.com/datasets/s-and-p-500-companies/master/data/constituents.csv",
        "https://datahub.io/core/s-and-p-500-companies/r/constituents.csv",
    ]
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    for url in urls:
        try:
            try:
                import requests

                resp = requests.get(url, headers=headers, timeout=15)
                resp.raise_for_status()
                text = resp.text
            except Exception:
                import urllib.request

                req = urllib.request.Request(url, headers=headers)
                with urllib.request.urlopen(req, timeout=15) as r:
                    text = r.read().decode("utf-8")
            df = pd.read_csv(io.StringIO(text))
            syms = df["Symbol"].astype(str).str.replace(".", "-", regex=False).tolist()
            if limit:
                return syms[:limit]
            return syms
        except Exception as exc:
            logging.warning("Failed to fetch S&P500 from %s: %s", url, exc)
    raise RuntimeError(
        "Unable to fetch S&P 500 constituents; provide a local tickers.txt or check network."
    )


def load_tickers_from_file(path: Path, limit: int | None = None) -> List[str]:
    """Read one ticker per line from a file (ignore comments / blank lines)."""
    lines = []
    with open(path, "r", encoding="utf-8") as fh:
        for ln in fh:
            s = ln.split("#", 1)[0].strip()
            if not s:
                continue
            lines.append(s.upper())
            if limit and len(lines) >= limit:
                break
    # remove duplicates while preserving order
    seen = set()
    out = []
    for t in lines:
        if t not in seen:
            seen.add(t)
            out.append(t)
    return out


# --- Download (chunked) ------------------------------------------------------
def chunked_download(
    ticker_list: List[str],
    chunk_size: int = 100,
    **yf_kwargs,
) -> Dict[str, pd.DataFrame]:
    """
    Download data in chunks and return a dict ticker -> DataFrame.
    Handles both multi-ticker and single-ticker returns from yfinance.
    """
    frames: Dict[str, pd.DataFrame] = {}
    for i in range(0, len(ticker_list), chunk_size):
        chunk = ticker_list[i : i + chunk_size]
        logging.info("Downloading chunk %d (%d tickers)...", i // chunk_size + 1, len(chunk))
        df = yf.download(tickers=chunk, **yf_kwargs, group_by="ticker", auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            for tk in chunk:
                if tk in df:
                    frames[tk] = df[tk].copy()
                else:
                    logging.warning("No data returned for %s in this chunk", tk)
        else:
            # single ticker returned
            frames[chunk[0]] = df.copy()
    return frames


# --- Processing single ticker ------------------------------------------------
def monthly_stats_for_period(
    df: pd.DataFrame, period_start: str, period_end: str
) -> Tuple[pd.DataFrame, int]:
    """
    Compute month-end min/max/mean of Close between period_start and period_end.
    Returns (stats_df, rows_used).
    """
    df.index = pd.to_datetime(df.index)
    df_period = df.loc[period_start:period_end]
    rows_used = int(df_period.shape[0])
    if rows_used == 0:
        return pd.DataFrame(), 0
    stats = df_period["Close"].resample("ME").agg(["min", "max", "mean"])
    stats.index = stats.index.strftime("%Y %b")
    stats = stats.reset_index().rename(columns={"index": "Period"})
    return stats, rows_used


# --- Main S&P500 flow --------------------------------------------------------
def run_sp500(
    max_tickers: int | None = None,
    chunk_size: int = 100,
    start_date: str = "2020-01-01",
    end_date: str = "2025-01-01",
    interval: str = "1d",
) -> None:
    _total_start = perf_counter()
    logging.info("SP500 mode enabled — fetching S&P 500 symbols")
    tickers_list = fetch_sp500_symbols(limit=max_tickers)
    logging.info("Loaded %d symbols (sample: %s)", len(tickers_list), tickers_list[:6])

    _download_start = perf_counter()
    data_dict = chunked_download(
        tickers_list, chunk_size=chunk_size, start=start_date, end=end_date, interval=interval
    )
    _download_time = perf_counter() - _download_start
    logging.info("Download completed for %d tickers in %.3fs", len(data_dict), _download_time)

    raw_counts = {tk: int(df.shape[0]) for tk, df in data_dict.items()}
    logging.info("Raw row counts (sample 10): %s", dict(list(raw_counts.items())[:10]))

    _processing_start = perf_counter()
    stats_frames = []
    rows_2024_counts = {}
    for tk, df in data_dict.items():
        try:
            stats, rows_used = monthly_stats_for_period(df, "2024-01-01", "2024-12-31")
            rows_2024_counts[tk] = rows_used
            if rows_used == 0 or stats.empty:
                continue
            stats["Ticker"] = tk
            stats_frames.append(stats)
        except Exception as exc:
            logging.warning("Failed to process %s: %s", tk, exc)
    _processing_time = perf_counter() - _processing_start
    logging.info("Processing (monthly stats for 2024) completed in %.3fs", _processing_time)

    _agg_start = perf_counter()
    if stats_frames:
        combined_monthly = pd.concat(stats_frames, ignore_index=True)
    else:
        combined_monthly = pd.DataFrame(columns=["Period", "min", "max", "mean", "Ticker"])
    _agg_time = perf_counter() - _agg_start
    logging.info("Aggregation completed in %.3fs; combined rows: %d", _agg_time, combined_monthly.shape[0])

    # Totals & save
    total_raw_rows = sum(raw_counts.values())
    total_2024_rows = sum(rows_2024_counts.values())
    total_aggregated_rows = int(combined_monthly.shape[0])
    _total_time = perf_counter() - _total_start
    logging.info(
        "Total elapsed: %.3fs (download %.3fs, processing %.3fs, aggregation %.3fs)",
        _total_time,
        _download_time,
        _processing_time,
        _agg_time,
    )
    logging.info(
        "Counts - raw: %d, rows_2024: %d, aggregated: %d",
        total_raw_rows,
        total_2024_rows,
        total_aggregated_rows,
    )

    out_csv = Path.cwd() / "combined_monthly_sp500_2024.csv"
    combined_monthly.to_csv(out_csv, index=False)
    logging.info("Saved combined CSV to %s", out_csv)
    print(f"\nS&P 500 processing complete. Results saved to {out_csv}")


# --- CLI ---------------------------------------------------------------------
def main() -> None:
    configure_logging()
    parser = argparse.ArgumentParser(description="Financial data processor")
    parser.add_argument("--sp500", action="store_true", help="Process the full S&P 500")
    parser.add_argument("--max-tickers", type=int, default=None, help="Limit tickers (for testing)")
    parser.add_argument("--chunk-size", type=int, default=100, help="yf.download chunk size")
    args = parser.parse_args()

    if args.sp500:
        try:
            run_sp500(max_tickers=args.max_tickers, chunk_size=args.chunk_size)
        except Exception as exc:
            logging.exception("Unhandled error in SP500 flow: %s", exc)
            raise
    else:
        # Keep your original hardcoded learning flow commented for reference:
        # ============ HARDCODED 4 TICKERS FLOW (Your Original Learning Code - Now Commented) ============
        # Choose a time period and interval
        # start_date = "2020-01-01"
        # end_date = "2025-01-01"
        # interval = "1d"  # daily data
        
        # # Define ticker symbols for each company
        # tickers = {
        #     "Apple": "AAPL",
        #     "Microsoft": "MSFT",
        #     "Tesla": "TSLA",
        #     "Amazon": "AMZN"
        # }
        
        # # --- TIMING: total start ---
        # _total_start = perf_counter()
        
        # # Download data for all tickers at once (measure)
        # _download_start = perf_counter()
        # data = yf.download(
        #     tickers=list(tickers.values()),
        #     start=start_date,
        #     end=end_date,
        #     interval=interval,
        #     group_by="ticker",
        #     auto_adjust=True
        # )
        # _download_time = perf_counter() - _download_start
        # logging.info(f"Download completed for {list(tickers.values())} in {_download_time:.3f}s")
        
        # # Split into separate DataFrames
        # apple_df = data["AAPL"]
        # microsoft_df = data["MSFT"]
        # tesla_df = data["TSLA"]
        # amazon_df = data["AMZN"]
        
        # # count raw rows per ticker
        # raw_counts = {
        #     "AAPL": int(apple_df.shape[0]),
        #     "MSFT": int(microsoft_df.shape[0]),
        #     "TSLA": int(tesla_df.shape[0]),
        #     "AMZN": int(amazon_df.shape[0])
        # }
        # logging.info(f"Raw row counts per ticker: {raw_counts}")
        
        # # Show basic info
        # # print("Apple data shape:", apple_df.shape)
        # # print("Microsoft data shape:", microsoft_df.shape)
        # # print("Tesla data shape:", tesla_df.shape)
        # # print("Amazon data shape:", amazon_df.shape)
        # 
        # # print("\nApple head():")
        # # print(apple_df.head())
        # 
        # # print("Apple start date:", apple_df.index.min())
        # # print("Microsoft start date:", microsoft_df.index.min())
        # # print("Tesla start date:", tesla_df.index.min())
        # # print("Amazon start date:", amazon_df.index.min())
        # 
        # # print("Apple Record Count:", apple_df.shape[0])
        # # print("Microsoft Record Count:", microsoft_df.shape[0])
        # # print("Tesla Record Count:", tesla_df.shape[0])
        # # print("Amazon Record Count:", amazon_df.shape[0])
        # 
        # # Just a test 
        # # test 2
        # 
        # # print("Apple start date:", apple_df.index.min())
        # # print("Microsoft start date:", microsoft_df.index.min())
        # # print("Tesla start date:", tesla_df.index.min())
        # # print("Amazon start date:", amazon_df.index.min())
        # 
        # # print("Apple column name :", apple_df.columns)
        # # print("Microsoft column name :", microsoft_df.columns)
        # # print("Tesla column name :", tesla_df.columns)
        # # print("Amazon column name :", amazon_df.columns)  
        # 
        # # Data for Apple year 2024
        # # apple_2024 = apple_df.loc["2024-01-01":"2024-12-31"]
        # # print("Apple 2024 shape:", apple_2024.shape)
        # # print(apple_2024.head())
        # # print(apple_2024.tail())
        # 
        # # --- PROCESSING: compute monthly stats for 2024 (measure) ---
        # _processing_start = perf_counter()
        # 
        # apple_2024 = apple_df.loc["2024-01-01":"2024-12-31"]
        # apple_monthly_stats = apple_2024["Close"].resample("ME").agg(["min", "max", "mean"])
        # apple_monthly_stats.index = apple_monthly_stats.index.strftime("%Y %b")
        # print("Apple 2024 monthly min/max/avg (Close):")
        # print(apple_monthly_stats)
        # print()  
        # print()
        # 
        # microsoft_2024 = microsoft_df.loc["2024-01-01":"2024-12-31"] 
        # monthly_stats = microsoft_2024["Close"].resample("ME").agg(["min", "max", "mean"])
        # monthly_stats.index = monthly_stats.index.strftime("%Y %b")
        # print(monthly_stats)
        # print()  
        # print()
        # 
        # tesla_2024 = tesla_df.loc["2024-01-01":"2024-12-31"]
        # tesla_monthly_stats = tesla_2024["Close"].resample("ME").agg(["min", "max", "mean"])
        # tesla_monthly_stats.index = tesla_monthly_stats.index.strftime("%Y %b")
        # print("Tesla 2024 monthly min/max/avg (Close):")
        # print(tesla_monthly_stats)
        # print()  
        # print()
        # 
        # amazon_2024 = amazon_df.loc["2024-01-01":"2024-12-31"]
        # amazon_monthly_stats = amazon_2024["Close"].resample("ME").agg(["min", "max", "mean"])
        # amazon_monthly_stats.index = amazon_monthly_stats.index.strftime("%Y %b")
        # print("Amazon 2024 monthly min/max/avg (Close):")
        # print(amazon_monthly_stats)
        # print()  
        # print()
        # 
        # _processing_time = perf_counter() - _processing_start
        # # log rows used in 2024 per ticker
        # apple_rows_2024 = int(apple_2024.shape[0])
        # microsoft_rows_2024 = int(microsoft_2024.shape[0])
        # tesla_rows_2024 = int(tesla_2024.shape[0])
        # amazon_rows_2024 = int(amazon_2024.shape[0])
        # logging.info(f"Processing (monthly stats for 2024) completed in {_processing_time:.3f}s")
        # logging.info(f"Rows in 2024 used per ticker: {{'AAPL':{apple_rows_2024}, 'MSFT':{microsoft_rows_2024}, 'TSLA':{tesla_rows_2024}, 'AMZN':{amazon_rows_2024}}}")
        # 
        # # --- AGGREGATION: combine all monthly stats into one table (measure) ---
        # _agg_start = perf_counter()
        # # create DataFrames with period and ticker to keep aggregated table structured
        # def mk_stats_df(stats_df, ticker):
        #     df = stats_df.copy()
        #     df = df.reset_index().rename(columns={"index": "Period"})
        #     df["Ticker"] = ticker
        #     return df
        # 
        # apple_stats_df = mk_stats_df(apple_monthly_stats, "AAPL")
        # ms_stats_df = mk_stats_df(monthly_stats, "MSFT")
        # tesla_stats_df = mk_stats_df(tesla_monthly_stats, "TSLA")
        # amazon_stats_df = mk_stats_df(amazon_monthly_stats, "AMZN")
        # 
        # combined_monthly = pd.concat(
        #     [apple_stats_df, ms_stats_df, tesla_stats_df, amazon_stats_df],
        #     ignore_index=True
        # )
        # _agg_time = perf_counter() - _agg_start
        # logging.info(f"Aggregation (combined monthly stats) completed in {_agg_time:.3f}s")
        # logging.info(f"Combined aggregated rows: {int(combined_monthly.shape[0])}")
        # 
        # # Total counts affected
        # total_raw_rows = sum(raw_counts.values())
        # total_2024_rows = apple_rows_2024 + microsoft_rows_2024 + tesla_rows_2024 + amazon_rows_2024
        # total_aggregated_rows = int(combined_monthly.shape[0])
        # 
        # # --- TOTAL TIME ---
        # _total_time = perf_counter() - _total_start
        # logging.info(
        #     f"Total script elapsed time: {_total_time:.3f}s "
        #     f"(download {_download_time:.3f}s, processing {_processing_time:.3f}s, aggregation {_agg_time:.3f}s)"
        # )
        # logging.info(f"Total raw rows processed: {total_raw_rows}, total 2024 rows used: {total_2024_rows}, total aggregated rows: {total_aggregated_rows}")
        # 
        # # Optionally save combined result
        # output_csv = Path.cwd() / "combined_monthly_2024.csv"
        # combined_monthly.to_csv(output_csv, index=False)
        # logging.info(f"Combined monthly CSV written to: {output_csv}")
        
        # --- Function Explanations (applies to code above) ---
        # The print() function below simply adds a blank line for output readability.
        
        # .loc[] : This is a pandas DataFrame function for selecting rows and/or columns by specific *labels*. 
        #          For example, using .loc["2024-01-01":"2024-12-31"] extracts all rows with date labels within this range, inclusive.
        #          This is very useful for working with time series data.
        
        # .resample("ME") : This pandas method is used to group time series data by a new frequency. 
        #                   "ME" means "Month End"; it creates groups ending on the last calendar day of each month.
        #                   After resampling, you can apply aggregation functions like min, max, mean, etc. to these groups.
        
        # .agg(["min", "max", "mean"]) : This is a way to apply multiple aggregation functions at once to your grouped data.
        #                                For example, after resampling monthly, this will output the minimum, maximum, and average value for each month.
        
        # *Note*: Each time .loc and .resample are used in code above, they work the same way, selecting date ranges and grouping the "Close" column by month respectively.
        
        print("Hardcoded flow is preserved as comments. Run with --sp500 to process S&P 500.")


if __name__ == "__main__":
    main()

# The original hardcoded block you used for learning remains commented near the end of this file.

