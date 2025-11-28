import pandas as pd
import yfinance as yf  # library to get data from Yahoo Finance
import logging
from time import perf_counter
from pathlib import Path

# configure logger (creates FinancialData.log next to this script)
log_path = Path(__file__).with_suffix('.log')
logging.basicConfig(
    filename=str(log_path),
    filemode='a',
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s'
)
# also print brief summary to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(logging.Formatter('%(levelname)s: %(message)s'))
logging.getLogger().addHandler(console_handler)

# Define ticker symbols for each company
tickers = {
    "Apple": "AAPL",
    "Microsoft": "MSFT",
    "Tesla": "TSLA",
    "Amazon": "AMZN"
}

# Choose a time period and interval
start_date = "2020-01-01"
end_date = "2025-01-01"
interval = "1d"  # daily data

# --- TIMING: total start ---
_total_start = perf_counter()

# Download data for all tickers at once (measure)
_download_start = perf_counter()
data = yf.download(
    tickers=list(tickers.values()),
    start=start_date,
    end=end_date,
    interval=interval,
    group_by="ticker",  # makes data["AAPL"], data["MSFT"], etc.
    auto_adjust=True    # adjusts for splits/dividends
)
_download_time = perf_counter() - _download_start
logging.info(f"Download completed for {list(tickers.values())} in {_download_time:.3f}s")

# Split into separate DataFrames
apple_df = data["AAPL"]
microsoft_df = data["MSFT"]
tesla_df = data["TSLA"]
amazon_df = data["AMZN"]

# count raw rows per ticker
raw_counts = {
    "AAPL": int(apple_df.shape[0]),
    "MSFT": int(microsoft_df.shape[0]),
    "TSLA": int(tesla_df.shape[0]),
    "AMZN": int(amazon_df.shape[0])
}
logging.info(f"Raw row counts per ticker: {raw_counts}")

# Show basic info
# print("Apple data shape:", apple_df.shape)
# print("Microsoft data shape:", microsoft_df.shape)
# print("Tesla data shape:", tesla_df.shape)
# print("Amazon data shape:", amazon_df.shape)

# print("\nApple head():")
# print(apple_df.head())


# print("Apple start date:", apple_df.index.min())
# print("Microsoft start date:", microsoft_df.index.min())
# print("Tesla start date:", tesla_df.index.min())
# print("Amazon start date:", amazon_df.index.min())



# print("Apple Record Count:", apple_df.shape[0])
# print("Microsoft Record Count:", microsoft_df.shape[0])
# print("Tesla Record Count:", tesla_df.shape[0])
# print("Amazon Record Count:", amazon_df.shape[0])


# Just a test 
# test 2


# print("Apple start date:", apple_df.index.min())
# print("Microsoft start date:", microsoft_df.index.min())
# print("Tesla start date:", tesla_df.index.min())
# print("Amazon start date:", amazon_df.index.min())



# print("Apple column name :", apple_df.columns)
# print("Microsoft column name :", microsoft_df.columns)
# print("Tesla column name :", tesla_df.columns)
# print("Amazon column name :", amazon_df.columns)  


# Data for Apple year 2024
# apple_2024 = apple_df.loc["2024-01-01":"2024-12-31"]
# print("Apple 2024 shape:", apple_2024.shape)
# print(apple_2024.head())
# print(apple_2024.tail())


# --- PROCESSING: compute monthly stats for 2024 (measure) ---
_processing_start = perf_counter()

apple_2024 = apple_df.loc["2024-01-01":"2024-12-31"]
apple_monthly_stats = apple_2024["Close"].resample("ME").agg(["min", "max", "mean"])
apple_monthly_stats.index = apple_monthly_stats.index.strftime("%Y %b")
print("Apple 2024 monthly min/max/avg (Close):")
print(apple_monthly_stats)
print()  
print()

microsoft_2024 = microsoft_df.loc["2024-01-01":"2024-12-31"] 
monthly_stats = microsoft_2024["Close"].resample("ME").agg(["min", "max", "mean"])
monthly_stats.index = monthly_stats.index.strftime("%Y %b")
print(monthly_stats)
print()  
print()

tesla_2024 = tesla_df.loc["2024-01-01":"2024-12-31"]
tesla_monthly_stats = tesla_2024["Close"].resample("ME").agg(["min", "max", "mean"])
tesla_monthly_stats.index = tesla_monthly_stats.index.strftime("%Y %b")
print("Tesla 2024 monthly min/max/avg (Close):")
print(tesla_monthly_stats)
print()  
print()

amazon_2024 = amazon_df.loc["2024-01-01":"2024-12-31"]
amazon_monthly_stats = amazon_2024["Close"].resample("ME").agg(["min", "max", "mean"])
amazon_monthly_stats.index = amazon_monthly_stats.index.strftime("%Y %b")
print("Amazon 2024 monthly min/max/avg (Close):")
print(amazon_monthly_stats)
print()  
print()

_processing_time = perf_counter() - _processing_start
# log rows used in 2024 per ticker
apple_rows_2024 = int(apple_2024.shape[0])
microsoft_rows_2024 = int(microsoft_2024.shape[0])
tesla_rows_2024 = int(tesla_2024.shape[0])
amazon_rows_2024 = int(amazon_2024.shape[0])
logging.info(f"Processing (monthly stats for 2024) completed in {_processing_time:.3f}s")
logging.info(f"Rows in 2024 used per ticker: {{'AAPL':{apple_rows_2024}, 'MSFT':{microsoft_rows_2024}, 'TSLA':{tesla_rows_2024}, 'AMZN':{amazon_rows_2024}}}")

# --- AGGREGATION: combine all monthly stats into one table (measure) ---
_agg_start = perf_counter()
# create DataFrames with period and ticker to keep aggregated table structured
def mk_stats_df(stats_df, ticker):
    df = stats_df.copy()
    df = df.reset_index().rename(columns={"index": "Period"})
    df["Ticker"] = ticker
    return df

apple_stats_df = mk_stats_df(apple_monthly_stats, "AAPL")
ms_stats_df = mk_stats_df(monthly_stats, "MSFT")
tesla_stats_df = mk_stats_df(tesla_monthly_stats, "TSLA")
amazon_stats_df = mk_stats_df(amazon_monthly_stats, "AMZN")

combined_monthly = pd.concat(
    [apple_stats_df, ms_stats_df, tesla_stats_df, amazon_stats_df],
    ignore_index=True
)
_agg_time = perf_counter() - _agg_start
logging.info(f"Aggregation (combined monthly stats) completed in {_agg_time:.3f}s")
logging.info(f"Combined aggregated rows: {int(combined_monthly.shape[0])}")

# Total counts affected
total_raw_rows = sum(raw_counts.values())
total_2024_rows = apple_rows_2024 + microsoft_rows_2024 + tesla_rows_2024 + amazon_rows_2024
total_aggregated_rows = int(combined_monthly.shape[0])

# --- TOTAL TIME ---
_total_time = perf_counter() - _total_start
logging.info(
    f"Total script elapsed time: {_total_time:.3f}s "
    f"(download {_download_time:.3f}s, processing {_processing_time:.3f}s, aggregation {_agg_time:.3f}s)"
)
logging.info(f"Total raw rows processed: {total_raw_rows}, total 2024 rows used: {total_2024_rows}, total aggregated rows: {total_aggregated_rows}")

# Optionally save combined result
output_csv = Path.cwd() / "combined_monthly_2024.csv"
combined_monthly.to_csv(output_csv, index=False)
logging.info(f"Combined monthly CSV written to: {output_csv}")

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

