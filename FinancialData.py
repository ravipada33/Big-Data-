import pandas as pd
import yfinance as yf  # library to get data from Yahoo Finance

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

# Download data for all tickers at once
data = yf.download(
    tickers=list(tickers.values()),
    start=start_date,
    end=end_date,
    interval=interval,
    group_by="ticker",  # makes data["AAPL"], data["MSFT"], etc.
    auto_adjust=True    # adjusts for splits/dividends
)

# Split into separate DataFrames
apple_df = data["AAPL"]
microsoft_df = data["MSFT"]
tesla_df = data["TSLA"]
amazon_df = data["AMZN"]


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

