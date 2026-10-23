# Financial Data Project






## Simple Steps

Open PowerShell and run these commands one by one:

### 1. Go to the project folder

```powershell
cd "D:\1 Study Material\ACTIVE\08. Big Data Technologies\Project\Big-Data-"
```

### 2. Install the required packages

```powershell
python -m pip install -r requirements.txt
```

### 3. Run the program

```powershell
python FinancialData.py --sp500
```

The program will download and process the S&P 500 data.

### 4. Check the results

Open these files:

- `combined_monthly_sp500_2024.csv`  
  Contains the final stock results.

- `FinancialData.log`  
  Contains the program time and processing information.

For a quick test with only 10 companies, use:

```powershell
python FinancialData.py --sp500 --max-tickers 10
```