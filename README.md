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

The program asks what you want to download. Enter a number for the first companies:

```text
10
```

Or enter a company name:

```text
Tesla
```

You can also provide the choice directly in the command:

```powershell
python FinancialData.py --sp500 --selection 10
python FinancialData.py --sp500 --selection Tesla
```

### 4. Check the results

Open these files:

- `Tesla_data_2025_YYYYMMDD_HHMMSS.csv` or `top_10_data_2025_YYYYMMDD_HHMMSS.csv`  
  Contains the final stock results. The filename follows your selection, and a new file is created for every run.

- `FinancialData.log`  
  Contains the program time and processing information.

For a quick test with the first 10 companies, use:

```powershell
python FinancialData.py --sp500 --max-tickers 10
```