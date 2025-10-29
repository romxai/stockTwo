# Data Folder

## 📂 Place Your Data Here

This folder should contain your stock price data CSV file.

### Required File

- **AAPL.csv** (or any stock ticker) - Stock price data in OHLCV format

### CSV Format

Your CSV file should have these columns:

```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.0,105.0,99.0,103.0,1000000
2020-01-02,103.0,107.0,102.0,106.0,1200000
2020-01-03,106.0,108.0,105.0,107.5,1100000
...
```

### Column Descriptions

- **Date**: Trading date (YYYY-MM-DD format)
- **Open**: Opening price
- **High**: Highest price of the day
- **Low**: Lowest price of the day
- **Close**: Closing price
- **Volume**: Number of shares traded

### Where to Get Data

You can download stock data from:

1. **Yahoo Finance**: https://finance.yahoo.com/

   - Search for a stock (e.g., AAPL)
   - Go to "Historical Data"
   - Select date range
   - Click "Download" → saves as CSV

2. **CSV from Code**: The pipeline can also download data automatically using `yfinance`

### Optional Files

- **news.json** - Financial news articles in JSON format (recommended)
- **news.csv** - Financial news articles in CSV format (alternative)

### News Data Formats

#### JSON Format (news.json)

Array of news article objects:

```json
[
  {
    "title": "Here's How Much Traders Expect Apple Stock To Move After Earnings This Week - Investopedia",
    "summary": "",
    "date": "2025-10-28T18:24:50",
    "symbol": "AAPL"
  },
  {
    "title": "Apple Announces New Product Line",
    "summary": "Apple unveiled new products today...",
    "date": "2025-10-29T10:00:00",
    "symbol": "AAPL"
  }
]
```

**Required fields:**

- `title`: News headline (used as primary text)
- `date`: Publication date (ISO format: YYYY-MM-DDTHH:MM:SS)

**Optional fields:**

- `summary`: Full article text (if empty, title is used)
- `symbol`: Stock ticker (for filtering)

#### CSV Format (news.csv)

```csv
date,title,text
2025-10-28,Apple earnings beat expectations,Apple reported strong quarterly earnings exceeding analyst expectations...
2025-10-29,Tech stocks rally,Technology stocks rose today on positive economic data...
```

**Columns:**

- `date`: Publication date
- `title`: News headline
- `text`: Full article text

### Notes

- Column names are **case-insensitive** (`Date` or `date` both work)
- Date format should be parseable by pandas (YYYY-MM-DD recommended)
- If no CSV file is found, the system will create **synthetic data** for demonstration

### Example: Download Data with Python

```python
import yfinance as yf

# Download Apple stock data
ticker = yf.Ticker("AAPL")
df = ticker.history(start="2020-01-01", end="2024-12-31")
df.to_csv("AAPL.csv")
```

### File Size

- Typical size: 100KB - 5MB depending on date range
- Recommended: 3-5 years of data for best results
