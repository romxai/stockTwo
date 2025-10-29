"""
Data loading and preprocessing module

This module handles loading stock price data (OHLCV - Open, High, Low, Close, Volume),
financial news data, and merging them by date. It also creates target labels for 
supervised learning (binary classification: will price go up or down?).
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
import os


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price data from CSV file
    
    Expects CSV with columns: Date, Open, High, Low, Close, Volume
    Handles various date column naming conventions and ensures proper sorting

    Args:
        file_path: Path to stock CSV file (e.g., 'data/AAPL.csv')

    Returns:
        DataFrame with OHLCV data sorted by date with lowercase column names
        
    Raises:
        FileNotFoundError: If the CSV file doesn't exist
        Exception: For other data loading errors
    """
    print(f"📈 Loading stock data from {file_path}...")

    try:
        # Read CSV file into DataFrame
        stock_data = pd.read_csv(file_path)

        # Ensure date column exists - handle various naming conventions
        if 'Date' in stock_data.columns:
            stock_data.rename(columns={'Date': 'date'}, inplace=True)
        elif 'date' not in stock_data.columns:
            # Check if date is in the index
            if stock_data.index.name in ['Date', 'date']:
                stock_data.reset_index(inplace=True)
                stock_data.rename(columns={stock_data.columns[0]: 'date'}, inplace=True)

        # Convert date strings to datetime objects for proper sorting and manipulation
        stock_data['date'] = pd.to_datetime(stock_data['date'])

        # Standardize all column names to lowercase for consistency
        stock_data.columns = [col.lower() for col in stock_data.columns]

        # Sort chronologically (oldest to newest) and reset index
        stock_data = stock_data.sort_values('date').reset_index(drop=True)

        print(f"   ✅ Loaded {len(stock_data)} days of data")
        print(f"   Date range: {stock_data['date'].min().date()} to {stock_data['date'].max().date()}")

        return stock_data

    except FileNotFoundError:
        print(f"   ❌ ERROR: File '{file_path}' not found!")
        raise
    except Exception as e:
        print(f"   ❌ ERROR loading stock data: {str(e)}")
        raise


def load_news_data(data_path: str = 'data') -> pd.DataFrame:
    """
    Load financial news data from CSV or JSON file
    
    Searches for news.csv or news.json in the data directory.
    JSON format is expected to be an array of objects with 'date', 'title', and 'text' fields.
    CSV format should have 'date', 'title', and 'text' columns.

    Args:
        data_path: Path to data directory containing news files

    Returns:
        DataFrame with columns: date, title, text (standardized format)
        
    Raises:
        FileNotFoundError: If no news files are found
        Exception: For JSON parsing or other loading errors
    """
    print("📰 Loading news data...")

    # Search for news files in supported formats (CSV or JSON)
    news_files = []
    for ext in ['.csv', '.json']:
        potential_file = os.path.join(data_path, f'news{ext}')
        if os.path.exists(potential_file):
            news_files.append(potential_file)

    if not news_files:
        print("❌ ERROR: No news files found in data directory!")
        raise FileNotFoundError(f"No news data files (news.csv or news.json) found in {data_path}")

    # Use the first found file (prefer CSV over JSON if both exist)
    news_file = news_files[0]
    print(f"   Found news file: {news_file}")

    try:
        if news_file.endswith('.csv'):
            # Load CSV format
            news_df = pd.read_csv(news_file)
            print(f"   ✅ Loaded {len(news_df)} news articles from CSV")

        elif news_file.endswith('.json'):
            # Load JSON format (expects array of objects with date, title, text/summary)
            import json
            with open(news_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)

            # Convert JSON objects to standardized DataFrame format
            news_list = []
            for item in news_data:
                # Extract title from JSON (required field)
                title = item.get('title', '')
                # Try multiple date field names (published, published_dt, date)
                date_str = item.get('date', item.get('published', item.get('published_dt', '')))

                # Extract text content, fallback to title if summary is empty
                text = item.get('summary', '').strip()
                if not text:
                    text = title  # Use title as text if no summary available

                news_list.append({
                    'date': date_str,
                    'title': title,
                    'text': text
                })

            news_df = pd.DataFrame(news_list)
            print(f"   ✅ Loaded {len(news_df)} news articles from JSON")

        # Standardize date format - convert strings to datetime, handle invalid dates
        if 'date' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
            # Remove timezone information for compatibility with stock data
            news_df['date'] = news_df['date'].dt.tz_localize(None)

        # Ensure required columns exist, create empty ones if missing
        if 'title' not in news_df.columns:
            news_df['title'] = ''
        if 'text' not in news_df.columns:
            news_df['text'] = news_df['title']  # Fallback: use title as text

        print(f"   Columns: {list(news_df.columns)}")
        return news_df

    except Exception as e:
        print(f"   ❌ ERROR loading news data: {str(e)}")
        raise


def merge_stock_news(stock_df: pd.DataFrame,
                    data_path: str = 'data') -> pd.DataFrame:
    """
    Merge stock and news data by date
    
    Performs a left join: keeps all stock dates, adds news where available.
    Multiple news articles on the same date are aggregated (concatenated).
    Days without news get empty strings for text fields.

    Args:
        stock_df: Stock price DataFrame with 'date' column
        data_path: Path to data directory containing news files

    Returns:
        Merged DataFrame with stock prices + news (title, text, news_combined columns)
        
    Raises:
        ValueError: If news data is empty or invalid
    """
    print("🔗 Merging stock and news data...")

    # Load news data from files
    news_df = load_news_data(data_path)

    # Validate that news data is not empty
    if news_df is None or len(news_df) == 0:
        raise ValueError("News data is empty or invalid")

    # Create copies to avoid modifying original DataFrames
    stock_df = stock_df.copy()
    news_df = news_df.copy()

    # Normalize date formats - remove timezone and time information
    # Step 1: Convert to UTC aware datetime, then remove timezone info
    stock_df['date'] = pd.to_datetime(stock_df['date'], utc=True).dt.tz_localize(None)
    news_df['date'] = pd.to_datetime(news_df['date'], utc=True, errors='coerce').dt.tz_localize(None)

    # Step 2: Extract just the date (remove time component)
    stock_df['date'] = stock_df['date'].dt.date
    news_df['date'] = news_df['date'].dt.date

    # Step 3: Convert back to datetime for merging (pandas requires datetime for joins)
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    news_df['date'] = pd.to_datetime(news_df['date'])

    # Aggregate multiple news articles per day into single entries
    # Join titles with ' | ' separator, concatenate all text with spaces
    news_agg = news_df.groupby('date').agg({
        'title': lambda x: ' | '.join(x.astype(str)),  # Preserve individual titles
        'text': lambda x: ' '.join(x.astype(str))      # Combine all text
    }).reset_index()

    # Left join: keep all stock dates, add news where available
    merged_df = stock_df.merge(news_agg, on='date', how='left')

    # Fill missing news with empty strings (days without news articles)
    merged_df['title'] = merged_df['title'].fillna('')
    merged_df['text'] = merged_df['text'].fillna('')
    # Create combined news field for NLP processing
    merged_df['news_combined'] = merged_df['title'] + ' ' + merged_df['text']

    # Calculate coverage statistics
    rows_with_news = (merged_df['news_combined'].str.len() > 0).sum()

    print(f"   ✅ Merged data: {len(merged_df)} rows")
    print(f"   Rows with news: {rows_with_news} ({rows_with_news/len(merged_df):.1%})")

    return merged_df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target labels for supervised learning
    
    Creates multiple prediction targets:
    - target: Binary classification (0=down, 1=up) based on next day's price
    - next_return: Continuous return percentage for next day
    - target_magnitude: Absolute value of return (how big is the move?)
    - target_volatility: Rolling volatility for next day
    
    These multi-task targets help the model learn different aspects of price movements.

    Args:
        df: DataFrame with 'close' price and 'returns' columns

    Returns:
        DataFrame with additional label columns, last row removed (no future data for it)
    """
    print("🎯 Creating target labels...")

    df = df.copy()

    # Calculate next day's return (percentage change)
    # shift(-1) looks forward one day, so this is the "future" we want to predict
    df['next_return'] = df['close'].shift(-1) / df['close'] - 1

    # Binary classification target: 1 if price goes up, 0 if down
    # This is our main prediction task: will tomorrow's close be higher than today?
    df['target'] = (df['next_return'] > 0).astype(int)

    # Additional regression targets for multi-task learning
    # Magnitude: How much will it move? (helps model understand significance)
    df['target_magnitude'] = np.abs(df['next_return'])
    # Volatility: How uncertain is the next day? (helps with risk assessment)
    df['target_volatility'] = df['returns'].rolling(window=5).std().shift(-1)

    # Remove last row - it has no "next day" data to predict
    df = df[:-1]

    # Print class balance to check for imbalance issues
    class_counts = df['target'].value_counts()
    print(f"\n   Class distribution:")
    for label, count in class_counts.items():
        print(f"     Class {label}: {count} ({count/len(df):.1%})")

    return df
