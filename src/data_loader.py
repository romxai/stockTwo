"""
Data loading and preprocessing module
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Tuple, List
import os


def load_stock_data(file_path: str) -> pd.DataFrame:
    """
    Load stock price data from CSV file

    Args:
        file_path: Path to stock CSV file

    Returns:
        DataFrame with OHLCV data
    """
    print(f"📈 Loading stock data from {file_path}...")

    try:
        stock_data = pd.read_csv(file_path)

        # Ensure date column exists
        if 'Date' in stock_data.columns:
            stock_data.rename(columns={'Date': 'date'}, inplace=True)
        elif 'date' not in stock_data.columns:
            if stock_data.index.name in ['Date', 'date']:
                stock_data.reset_index(inplace=True)
                stock_data.rename(columns={stock_data.columns[0]: 'date'}, inplace=True)

        # Convert to datetime
        stock_data['date'] = pd.to_datetime(stock_data['date'])

        # Rename columns to lowercase
        stock_data.columns = [col.lower() for col in stock_data.columns]

        # Sort by date
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


def create_synthetic_news(stock_df: pd.DataFrame) -> pd.DataFrame:
    """
    Create synthetic news data for demonstration

    Args:
        stock_df: Stock price DataFrame

    Returns:
        Synthetic news DataFrame
    """
    print("🔬 Creating synthetic news data...")

    dates = stock_df['date'].values

    news_templates = [
        "Company announces strong quarterly earnings, exceeding analyst expectations.",
        "Market volatility affects tech stocks as investors await Fed decision.",
        "Company unveils new product lineup, stock rises on positive reception.",
        "Concerns about supply chain disruptions weigh on shares.",
        "Analysts upgrade stock target price citing strong demand.",
        "Company faces regulatory challenges in key markets.",
        "Strong consumer demand drives revenue growth.",
        "Tech sector pullback affects stock performance."
    ]

    news_data = []
    for date in dates:
        n_articles = np.random.randint(2, 6)
        for _ in range(n_articles):
            news_data.append({
                'date': date,
                'title': np.random.choice(news_templates),
                'text': np.random.choice(news_templates) + " " + np.random.choice(news_templates)
            })

    news_df = pd.DataFrame(news_data)
    print(f"   ✅ Created {len(news_df)} synthetic news articles")

    return news_df


def load_news_data(data_path: str = 'data') -> pd.DataFrame:
    """
    Load financial news data from CSV or JSON file

    Args:
        data_path: Path to data directory

    Returns:
        DataFrame with news articles
    """
    print("📰 Loading news data...")

    # Try different file formats
    news_files = []
    for ext in ['.csv', '.json']:
        potential_file = os.path.join(data_path, f'news{ext}')
        if os.path.exists(potential_file):
            news_files.append(potential_file)

    if not news_files:
        print("⚠️  No news files found, creating synthetic news data for demonstration...")
        return create_synthetic_news(pd.DataFrame())  # Will be called with stock_df later

    # Use the first found file
    news_file = news_files[0]
    print(f"   Found news file: {news_file}")

    try:
        if news_file.endswith('.csv'):
            # Load CSV format
            news_df = pd.read_csv(news_file)
            print(f"   ✅ Loaded {len(news_df)} news articles from CSV")

        elif news_file.endswith('.json'):
            # Load JSON format (array of objects)
            import json
            with open(news_file, 'r', encoding='utf-8') as f:
                news_data = json.load(f)

            # Convert to DataFrame
            news_list = []
            for item in news_data:
                # Extract title and date from JSON
                title = item.get('title', '')
                date_str = item.get('date', item.get('published', item.get('published_dt', '')))

                # Use title as text if summary is empty
                text = item.get('summary', '').strip()
                if not text:
                    text = title

                news_list.append({
                    'date': date_str,
                    'title': title,
                    'text': text
                })

            news_df = pd.DataFrame(news_list)
            print(f"   ✅ Loaded {len(news_df)} news articles from JSON")

        # Standardize date format
        if 'date' in news_df.columns:
            news_df['date'] = pd.to_datetime(news_df['date'], errors='coerce')
            # Remove timezone info if present
            news_df['date'] = news_df['date'].dt.tz_localize(None)

        # Ensure we have the required columns
        if 'title' not in news_df.columns:
            news_df['title'] = ''
        if 'text' not in news_df.columns:
            news_df['text'] = news_df['title']  # Use title as text if no text column

        print(f"   Columns: {list(news_df.columns)}")
        return news_df

    except Exception as e:
        print(f"   ❌ ERROR loading news data: {str(e)}")
        print("   Creating synthetic news data instead...")
        return create_synthetic_news(pd.DataFrame())


def merge_stock_news(stock_df: pd.DataFrame,
                    data_path: str = 'data') -> pd.DataFrame:
    """
    Merge stock and news data by date

    Args:
        stock_df: Stock price DataFrame
        data_path: Path to data directory

    Returns:
        Merged DataFrame
    """
    print("🔗 Merging stock and news data...")

    # Load news data
    news_df = load_news_data(data_path)

    # If no news data was loaded, create synthetic news based on stock dates
    if news_df is None or len(news_df) == 0:
        news_df = create_synthetic_news(stock_df)

    stock_df = stock_df.copy()
    news_df = news_df.copy()

    # Handle timezone-aware dates
    stock_df['date'] = pd.to_datetime(stock_df['date'], utc=True).dt.tz_localize(None)
    news_df['date'] = pd.to_datetime(news_df['date'], utc=True, errors='coerce').dt.tz_localize(None)

    # Convert to date only
    stock_df['date'] = stock_df['date'].dt.date
    news_df['date'] = news_df['date'].dt.date

    # Convert back to datetime
    stock_df['date'] = pd.to_datetime(stock_df['date'])
    news_df['date'] = pd.to_datetime(news_df['date'])

    # Aggregate news by date
    news_agg = news_df.groupby('date').agg({
        'title': lambda x: ' | '.join(x.astype(str)),
        'text': lambda x: ' '.join(x.astype(str))
    }).reset_index()

    # Merge
    merged_df = stock_df.merge(news_agg, on='date', how='left')

    # Fill missing news
    merged_df['title'] = merged_df['title'].fillna('')
    merged_df['text'] = merged_df['text'].fillna('')
    merged_df['news_combined'] = merged_df['title'] + ' ' + merged_df['text']

    rows_with_news = (merged_df['news_combined'].str.len() > 0).sum()

    print(f"   ✅ Merged data: {len(merged_df)} rows")
    print(f"   Rows with news: {rows_with_news} ({rows_with_news/len(merged_df):.1%})")

    return merged_df


def create_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create target labels for prediction, filtering for significant moves.
    """
    print("🎯 Creating target labels...")

    df = df.copy()

    # Define a significance threshold (e.g., 0.5%)
    THRESHOLD = 0.005  # You can tune this (0.005, 0.01, etc.)

    # Next day return
    df['next_return'] = df['close'].shift(-1) / df['close'] - 1

    # Magnitude and volatility
    df['target_magnitude'] = np.abs(df['next_return'])
    df['target_volatility'] = df['returns'].rolling(window=5).std().shift(-1)
    
    # --- START NEW LABEL LOGIC ---

    # Create target based on the threshold
    def assign_label(ret):
        if ret > THRESHOLD:
            return 1  # Significant Up
        elif ret < -THRESHOLD:
            return 0  # Significant Down
        else:
            return np.nan  # Noise (will be dropped)

    df['target'] = df['next_return'].apply(assign_label)

    # Remove last row (which has no next_return)
    df = df.iloc[:-1]

    # **IMPORTANT**: Drop all rows that were 'noise'
    rows_before = len(df)
    df = df.dropna(subset=['target'])
    rows_after = len(df)
    
    print(f"   Filtered for significant moves (>{THRESHOLD*100}%):")
    print(f"   Removed {rows_before - rows_after} 'noise' samples ({ (rows_before - rows_after) / rows_before:.1%})")
    print(f"   Remaining samples: {rows_after}")

    # Convert target to integer
    df['target'] = df['target'].astype(int)

    # --- END NEW LABEL LOGIC ---

    # Class distribution
    class_counts = df['target'].value_counts()
    print(f"\n   New Class distribution:")
    for label, count in class_counts.items():
        if len(df) > 0:
            print(f"     Class {label}: {count} ({count/len(df):.1%})")

    return df
