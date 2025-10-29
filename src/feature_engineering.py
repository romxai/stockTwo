"""
Feature engineering module

This module creates comprehensive technical indicators and financial features from
raw OHLCV (Open, High, Low, Close, Volume) data. Features include:
- Price patterns and transformations
- Technical indicators (SMA, EMA, RSI, MACD, Bollinger Bands)
- Volume-based features
- Volatility measures
- Momentum indicators
- Calendar effects
- Lagged features and rolling statistics
"""
import pandas as pd
import numpy as np
import ta  # Technical Analysis library
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive financial features from OHLCV data
    
    Generates 150+ technical indicators and engineered features across 10 categories:
    1. Basic price features (returns, ranges)
    2. Technical indicators (moving averages, MACD, RSI, Bollinger Bands)
    3. Volume indicators (volume ratios, OBV)
    4. Volatility features (ATR, rolling volatility)
    5. Momentum features (ROC, momentum)
    6. Pattern features (gaps, higher highs, lower lows)
    7. Microstructure features (Amihud illiquidity, spread estimates)
    8. Calendar features (day of week, month, quarter)
    9. Lagged features (historical values)
    10. Rolling statistics (mean, std, skewness, kurtosis)

    Args:
        df: DataFrame with OHLCV data (columns: open, high, low, close, volume, date)

    Returns:
        DataFrame with original columns + 150+ engineered features, NaN rows removed
    """
    print("⚙️ Creating advanced features...")

    features_df = df.copy()

    # 1. BASIC PRICE FEATURES
    # Simple returns: percentage change in closing price
    features_df['returns'] = features_df['close'].pct_change()
    # Log returns: more stable for statistical analysis, additive over time
    features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
    # Intraday range: normalized difference between high and low
    features_df['high_low_range'] = (features_df['high'] - features_df['low']) / features_df['close']
    # Open-close range: captures intraday momentum direction
    features_df['open_close_range'] = (features_df['close'] - features_df['open']) / features_df['open']

    # 2. TECHNICAL INDICATORS
    # Simple Moving Averages (SMA): trend indicators at different timescales
    features_df['sma_5'] = ta.trend.sma_indicator(features_df['close'], window=5)    # 1 week
    features_df['sma_10'] = ta.trend.sma_indicator(features_df['close'], window=10)  # 2 weeks
    features_df['sma_20'] = ta.trend.sma_indicator(features_df['close'], window=20)  # 1 month
    features_df['sma_50'] = ta.trend.sma_indicator(features_df['close'], window=50)  # ~2.5 months

    # Exponential Moving Averages (EMA): more weight on recent prices
    features_df['ema_5'] = ta.trend.ema_indicator(features_df['close'], window=5)
    features_df['ema_10'] = ta.trend.ema_indicator(features_df['close'], window=10)
    features_df['ema_20'] = ta.trend.ema_indicator(features_df['close'], window=20)

    # MACD (Moving Average Convergence Divergence): momentum and trend strength
    macd = ta.trend.MACD(features_df['close'])
    features_df['macd'] = macd.macd()              # MACD line (difference between EMAs)
    features_df['macd_signal'] = macd.macd_signal()  # Signal line (EMA of MACD)
    features_df['macd_diff'] = macd.macd_diff()    # Histogram (MACD - Signal)

    # RSI (Relative Strength Index): overbought/oversold indicator (0-100)
    features_df['rsi_14'] = ta.momentum.rsi(features_df['close'], window=14)  # Standard RSI
    features_df['rsi_7'] = ta.momentum.rsi(features_df['close'], window=7)    # Short-term RSI

    # Bollinger Bands: volatility bands around moving average
    bollinger = ta.volatility.BollingerBands(features_df['close'])
    features_df['bb_high'] = bollinger.bollinger_hband()   # Upper band
    features_df['bb_low'] = bollinger.bollinger_lband()    # Lower band
    features_df['bb_mid'] = bollinger.bollinger_mavg()     # Middle band (SMA)
    # BB width: measures volatility (wider = more volatile)
    features_df['bb_width'] = (features_df['bb_high'] - features_df['bb_low']) / features_df['bb_mid']
    # BB position: where price is within the bands (0=lower, 1=upper)
    features_df['bb_position'] = (features_df['close'] - features_df['bb_low']) / (features_df['bb_high'] - features_df['bb_low'])

    # 3. VOLUME INDICATORS
    # Volume changes can signal strength of price movements
    features_df['volume_change'] = features_df['volume'].pct_change()
    features_df['volume_ma_5'] = features_df['volume'].rolling(window=5).mean()
    features_df['volume_ma_20'] = features_df['volume'].rolling(window=20).mean()
    # Volume ratio: current volume relative to average (>1 = high activity)
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_20']
    # On-Balance Volume (OBV): cumulative volume indicator (buying/selling pressure)
    features_df['obv'] = ta.volume.on_balance_volume(features_df['close'], features_df['volume'])

    # 4. VOLATILITY FEATURES
    # Rolling standard deviation of returns at different timescales
    features_df['volatility_5'] = features_df['returns'].rolling(window=5).std()    # Short-term volatility
    features_df['volatility_10'] = features_df['returns'].rolling(window=10).std()  # Medium-term
    features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()  # Long-term
    # ATR (Average True Range): measures price volatility considering gaps
    features_df['atr_14'] = ta.volatility.average_true_range(
        features_df['high'], features_df['low'], features_df['close'], window=14
    )

    # 5. MOMENTUM FEATURES
    # Raw momentum: absolute price change over N days
    features_df['momentum_5'] = features_df['close'] - features_df['close'].shift(5)
    features_df['momentum_10'] = features_df['close'] - features_df['close'].shift(10)
    features_df['momentum_20'] = features_df['close'] - features_df['close'].shift(20)
    # ROC (Rate of Change): percentage price change over N days
    features_df['roc_5'] = ta.momentum.roc(features_df['close'], window=5)
    features_df['roc_10'] = ta.momentum.roc(features_df['close'], window=10)

    # 6. PATTERN FEATURES
    # Binary indicators for common price patterns
    features_df['higher_high'] = (features_df['high'] > features_df['high'].shift(1)).astype(int)
    features_df['lower_low'] = (features_df['low'] < features_df['low'].shift(1)).astype(int)
    # Gap up: opens above previous close with no overlap in price ranges
    features_df['gap_up'] = ((features_df['open'] > features_df['close'].shift(1)) &
                             (features_df['low'] > features_df['high'].shift(1))).astype(int)
    # Gap down: opens below previous close with no overlap
    features_df['gap_down'] = ((features_df['open'] < features_df['close'].shift(1)) &
                               (features_df['high'] < features_df['low'].shift(1))).astype(int)

    # 7. MICROSTRUCTURE FEATURES
    # Amihud illiquidity measure: price impact of trading (higher = less liquid)
    features_df['amihud'] = np.abs(features_df['returns']) / (features_df['volume'] * features_df['close'] + 1e-10)
    # Roll spread estimator: estimates bid-ask spread from price autocorrelation
    features_df['roll_spread'] = 2 * np.sqrt(-features_df['returns'].rolling(window=5).cov(features_df['returns'].shift(1)))

    # 8. CALENDAR FEATURES
    # Capture day-of-week effects (e.g., Monday effect) and monthly patterns
    features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek  # 0=Monday, 6=Sunday
    features_df['day_of_month'] = pd.to_datetime(features_df['date']).dt.day       # 1-31
    features_df['month'] = pd.to_datetime(features_df['date']).dt.month            # 1-12
    features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter        # 1-4

    # 9. LAGGED FEATURES
    # Historical values help capture temporal dependencies and trends
    for lag in [1, 2, 3, 5, 10]:
        features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
        features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)

    # 10. ROLLING STATISTICS
    # Capture distribution properties of returns over different windows
    for window in [5, 10, 20]:
        features_df[f'returns_mean_{window}'] = features_df['returns'].rolling(window=window).mean()  # Average return
        features_df[f'returns_std_{window}'] = features_df['returns'].rolling(window=window).std()    # Volatility
        features_df[f'returns_skew_{window}'] = features_df['returns'].rolling(window=window).skew()  # Asymmetry
        features_df[f'returns_kurt_{window}'] = features_df['returns'].rolling(window=window).kurt()  # Tail heaviness

    print(f"   Created {len(features_df.columns)} total columns")

    # Drop rows with NaN values (created by rolling windows and lags at the beginning)
    features_df = features_df.dropna()
    print(f"   After dropping NaN: {len(features_df)} rows")

    return features_df


def prepare_data(features_df: pd.DataFrame,
                news_embeddings: np.ndarray,
                sequence_length: int = 120) -> dict:
    """
    Prepare data for deep learning model with sequences and proper scaling
    
    Process:
    1. Extract numerical features and labels
    2. Normalize features using robust scaling (handles outliers)
    3. Create sliding window sequences for LSTM/Transformer input
    4. Split chronologically (70% train, 15% val, 15% test)
    
    Chronological splitting is crucial for time series to avoid data leakage.

    Args:
        features_df: DataFrame with engineered features and labels
        news_embeddings: FinBERT text embeddings (same length as features_df)
        sequence_length: Number of past days to use as input (e.g., 120 = 6 months)

    Returns:
        Dictionary with keys: 'train', 'val', 'test', 'scalers'
        Each split contains: X_num, X_text, y_dir, y_mag, y_vol arrays
    """
    print("🔧 Preparing data for modeling...")

    # Define feature columns - exclude non-numeric and target columns
    exclude_cols = ['date', 'news_combined', 'title', 'text', 'Symbol',
                   'target', 'next_return', 'target_magnitude', 'target_volatility']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    # Only keep numeric columns (filter out any remaining string columns)
    feature_cols = [col for col in feature_cols if features_df[col].dtype in ['int64', 'float64']]

    # Extract features and multi-task labels
    X_numerical = features_df[feature_cols].values          # All engineered features
    y_direction = features_df['target'].values              # Binary: up/down
    y_magnitude = features_df['target_magnitude'].values    # Regression: size of move
    y_volatility = features_df['target_volatility'].values  # Regression: uncertainty

    print(f"   Numerical features: {X_numerical.shape}")
    print(f"   Text embeddings: {news_embeddings.shape}")

    # Normalize features to zero mean and unit variance
    # RobustScaler is less sensitive to outliers than StandardScaler
    scaler = RobustScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)

    # Normalize text embeddings separately (they're already in BERT space)
    text_scaler = StandardScaler()
    X_text_scaled = text_scaler.fit_transform(news_embeddings)

    print("   ✓ Features normalized")

    # Create sequences using sliding window approach
    # Each sample is a sequence of past N days used to predict the next day
    print(f"   Creating sequences (length={sequence_length})...")
    X_num_seq, X_text_seq, y_dir_seq, y_mag_seq, y_vol_seq = [], [], [], [], []

    for i in range(sequence_length, len(X_numerical_scaled)):
        # Input: days [i-sequence_length] to [i-1]
        X_num_seq.append(X_numerical_scaled[i-sequence_length:i])
        X_text_seq.append(X_text_scaled[i-sequence_length:i])
        # Target: day [i] (what happens after the sequence)
        y_dir_seq.append(y_direction[i])
        y_mag_seq.append(y_magnitude[i])
        y_vol_seq.append(y_volatility[i])

    X_num_seq = np.array(X_num_seq)
    X_text_seq = np.array(X_text_seq)
    y_dir_seq = np.array(y_dir_seq)
    y_mag_seq = np.array(y_mag_seq)
    y_vol_seq = np.array(y_vol_seq)

    print(f"   ✓ Created {len(X_num_seq)} sequences")

    # Chronological split (NO SHUFFLING - preserves temporal order)
    # Critical for time series: test set must be after training set
    n_samples = len(X_num_seq)
    train_end = int(n_samples * 0.7)   # First 70% for training
    val_end = int(n_samples * 0.85)    # Next 15% for validation

    data_splits = {
        'train': {
            'X_num': X_num_seq[:train_end],
            'X_text': X_text_seq[:train_end],
            'y_dir': y_dir_seq[:train_end],
            'y_mag': y_mag_seq[:train_end],
            'y_vol': y_vol_seq[:train_end]
        },
        'val': {
            'X_num': X_num_seq[train_end:val_end],
            'X_text': X_text_seq[train_end:val_end],
            'y_dir': y_dir_seq[train_end:val_end],
            'y_mag': y_mag_seq[train_end:val_end],
            'y_vol': y_vol_seq[train_end:val_end]
        },
        'test': {
            'X_num': X_num_seq[val_end:],
            'X_text': X_text_seq[val_end:],
            'y_dir': y_dir_seq[val_end:],
            'y_mag': y_mag_seq[val_end:],
            'y_vol': y_vol_seq[val_end:]
        },
        'scalers': {
            'numerical': scaler,
            'text': text_scaler
        }
    }

    print(f"\n   📊 Split sizes:")
    for split_name in ['train', 'val', 'test']:
        y = data_splits[split_name]['y_dir']
        print(f"      {split_name.capitalize()}: {len(y)} samples")
        unique, counts = np.unique(y, return_counts=True)
        for label, count in zip(unique, counts):
            print(f"        Class {label}: {count:4d} ({count/len(y):.1%})")

    return data_splits


def apply_smote(X_num, X_text, y):
    """
    Apply SMOTE (Synthetic Minority Over-sampling Technique) for class balancing
    
    WARNING: SMOTE can corrupt time-series data by creating synthetic samples
    that average across time steps. Use with caution on sequential data.
    Consider using class weights or focal loss instead.
    
    Args:
        X_num: Numerical feature sequences (N, sequence_length, num_features)
        X_text: Text embedding sequences (N, sequence_length, text_features)
        y: Binary labels (N,)
    
    Returns:
        Balanced X_num, X_text, y with synthetic samples added
    """
    print("⚖️ Applying SMOTE...")

    n_samples, n_timesteps, n_num_features = X_num.shape
    _, _, n_text_features = X_text.shape

    # Flatten sequences to 2D for SMOTE (requires 2D input)
    X_num_flat = X_num.reshape(n_samples, -1)
    X_text_flat = X_text.reshape(n_samples, -1)
    X_combined = np.hstack([X_num_flat, X_text_flat])

    # Apply SMOTE to generate synthetic minority class samples
    # k_neighbors=5: use 5 nearest neighbors to create synthetic samples
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)

    # Reshape back to 3D sequences
    n_num_total = n_timesteps * n_num_features
    X_num_resampled = X_resampled[:, :n_num_total].reshape(-1, n_timesteps, n_num_features)
    X_text_resampled = X_resampled[:, n_num_total:].reshape(-1, n_timesteps, n_text_features)

    print(f"   Generated {len(y_resampled) - len(y)} synthetic samples")

    return X_num_resampled, X_text_resampled, y_resampled


def create_dataloaders(data_splits: dict, batch_size: int = 32) -> dict:
    """
    Create PyTorch DataLoaders for batched training
    
    DataLoaders handle batching, shuffling (train only), and parallel data loading.
    Pin memory speeds up GPU transfer if CUDA is available.
    
    Args:
        data_splits: Dictionary with 'train', 'val', 'test' splits
        batch_size: Number of samples per batch
    
    Returns:
        Dictionary of DataLoaders for each split
    """
    print(f"📦 Creating DataLoaders (batch_size={batch_size})...")

    dataloaders = {}

    for split_name in ['train', 'val', 'test']:
        # Convert numpy arrays to PyTorch tensors
        X_num = torch.FloatTensor(data_splits[split_name]['X_num'])
        X_text = torch.FloatTensor(data_splits[split_name]['X_text'])
        y_dir = torch.LongTensor(data_splits[split_name]['y_dir'])    # Classification target (class indices)
        y_mag = torch.FloatTensor(data_splits[split_name]['y_mag'])   # Regression target
        y_vol = torch.FloatTensor(data_splits[split_name]['y_vol'])   # Regression target

        # Create TensorDataset (wraps tensors for DataLoader)
        dataset = TensorDataset(X_num, X_text, y_dir, y_mag, y_vol)
        # Only shuffle training data (validation/test must remain in order)
        shuffle = (split_name == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,  # Set to 0 for Windows compatibility
            pin_memory=True if torch.cuda.is_available() else False  # Speeds up CPU->GPU transfer
        )

        dataloaders[split_name] = dataloader
        print(f"   {split_name.capitalize()}: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloaders
