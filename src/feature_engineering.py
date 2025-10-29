"""
Feature engineering module
"""
import pandas as pd
import numpy as np
import ta
from typing import Tuple
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.preprocessing import StandardScaler, RobustScaler
from imblearn.over_sampling import SMOTE


def create_advanced_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create comprehensive financial features

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with 150+ features
    """
    print("⚙️ Creating advanced features...")

    features_df = df.copy()

    # 1. BASIC PRICE FEATURES
    features_df['returns'] = features_df['close'].pct_change()
    features_df['log_returns'] = np.log(features_df['close'] / features_df['close'].shift(1))
    features_df['high_low_range'] = (features_df['high'] - features_df['low']) / features_df['close']
    features_df['open_close_range'] = (features_df['close'] - features_df['open']) / features_df['open']

    # 2. TECHNICAL INDICATORS
    features_df['sma_5'] = ta.trend.sma_indicator(features_df['close'], window=5)
    features_df['sma_10'] = ta.trend.sma_indicator(features_df['close'], window=10)
    features_df['sma_20'] = ta.trend.sma_indicator(features_df['close'], window=20)
    features_df['sma_50'] = ta.trend.sma_indicator(features_df['close'], window=50)

    features_df['ema_5'] = ta.trend.ema_indicator(features_df['close'], window=5)
    features_df['ema_10'] = ta.trend.ema_indicator(features_df['close'], window=10)
    features_df['ema_20'] = ta.trend.ema_indicator(features_df['close'], window=20)

    # MACD
    macd = ta.trend.MACD(features_df['close'])
    features_df['macd'] = macd.macd()
    features_df['macd_signal'] = macd.macd_signal()
    features_df['macd_diff'] = macd.macd_diff()

    # RSI
    features_df['rsi_14'] = ta.momentum.rsi(features_df['close'], window=14)
    features_df['rsi_7'] = ta.momentum.rsi(features_df['close'], window=7)

    # Bollinger Bands
    bollinger = ta.volatility.BollingerBands(features_df['close'])
    features_df['bb_high'] = bollinger.bollinger_hband()
    features_df['bb_low'] = bollinger.bollinger_lband()
    features_df['bb_mid'] = bollinger.bollinger_mavg()
    features_df['bb_width'] = (features_df['bb_high'] - features_df['bb_low']) / features_df['bb_mid']
    features_df['bb_position'] = (features_df['close'] - features_df['bb_low']) / (features_df['bb_high'] - features_df['bb_low'])

    # 3. VOLUME INDICATORS
    features_df['volume_change'] = features_df['volume'].pct_change()
    features_df['volume_ma_5'] = features_df['volume'].rolling(window=5).mean()
    features_df['volume_ma_20'] = features_df['volume'].rolling(window=20).mean()
    features_df['volume_ratio'] = features_df['volume'] / features_df['volume_ma_20']
    features_df['obv'] = ta.volume.on_balance_volume(features_df['close'], features_df['volume'])

    # 4. VOLATILITY FEATURES
    features_df['volatility_5'] = features_df['returns'].rolling(window=5).std()
    features_df['volatility_10'] = features_df['returns'].rolling(window=10).std()
    features_df['volatility_20'] = features_df['returns'].rolling(window=20).std()
    features_df['atr_14'] = ta.volatility.average_true_range(
        features_df['high'], features_df['low'], features_df['close'], window=14
    )

    # 5. MOMENTUM FEATURES
    features_df['momentum_5'] = features_df['close'] - features_df['close'].shift(5)
    features_df['momentum_10'] = features_df['close'] - features_df['close'].shift(10)
    features_df['momentum_20'] = features_df['close'] - features_df['close'].shift(20)
    features_df['roc_5'] = ta.momentum.roc(features_df['close'], window=5)
    features_df['roc_10'] = ta.momentum.roc(features_df['close'], window=10)

    # 6. PATTERN FEATURES
    features_df['higher_high'] = (features_df['high'] > features_df['high'].shift(1)).astype(int)
    features_df['lower_low'] = (features_df['low'] < features_df['low'].shift(1)).astype(int)
    features_df['gap_up'] = ((features_df['open'] > features_df['close'].shift(1)) &
                             (features_df['low'] > features_df['high'].shift(1))).astype(int)
    features_df['gap_down'] = ((features_df['open'] < features_df['close'].shift(1)) &
                               (features_df['high'] < features_df['low'].shift(1))).astype(int)

    # 7. MICROSTRUCTURE FEATURES
    features_df['amihud'] = np.abs(features_df['returns']) / (features_df['volume'] * features_df['close'] + 1e-10)
    features_df['roll_spread'] = 2 * np.sqrt(-features_df['returns'].rolling(window=5).cov(features_df['returns'].shift(1)))

    # 8. CALENDAR FEATURES
    features_df['day_of_week'] = pd.to_datetime(features_df['date']).dt.dayofweek
    features_df['day_of_month'] = pd.to_datetime(features_df['date']).dt.day
    features_df['month'] = pd.to_datetime(features_df['date']).dt.month
    features_df['quarter'] = pd.to_datetime(features_df['date']).dt.quarter

    # 9. LAGGED FEATURES
    for lag in [1, 2, 3, 5, 10]:
        features_df[f'close_lag_{lag}'] = features_df['close'].shift(lag)
        features_df[f'returns_lag_{lag}'] = features_df['returns'].shift(lag)
        features_df[f'volume_lag_{lag}'] = features_df['volume'].shift(lag)

    # 10. ROLLING STATISTICS
    for window in [5, 10, 20]:
        features_df[f'returns_mean_{window}'] = features_df['returns'].rolling(window=window).mean()
        features_df[f'returns_std_{window}'] = features_df['returns'].rolling(window=window).std()
        features_df[f'returns_skew_{window}'] = features_df['returns'].rolling(window=window).skew()
        features_df[f'returns_kurt_{window}'] = features_df['returns'].rolling(window=window).kurt()

    print(f"   Created {len(features_df.columns)} total columns")

    # Drop NaN rows
    features_df = features_df.dropna()
    print(f"   After dropping NaN: {len(features_df)} rows")

    return features_df


def prepare_data(features_df: pd.DataFrame,
                news_embeddings: np.ndarray,
                sequence_length: int = 120) -> dict:
    """
    Prepare data for deep learning model

    Args:
        features_df: DataFrame with features
        news_embeddings: FinBERT embeddings
        sequence_length: Number of days in sequence

    Returns:
        Dictionary with train/val/test splits
    """
    print("🔧 Preparing data for modeling...")

    # Define feature columns
    exclude_cols = ['date', 'news_combined', 'title', 'text', 'Symbol',
                   'target', 'next_return', 'target_magnitude', 'target_volatility']
    feature_cols = [col for col in features_df.columns if col not in exclude_cols]
    feature_cols = [col for col in feature_cols if features_df[col].dtype in ['int64', 'float64']]

    # Extract features and labels
    X_numerical = features_df[feature_cols].values
    y_direction = features_df['target'].values
    y_magnitude = features_df['target_magnitude'].values
    y_volatility = features_df['target_volatility'].values

    print(f"   Numerical features: {X_numerical.shape}")
    print(f"   Text embeddings: {news_embeddings.shape}")

    # Normalize features
    scaler = RobustScaler()
    X_numerical_scaled = scaler.fit_transform(X_numerical)

    text_scaler = StandardScaler()
    X_text_scaled = text_scaler.fit_transform(news_embeddings)

    print("   ✓ Features normalized")

    # Create sequences
    print(f"   Creating sequences (length={sequence_length})...")
    X_num_seq, X_text_seq, y_dir_seq, y_mag_seq, y_vol_seq = [], [], [], [], []

    for i in range(sequence_length, len(X_numerical_scaled)):
        X_num_seq.append(X_numerical_scaled[i-sequence_length:i])
        X_text_seq.append(X_text_scaled[i-sequence_length:i])
        y_dir_seq.append(y_direction[i])
        y_mag_seq.append(y_magnitude[i])
        y_vol_seq.append(y_volatility[i])

    X_num_seq = np.array(X_num_seq)
    X_text_seq = np.array(X_text_seq)
    y_dir_seq = np.array(y_dir_seq)
    y_mag_seq = np.array(y_mag_seq)
    y_vol_seq = np.array(y_vol_seq)

    print(f"   ✓ Created {len(X_num_seq)} sequences")

    # Chronological split
    n_samples = len(X_num_seq)
    train_end = int(n_samples * 0.7)
    val_end = int(n_samples * 0.85)

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
    """Apply SMOTE for class balancing"""
    print("⚖️ Applying SMOTE...")

    n_samples, n_timesteps, n_num_features = X_num.shape
    _, _, n_text_features = X_text.shape

    # Flatten
    X_num_flat = X_num.reshape(n_samples, -1)
    X_text_flat = X_text.reshape(n_samples, -1)
    X_combined = np.hstack([X_num_flat, X_text_flat])

    # Apply SMOTE
    smote = SMOTE(random_state=42, k_neighbors=5)
    X_resampled, y_resampled = smote.fit_resample(X_combined, y)

    # Reshape
    n_num_total = n_timesteps * n_num_features
    X_num_resampled = X_resampled[:, :n_num_total].reshape(-1, n_timesteps, n_num_features)
    X_text_resampled = X_resampled[:, n_num_total:].reshape(-1, n_timesteps, n_text_features)

    print(f"   Generated {len(y_resampled) - len(y)} synthetic samples")

    return X_num_resampled, X_text_resampled, y_resampled


def create_dataloaders(data_splits: dict, batch_size: int = 32) -> dict:
    """Create PyTorch DataLoaders"""
    print(f"📦 Creating DataLoaders (batch_size={batch_size})...")

    dataloaders = {}

    for split_name in ['train', 'val', 'test']:
        X_num = torch.FloatTensor(data_splits[split_name]['X_num'])
        X_text = torch.FloatTensor(data_splits[split_name]['X_text'])
        y_dir = torch.LongTensor(data_splits[split_name]['y_dir'])
        y_mag = torch.FloatTensor(data_splits[split_name]['y_mag'])
        y_vol = torch.FloatTensor(data_splits[split_name]['y_vol'])

        dataset = TensorDataset(X_num, X_text, y_dir, y_mag, y_vol)
        shuffle = (split_name == 'train')

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=0,
            pin_memory=True if torch.cuda.is_available() else False
        )

        dataloaders[split_name] = dataloader
        print(f"   {split_name.capitalize()}: {len(dataset)} samples, {len(dataloader)} batches")

    return dataloaders
