"""
Configuration file for stock prediction model

This module contains all hyperparameters, paths, and configuration settings
for the stock prediction system. Centralizing configuration here makes it
easy to tune the model without modifying code in multiple files.
"""
import torch
import os

# ============================================================================
# PATHS
# ============================================================================
# Define all directory paths relative to the project root for portability
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'models')
LOGS_DIR = os.path.join(BASE_DIR, 'logs')

# Data files (place your CSV files in the 'data' folder)
STOCK_DATA_FILE = os.path.join(DATA_DIR, 'AAPL.csv')

# Model checkpoint paths
BEST_MODEL_PATH = os.path.join(MODELS_DIR, 'best_model.pt')
FULL_PACKAGE_PATH = os.path.join(MODELS_DIR, 'stock_predictor_complete.pkl')

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================
# Automatically detect and use GPU if available, otherwise fall back to CPU
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# RANDOM SEEDS (for reproducibility)
# ============================================================================
# Fixed seed ensures reproducible results across runs
SEED = 42

# ============================================================================
# DATA PROCESSING
# ============================================================================
# Number of past days to use as input for prediction (sliding window)
SEQUENCE_LENGTH = 120  # 120 trading days ≈ 6 months of data
# Validation and test set sizes (as fraction of total data)
TEST_SIZE = 0.15   # 15% of data for final testing
VAL_SIZE = 0.15    # 15% of data for validation during training

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
# Neural network architecture hyperparameters
MODEL_CONFIG = {
    'd_model': 128,          # Hidden dimension size (reduced from 512 to prevent overfitting)
    'num_heads': 4,          # Number of attention heads in multi-head attention (reduced from 8)
    'num_lstm_layers': 2,    # Number of LSTM layers for sequential processing (reduced from 3)
    'num_transformer_layers': 2,  # Number of transformer encoder layers (reduced from 4)
    'num_regimes': 4,        # Number of market regimes to detect (bullish/bearish/volatile/stable)
    'dropout': 0.4           # Dropout rate to prevent overfitting (increased from 0.3)
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
# Training loop hyperparameters
TRAINING_CONFIG = {
    'epochs': 200,                      # Maximum number of training epochs
    'batch_size': 32,                   # Number of samples per batch
    'learning_rate': 1e-4,              # Initial learning rate for AdamW optimizer
    'weight_decay': 1e-5,               # L2 regularization strength
    'gradient_accumulation_steps': 4,   # Accumulate gradients over N batches (effective batch size = 32*4 = 128)
    'max_grad_norm': 1.0,               # Gradient clipping threshold to prevent exploding gradients
    'patience': 30,                     # Early stopping: stop if no improvement after N epochs
    'target_accuracy': 0.80,            # Training stops early if this accuracy is achieved
    'warmup_ratio': 0.3,                # Fraction of training for learning rate warmup (for OneCycle scheduler)
}

# ============================================================================
# FINBERT CONFIGURATION
# ============================================================================
# FinBERT is a BERT model fine-tuned on financial text for sentiment analysis
FINBERT_MODEL_NAME = 'yiyanghkust/finbert-tone'  # Pre-trained FinBERT model
FALLBACK_MODEL_NAME = 'distilbert-base-uncased'  # Fallback if FinBERT unavailable
MAX_TEXT_LENGTH = 512           # Maximum token length for BERT input
TEXT_BATCH_SIZE = 16            # Batch size for text embedding extraction

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 10   # Print training progress every N batches
SAVE_INTERVAL = 5   # Save model checkpoint every N epochs

# ============================================================================
# VISUALIZATION
# ============================================================================
PLOT_DPI = 300            # Resolution for saved plots
FIGURE_SIZE = (15, 10)    # Default figure size (width, height) in inches
