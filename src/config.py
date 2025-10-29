"""
Configuration file for stock prediction model
"""
import torch
import os

# ============================================================================
# PATHS
# ============================================================================
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
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# RANDOM SEEDS (for reproducibility)
# ============================================================================
SEED = 42

# ============================================================================
# DATA PROCESSING
# ============================================================================
SEQUENCE_LENGTH = 120  # Number of days to look back
TEST_SIZE = 0.15
VAL_SIZE = 0.15

# ============================================================================
# MODEL ARCHITECTURE
# ============================================================================
MODEL_CONFIG = {
    'd_model': 128,          # Was 512
    'num_heads': 4,            # Was 8
    'num_lstm_layers': 2,      # Was 3
    'num_transformer_layers': 2, # Was 4
    'num_regimes': 4,
    'dropout': 0.4           # INCREASE dropout to fight overfitting
}

# ============================================================================
# TRAINING CONFIGURATION
# ============================================================================
TRAINING_CONFIG = {
    'epochs': 200,
    'batch_size': 32,
    'learning_rate': 5e-4,
    'weight_decay': 1e-5,
    'gradient_accumulation_steps': 4,
    'max_grad_norm': 1.0,
    'patience': 50,  # Early stopping patience
    'target_accuracy': 0.80,
    'warmup_ratio': 0.3,  # For OneCycle scheduler
}

# ============================================================================
# FINBERT CONFIGURATION
# ============================================================================
FINBERT_MODEL_NAME = 'yiyanghkust/finbert-tone'
FALLBACK_MODEL_NAME = 'distilbert-base-uncased'
MAX_TEXT_LENGTH = 512
TEXT_BATCH_SIZE = 16

# ============================================================================
# LOGGING
# ============================================================================
LOG_INTERVAL = 10  # Log every N batches
SAVE_INTERVAL = 5   # Save checkpoint every N epochs

# ============================================================================
# VISUALIZATION
# ============================================================================
PLOT_DPI = 300
FIGURE_SIZE = (15, 10)
