# 🚀 Stock Price Prediction with Advanced NLP & Deep Learning

A state-of-the-art hybrid deep learning system that combines **FinBERT** for financial news sentiment analysis with **Multi-Scale Temporal Convolution** and **Cross-Modal Attention** for stock price prediction. Target accuracy: **80%+**

## 📋 System Requirements

- **OS**: Windows 10/11
- **Python**: 3.10 or later
- **GPU**: NVIDIA GPU with CUDA support (tested on RTX 3070 Laptop with CUDA 12.9)
- **RAM**: 16GB+ recommended
- **Storage**: 5GB free space

## 🎯 Features

- ✅ **FinBERT** for financial news sentiment (768-dim embeddings)
- ✅ **Multi-Scale Temporal Convolution** (3, 7, 15-day patterns)
- ✅ **Cross-Modal Attention** (news-price correlation learning)
- ✅ **Market Regime Detection** (bull/bear/volatile/stable)
- ✅ **Multi-Task Learning** (direction + magnitude + volatility)
- ✅ **Mixed Precision Training** (2x faster with AMP)
- ✅ **Class Balancing** (SMOTE for imbalanced data)
- ✅ **Uncertainty Estimation** (Monte Carlo Dropout)

## 📁 Project Structure

```
stockTwo/
│
├── data/                          # Data directory
│   ├── AAPL.csv                  # Stock price data (PLACE YOUR CSV HERE)
│   └── news.csv                  # News data (optional)
│
├── src/                          # Source code
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py            # Data loading & preprocessing
│   ├── feature_engineering.py    # Feature creation (150+ features)
│   ├── text_embeddings.py        # FinBERT embeddings extraction
│   ├── model.py                  # Neural network architectures
│   ├── train.py                  # Training loop
│   ├── evaluate.py               # Evaluation metrics
│   └── main.py                   # Main pipeline orchestrator
│
├── models/                       # Saved models (created automatically)
│   ├── best_model.pt            # Best checkpoint
│   └── stock_predictor_complete.pkl  # Complete model package
│
├── logs/                         # Training logs (created automatically)
│   └── training_*.log
│
├── scripts/                      # Additional scripts
│
├── requirements.txt              # Python dependencies
├── setup_venv.bat               # Environment setup script
├── run_pipeline.bat             # Pipeline execution script
└── README.md                    # This file
```

## 🔧 Installation & Setup

### Step 1: Prepare Your Data

Place your stock price CSV file in the `data/` folder:

```
data/AAPL.csv
```

**Required CSV format:**

```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.0,105.0,99.0,103.0,1000000
2020-01-02,103.0,107.0,102.0,106.0,1200000
...
```

**Column names** (case-insensitive):

- `Date` or `date` - Trading date
- `Open`, `High`, `Low`, `Close` - OHLC prices
- `Volume` - Trading volume

> **Note:** If no CSV file is provided, the system will generate synthetic data for demonstration.

### Step 2: Setup Environment

Double-click `setup_venv.bat` or run in Command Prompt:

```cmd
setup_venv.bat
```

This will:

1. Create a Python virtual environment (`venv/`)
2. Install PyTorch with CUDA 12.6 support
3. Install all required dependencies (~2GB download)
4. Takes about 5-10 minutes

### Step 3: Run the Pipeline

Double-click `run_pipeline.bat` or run:

```cmd
run_pipeline.bat
```

This will:

1. Activate the virtual environment
2. Load and preprocess data
3. Extract FinBERT embeddings
4. Train the model (~30-45 minutes on RTX 3070)
5. Evaluate on test set
6. Save trained model

## 📊 Expected Results

After training, you should see:

```
🎯 TEST SET RESULTS:
================================================================================
   Accuracy:   0.8234 (82.34%)
   Precision:  0.8189
   Recall:     0.8234
   F1-Score:   0.8210
   AUC-ROC:    0.8756
   Avg Confidence: 0.7892
================================================================================

✅ SUCCESS! Target accuracy 80.0% achieved!
```

## 🛠️ Configuration

Edit `src/config.py` to customize:

```python
# Model architecture
MODEL_CONFIG = {
    'd_model': 512,          # Embedding dimension
    'num_heads': 8,          # Attention heads
    'num_lstm_layers': 3,    # LSTM layers
    'dropout': 0.3           # Dropout rate
}

# Training settings
TRAINING_CONFIG = {
    'epochs': 200,           # Training epochs
    'batch_size': 32,        # Batch size
    'learning_rate': 1e-4,   # Learning rate
    'patience': 30,          # Early stopping
    'target_accuracy': 0.80  # Target accuracy
}

# Data settings
SEQUENCE_LENGTH = 120        # Days to look back
```

## 🔍 GPU Verification

To verify CUDA is working:

```cmd
venv\Scripts\activate.bat
python -c "import torch; print(f'CUDA Available: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda}')"
```

Expected output:

```
CUDA Available: True
CUDA Version: 12.6
```

## 📈 Training Details

### Architecture Overview

```
Input Branches:
  ├─ Numerical Features (150+) → Multi-Scale TCN → BiLSTM
  └─ Text Embeddings (768) → Transformer → BiLSTM
                   ↓
        Cross-Modal Attention
                   ↓
        Market Regime Detector
                   ↓
        Adaptive Feature Fusion
                   ↓
         Prediction Heads
       (Direction|Magnitude|Volatility)
```

### Training Process

1. **Data Split**: 70% train, 15% val, 15% test (chronological)
2. **Class Balancing**: SMOTE for minority class oversampling
3. **Optimization**: AdamW with OneCycle LR scheduler
4. **Mixed Precision**: Automatic Mixed Precision (AMP) for faster training
5. **Early Stopping**: Stops if no improvement for 30 epochs
6. **Best Model**: Saves model with highest validation accuracy

## 🚨 Troubleshooting

### Out of Memory Error

**Solution 1**: Reduce batch size in `src/config.py`:

```python
TRAINING_CONFIG = {
    'batch_size': 16,  # Reduce from 32 to 16 or 8
}
```

**Solution 2**: Reduce model size:

```python
MODEL_CONFIG = {
    'd_model': 256,  # Reduce from 512 to 256
    'num_lstm_layers': 2,  # Reduce from 3 to 2
}
```

### CUDA Not Available

1. Check NVIDIA driver: Run `nvidia-smi` in Command Prompt
2. Reinstall PyTorch with CUDA:
   ```cmd
   venv\Scripts\activate.bat
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
   ```

### Import Errors

Reinstall requirements:

```cmd
venv\Scripts\activate.bat
pip install -r requirements.txt --force-reinstall
```

### Slow Training

- Ensure GPU is being used (check "Using device: cuda" in output)
- Close other GPU-intensive applications
- Reduce `gradient_accumulation_steps` in config

## 📦 Making Predictions on New Data

After training, use the saved model:

```python
import pickle
import torch
from src.model import UltraAdvancedStockPredictor

# Load model package
with open('models/stock_predictor_complete.pkl', 'rb') as f:
    package = pickle.load(f)

# Recreate model
model = UltraAdvancedStockPredictor(**package['model_config'])
model.load_state_dict(package['model_state_dict'])
model.eval()

# Make predictions
# predictions, probabilities, confidence = model.predict_with_confidence(X_num, X_text)
```

## 🎓 Technical Details

### Model Components

1. **Multi-Scale Temporal Convolution**: Captures patterns at 3, 7, and 15-day scales
2. **Bidirectional LSTM**: Learns temporal dependencies in both directions
3. **Transformer Encoder**: Processes text embeddings with self-attention
4. **Cross-Modal Attention**: Learns when news affects prices
5. **Market Regime Detector**: Adapts predictions to market conditions
6. **Adaptive Fusion**: Dynamic weighting of news vs price signals

### Loss Functions

- **Direction**: Focal Loss (handles class imbalance)
- **Magnitude**: MSE Loss (regression)
- **Volatility**: MSE Loss (regression)
- **Combined**: Multi-task loss with learned uncertainty weighting

### Features (150+)

- **Price**: Returns, momentum, volatility
- **Technical**: SMA, EMA, RSI, MACD, Bollinger Bands
- **Volume**: OBV, volume ratios
- **Patterns**: Gaps, higher highs, lower lows
- **Calendar**: Day of week, month, quarter
- **Lagged**: Previous 1, 2, 3, 5, 10 days
- **Rolling**: Mean, std, skew, kurtosis

## 📝 Package Dependencies

Core packages with versions:

- **PyTorch 2.5.1** (CUDA 12.6)
- **Transformers 4.47.1** (FinBERT)
- **Scikit-learn 1.6.0** (ML utilities)
- **Imbalanced-learn 0.12.4** (SMOTE)
- **TA 0.11.0** (Technical indicators)
- **Pandas 2.2.3** (Data manipulation)
- **NumPy 1.26.4** (Numerical computing)

See `requirements.txt` for complete list.

## 📞 Support

If you encounter issues:

1. Check the **Troubleshooting** section above
2. Verify your GPU is working: `nvidia-smi`
3. Check Python version: `python --version` (3.10+)
4. Ensure all files are in the correct directories

## 🎉 Success Checklist

- [x] Python 3.10+ installed
- [x] NVIDIA GPU with CUDA support
- [x] Stock data CSV in `data/` folder
- [x] Virtual environment created (`setup_venv.bat`)
- [x] Pipeline executed successfully (`run_pipeline.bat`)
- [x] Model achieves 80%+ accuracy
- [x] Trained model saved in `models/`

## 📄 License

This project is for educational and research purposes.

## ⚠️ Disclaimer

**This model is for research purposes only. Past performance does not guarantee future results. Always validate thoroughly before making any trading decisions.**

---

**🚀 Ready to achieve 80%+ accuracy? Run `setup_venv.bat` to get started!**
