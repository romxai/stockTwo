# 📁 COMPLETE FILE STRUCTURE

## Overview

This document shows the complete file and folder structure of the stock prediction project.

```
d:\Users\Fiona\Desktop\projects\DLNLP\stockTwo\
│
├── 📂 data/                                    # Data directory
│   ├── README.md                              # Data format instructions
│   └── AAPL.csv                               # ← PLACE YOUR CSV HERE
│
├── 📂 src/                                     # Source code
│   ├── __init__.py                            # Package initialization
│   ├── config.py                              # Configuration settings
│   ├── data_loader.py                         # Data loading & preprocessing
│   ├── feature_engineering.py                 # Feature creation (150+)
│   ├── text_embeddings.py                     # FinBERT embeddings
│   ├── model.py                               # Neural network models
│   ├── train.py                               # Training loop
│   ├── evaluate.py                            # Evaluation metrics
│   └── main.py                                # Main pipeline
│
├── 📂 models/                                  # Saved models (auto-created)
│   ├── .gitkeep                               # Git placeholder
│   ├── best_model.pt                          # Best checkpoint (created)
│   └── stock_predictor_complete.pkl           # Full package (created)
│
├── 📂 logs/                                    # Training logs (auto-created)
│   └── .gitkeep                               # Git placeholder
│
├── 📂 scripts/                                 # Utility scripts
│   └── verify_installation.py                 # Installation verification
│
├── 📂 venv/                                    # Virtual environment (created by setup)
│   ├── Scripts/
│   │   ├── activate.bat
│   │   ├── python.exe
│   │   └── pip.exe
│   └── Lib/
│
├── 📄 requirements.txt                         # Python dependencies
├── 📄 setup_venv.bat                          # Environment setup script
├── 📄 run_pipeline.bat                        # Pipeline execution script
├── 📄 QUICK_START.bat                         # Interactive wizard
├── 📄 README.md                               # Complete documentation
├── 📄 PROJECT_SUMMARY.md                      # Quick reference guide
├── 📄 FILE_STRUCTURE.md                       # This file
├── 📄 .gitignore                              # Git ignore rules
└── 📄 Stock_Prediction_Advanced_NLP_DL.ipynb  # Original notebook
```

---

## 📂 Detailed File Descriptions

### **Root Directory Files**

| File                 | Purpose                               | Run?            |
| -------------------- | ------------------------------------- | --------------- |
| `requirements.txt`   | All Python dependencies with versions | No              |
| `setup_venv.bat`     | Creates venv and installs packages    | **Run FIRST**   |
| `run_pipeline.bat`   | Executes the full training pipeline   | **Run SECOND**  |
| `QUICK_START.bat`    | Interactive setup wizard              | **Or run this** |
| `README.md`          | Complete user documentation           | Read            |
| `PROJECT_SUMMARY.md` | Quick reference guide                 | Read            |
| `FILE_STRUCTURE.md`  | This file                             | Read            |
| `.gitignore`         | Git version control ignore rules      | No              |

---

### **data/ Folder**

**Purpose**: Store your stock price CSV files

| File        | Description                  | Required?            |
| ----------- | ---------------------------- | -------------------- |
| `README.md` | Instructions for data format | Documentation        |
| `AAPL.csv`  | Your stock data              | **YES - Place here** |
| `news.csv`  | News data (CSV format)       | **YES - Place here** |
| `news.json` | News data (JSON format)      | **YES - or use CSV** |

**CSV Format**:

```csv
Date,Open,High,Low,Close,Volume
2020-01-01,100.0,105.0,99.0,103.0,1000000
```

---

### **src/ Folder**

**Purpose**: All Python source code

| File                     | Lines | Purpose                      |
| ------------------------ | ----- | ---------------------------- |
| `__init__.py`            | 3     | Package initialization       |
| `config.py`              | 80    | All configuration settings   |
| `data_loader.py`         | 180   | Load & preprocess data       |
| `feature_engineering.py` | 320   | Create 150+ features         |
| `text_embeddings.py`     | 130   | Extract FinBERT embeddings   |
| `model.py`               | 420   | Neural network architectures |
| `train.py`               | 230   | Training loop & optimization |
| `evaluate.py`            | 150   | Evaluation & metrics         |
| `main.py`                | 250   | Main pipeline orchestrator   |

**Total**: ~1,763 lines of Python code

---

### **models/ Folder**

**Purpose**: Store trained models (created automatically)

| File                           | Size   | Created When              |
| ------------------------------ | ------ | ------------------------- |
| `.gitkeep`                     | -      | Initial (git placeholder) |
| `best_model.pt`                | ~70 MB | During training           |
| `stock_predictor_complete.pkl` | ~75 MB | After training            |

---

### **logs/ Folder**

**Purpose**: Store training logs and visualizations

Created automatically during training.

---

### **scripts/ Folder**

**Purpose**: Utility scripts

| File                     | Purpose                              |
| ------------------------ | ------------------------------------ |
| `verify_installation.py` | Check all dependencies are installed |

---

### **venv/ Folder**

**Purpose**: Python virtual environment (isolated dependencies)

- Created by `setup_venv.bat`
- Contains Python interpreter and all packages
- ~2-3 GB after installation
- **Do NOT commit to git** (in .gitignore)

---

## 🔄 Execution Flow

### **Initial Setup**

```
1. User places AAPL.csv in data/
2. User runs setup_venv.bat
   ├── Creates venv/
   ├── Installs PyTorch with CUDA 12.6
   └── Installs all dependencies
3. User runs run_pipeline.bat
```

### **Pipeline Execution**

```
run_pipeline.bat
  ├── Activates venv
  ├── Runs src/main.py
  │   ├── config.py → Load settings
  │   ├── data_loader.py → Load AAPL.csv
  │   ├── feature_engineering.py → Create features
  │   ├── text_embeddings.py → Extract embeddings
  │   ├── model.py → Build neural network
  │   ├── train.py → Train model (30-45 min)
  │   └── evaluate.py → Test performance
  └── Saves to models/
```

---

## 📦 Package Dependencies (requirements.txt)

### **Core Deep Learning**

```
torch==2.5.1+cu126              # PyTorch with CUDA 12.6
torchvision==0.20.1+cu126       # Vision utilities
torchaudio==2.5.1+cu126         # Audio utilities
```

### **NLP & Transformers**

```
transformers==4.47.1            # HuggingFace transformers
tokenizers==0.21.0              # Tokenization
huggingface-hub==0.27.0         # Model hub access
```

### **Data Processing**

```
numpy==1.26.4                   # Numerical computing
pandas==2.2.3                   # Data manipulation
scikit-learn==1.6.0             # ML utilities
imbalanced-learn==0.12.4        # SMOTE for balancing
```

### **Technical Analysis**

```
ta==0.11.0                      # Technical indicators
yfinance==0.2.49                # Stock data download
```

### **Visualization**

```
matplotlib==3.9.3               # Plotting
seaborn==0.13.2                 # Statistical viz
plotly==5.24.1                  # Interactive plots
```

---

## 💾 Storage Requirements

| Item                | Size        | Notes              |
| ------------------- | ----------- | ------------------ |
| Project files       | ~5 MB       | Source code & docs |
| Virtual environment | ~2-3 GB     | After setup        |
| PyTorch models      | ~70-150 MB  | After training     |
| Data (3-5 years)    | ~1-5 MB     | AAPL.csv           |
| HuggingFace cache   | ~500 MB     | FinBERT downloads  |
| **Total**           | **~3-4 GB** | Full installation  |

---

## 🚀 Quick Start Commands

### **1. Setup (One-time)**

```cmd
setup_venv.bat
```

### **2. Verify**

```cmd
venv\Scripts\activate.bat
python scripts\verify_installation.py
```

### **3. Run Pipeline**

```cmd
run_pipeline.bat
```

### **4. Manual Execution**

```cmd
venv\Scripts\activate.bat
cd src
python main.py
```

---

## 📊 Expected Output Files

After successful execution:

```
models/
├── best_model.pt (70 MB)
│   ├── Model weights
│   ├── Optimizer state
│   ├── Training history
│   └── Best validation accuracy
│
└── stock_predictor_complete.pkl (75 MB)
    ├── Model architecture config
    ├── Trained weights
    ├── Feature scalers
    ├── Performance metrics
    └── Training history
```

---

## ✅ File Verification Checklist

Before running:

- [ ] `data/AAPL.csv` exists and is valid CSV
- [ ] `src/config.py` settings are correct
- [ ] `requirements.txt` is complete
- [ ] `setup_venv.bat` executed successfully
- [ ] `venv/` folder exists with Python

After setup:

- [ ] `venv/Scripts/python.exe` exists
- [ ] Can run `python -c "import torch; print(torch.cuda.is_available())"`
- [ ] Returns `True` (CUDA available)

After training:

- [ ] `models/best_model.pt` created
- [ ] `models/stock_predictor_complete.pkl` created
- [ ] Console shows 80%+ accuracy
- [ ] No error messages

---

## 🎯 Success Indicators

### **Setup Successful**

```
✅ Virtual environment created: venv\
✅ All dependencies installed successfully!
```

### **Pipeline Successful**

```
🎯 TEST SET RESULTS:
   Accuracy:   0.8234 (82.34%)
✅ SUCCESS! Target accuracy 80.0% achieved!
```

### **Files Created**

```
✅ Model saved to: models\best_model.pt
✅ Complete package: models\stock_predictor_complete.pkl
```

---

## 📚 Documentation Files

| File                 | Purpose             | Audience         |
| -------------------- | ------------------- | ---------------- |
| `README.md`          | Complete user guide | Users            |
| `PROJECT_SUMMARY.md` | Quick reference     | Quick lookup     |
| `FILE_STRUCTURE.md`  | This file           | Developers       |
| `data/README.md`     | Data format guide   | Data preparation |

---

## 🎉 You're Ready!

All files are in place. Follow these steps:

1. **Place data**: Copy `AAPL.csv` to `data/` folder
2. **Setup**: Run `setup_venv.bat` (5-10 min)
3. **Verify**: Run `scripts\verify_installation.py`
4. **Execute**: Run `run_pipeline.bat` (30-45 min)
5. **Success**: Check `models/` for trained model

**Good luck! 🚀**
