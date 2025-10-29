"""
Quick test script to verify installation
"""
import sys

print("="*80)
print("VERIFYING INSTALLATION")
print("="*80)

# Test 1: Python version
print("\n[1/7] Checking Python version...")
print(f"Python: {sys.version}")
major, minor = sys.version_info[:2]
if major >= 3 and minor >= 10:
    print("✅ Python version OK")
else:
    print("⚠️ Python 3.10+ recommended")

# Test 2: PyTorch
print("\n[2/7] Checking PyTorch...")
try:
    import torch
    print(f"PyTorch version: {torch.__version__}")
    print("✅ PyTorch installed")
except ImportError:
    print("❌ PyTorch not installed!")
    sys.exit(1)

# Test 3: CUDA
print("\n[3/7] Checking CUDA...")
if torch.cuda.is_available():
    print(f"✅ CUDA available: {torch.version.cuda}")
    print(f"   GPU: {torch.cuda.get_device_name(0)}")
    print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
else:
    print("⚠️ CUDA not available (will use CPU - training will be slower)")

# Test 4: Transformers
print("\n[4/7] Checking Transformers...")
try:
    import transformers
    print(f"Transformers version: {transformers.__version__}")
    print("✅ Transformers installed")
except ImportError:
    print("❌ Transformers not installed!")
    sys.exit(1)

# Test 5: Data processing libraries
print("\n[5/7] Checking data processing libraries...")
try:
    import pandas as pd
    import numpy as np
    from sklearn.preprocessing import StandardScaler
    print(f"Pandas: {pd.__version__}")
    print(f"NumPy: {np.__version__}")
    print("✅ Data processing libraries installed")
except ImportError as e:
    print(f"❌ Missing library: {e}")
    sys.exit(1)

# Test 6: Class balancing
print("\n[6/7] Checking imbalanced-learn...")
try:
    from imblearn.over_sampling import SMOTE
    print("✅ Imbalanced-learn installed")
except ImportError:
    print("❌ Imbalanced-learn not installed!")
    sys.exit(1)

# Test 7: Technical indicators
print("\n[7/7] Checking technical analysis library...")
try:
    import ta
    print("✅ TA library installed")
except ImportError:
    print("❌ TA library not installed!")
    sys.exit(1)

# Summary
print("\n" + "="*80)
print("VERIFICATION COMPLETE!")
print("="*80)
print("\n✅ All dependencies are installed correctly!")
print("\nYour system is ready to run the stock prediction pipeline.")
print("\nNext step: Run 'run_pipeline.bat' to start training!")
print("="*80)
