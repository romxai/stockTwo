"""
Main pipeline script for stock prediction

This script orchestrates the entire end-to-end pipeline:
1. Load stock price and news data
2. Engineer 150+ technical features
3. Extract FinBERT text embeddings from news
4. Prepare sequences for deep learning
5. Build and train hybrid neural network
6. Evaluate on test set
7. Save model package for deployment

The pipeline is designed to be reproducible and modular.
"""
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

import torch
import numpy as np
import warnings
warnings.filterwarnings('ignore')  # Suppress sklearn/numpy warnings

from config import *
from data_loader import load_stock_data, merge_stock_news, create_labels
from feature_engineering import create_advanced_features, prepare_data, apply_smote, create_dataloaders
from text_embeddings import load_finbert, extract_finbert_features
from model import UltraAdvancedStockPredictor
from train import train_model
from evaluate import evaluate_full, calculate_metrics, print_evaluation_results
import pickle


def set_seeds(seed=42):
    """
    Set random seeds for reproducibility
    
    Ensures that runs with the same data and hyperparameters produce
    identical results (important for debugging and fair comparison).
    
    Args:
        seed: Random seed value
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)  # For multi-GPU setups


def main():
    """
    Main pipeline - orchestrates all steps from data loading to model evaluation
    
    Pipeline stages:
    1. Data Loading: Load stock prices and news
    2. Feature Engineering: Create 150+ technical indicators
    3. Text Embeddings: Extract FinBERT embeddings from news
    4. Data Preparation: Create sequences and splits
    5. Model Building: Initialize neural network
    6. Training: Train with early stopping
    7. Evaluation: Test set performance
    8. Saving: Save complete model package
    """
    print("="*80)
    print("🚀 STOCK PRICE PREDICTION WITH ADVANCED NLP & DEEP LEARNING")
    print("="*80)

    # Set random seeds for reproducibility across runs
    set_seeds(SEED)

    # Check device and print GPU info if available
    print(f"\n🔥 Using device: {DEVICE}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
        print(f"   Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"   CUDA Version: {torch.version.cuda}")

    # ============================================================================
    # STEP 1: LOAD DATA
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 1: LOADING DATA")
    print("="*80)

    stock_df = load_stock_data(STOCK_DATA_FILE)
    merged_df = merge_stock_news(stock_df, DATA_DIR)

    # ============================================================================
    # STEP 2: FEATURE ENGINEERING
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 2: FEATURE ENGINEERING")
    print("="*80)

    features_df = create_advanced_features(merged_df)
    features_df = create_labels(features_df)

    # ============================================================================
    # STEP 3: TEXT EMBEDDINGS
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 3: EXTRACTING TEXT EMBEDDINGS")
    print("="*80)

    finbert_model, finbert_tokenizer = load_finbert(FINBERT_MODEL_NAME, DEVICE)
    news_texts = features_df['news_combined'].tolist()
    news_embeddings, news_sentiments = extract_finbert_features(
        news_texts, finbert_model, finbert_tokenizer, DEVICE, TEXT_BATCH_SIZE
    )

    # Add sentiment scores
    features_df['sentiment_positive'] = news_sentiments[:, 0]
    features_df['sentiment_negative'] = news_sentiments[:, 1]
    features_df['sentiment_neutral'] = news_sentiments[:, 2]

    print(f"\n📊 Final feature summary:")
    print(f"   Numerical features: {len(features_df.columns) - 1}")
    print(f"   Text embeddings: {news_embeddings.shape[1]}")

    # ============================================================================
    # STEP 4: DATA PREPARATION
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 4: DATA PREPARATION")
    print("="*80)

    data_splits = prepare_data(features_df, news_embeddings, SEQUENCE_LENGTH)

    # ============================================================================
    # STEP 4.5: CLASS BALANCING (COMMENTED OUT - SMOTE CORRUPTS TIME-SERIES DATA)
    # ============================================================================
    # SMOTE (Synthetic Minority Over-sampling) is NOT USED because:
    # 1. It creates synthetic samples by interpolating between existing samples
    # 2. This averaging destroys temporal structure in time series data
    # 3. Alternative: Use class weights or Focal Loss (implemented in model.py)
    #
    # print("\n" + "="*80)
    # print("STEP 4.5: APPLYING SMOTE FOR CLASS BALANCING")
    # print("="*80)
    #
    # # COMMENTED OUT: SMOTE corrupts time-series data by averaging across time steps
    # # The FocalLoss in MultiTaskLoss already handles class imbalance
    # data_splits['train']['X_num'], data_splits['train']['X_text'], data_splits['train']['y_dir'] = \
    #     apply_smote(
    #         data_splits['train']['X_num'],
    #         data_splits['train']['X_text'],
    #         data_splits['train']['y_dir']
    #     )
    #
    # # Update magnitude and volatility targets
    # n_new = len(data_splits['train']['y_dir'])
    # n_old = len(data_splits['train']['y_mag'])
    # if n_new > n_old:
    #     indices = np.random.choice(n_old, n_new - n_old, replace=True)
    #     data_splits['train']['y_mag'] = np.concatenate([
    #         data_splits['train']['y_mag'],
    #         data_splits['train']['y_mag'][indices]
    #     ])
    #     data_splits['train']['y_vol'] = np.concatenate([
    #         data_splits['train']['y_vol'],
    #         data_splits['train']['y_vol'][indices]
    #     ])

    # Create PyTorch DataLoaders for efficient batching
    dataloaders = create_dataloaders(data_splits, TRAINING_CONFIG['batch_size'])

    # ============================================================================
    # STEP 5: BUILD MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 5: BUILDING MODEL")
    print("="*80)

    # Infer input dimensions from data
    sample_batch = next(iter(dataloaders['train']))
    num_numerical_features = sample_batch[0].shape[-1]  # Technical indicators count
    num_text_features = sample_batch[1].shape[-1]       # FinBERT embedding dimension (768)

    print(f"   Numerical features: {num_numerical_features}")
    print(f"   Text features: {num_text_features}")

    # Initialize model with configuration from config.py
    model = UltraAdvancedStockPredictor(
        num_numerical_features=num_numerical_features,
        num_text_features=num_text_features,
        **MODEL_CONFIG  # Unpack d_model, num_heads, dropout, etc.
    ).to(DEVICE)

    # Print model statistics
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n📊 Model Statistics:")
    print(f"   Total parameters: {total_params:,}")
    print(f"   Model size: ~{total_params * 4 / 1e6:.1f} MB (FP32)")

    # ============================================================================
    # STEP 6: TRAIN MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 6: TRAINING MODEL")
    print("="*80)

    history, best_val_acc = train_model(
        model, dataloaders, TRAINING_CONFIG, DEVICE, BEST_MODEL_PATH
    )

    # ============================================================================
    # STEP 7: EVALUATE MODEL
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 7: EVALUATING MODEL")
    print("="*80)

    # Load best model checkpoint (based on validation accuracy during training)
    checkpoint = torch.load(BEST_MODEL_PATH)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"✅ Loaded best model from epoch {checkpoint['epoch'] + 1}")

    # Evaluate on held-out test set (never seen during training/validation)
    test_preds, test_probs, test_labels, test_confidence = evaluate_full(
        model, dataloaders['test'], DEVICE
    )

    # Calculate comprehensive metrics
    metrics = calculate_metrics(test_labels, test_preds, test_probs)

    # Display results in formatted table
    print_evaluation_results(metrics, test_confidence.mean())

    # Check if target accuracy was achieved
    if metrics['accuracy'] >= TRAINING_CONFIG['target_accuracy']:
        print(f"\n✅ SUCCESS! Target accuracy {TRAINING_CONFIG['target_accuracy']:.1%} achieved!")
    else:
        print(f"\n⚠️  Target accuracy {TRAINING_CONFIG['target_accuracy']:.1%} not quite reached")
        print(f"   Gap: {(TRAINING_CONFIG['target_accuracy'] - metrics['accuracy']):.2%}")

    # ============================================================================
    # STEP 8: SAVE COMPLETE PACKAGE
    # ============================================================================
    print("\n" + "="*80)
    print("STEP 8: SAVING MODEL PACKAGE")
    print("="*80)

    # Package everything needed for deployment/inference
    model_package = {
        'model_state_dict': model.state_dict(),           # Trained weights
        'model_config': {                                  # Architecture config
            'num_numerical_features': num_numerical_features,
            'num_text_features': num_text_features,
            **MODEL_CONFIG
        },
        'scalers': data_splits['scalers'],               # For normalizing new data
        'training_config': TRAINING_CONFIG,               # Training hyperparameters
        'performance': {                                   # Test set metrics
            'test_accuracy': metrics['accuracy'],
            'test_f1': metrics['f1'],
            'test_auc': metrics['auc'],
            'test_precision': metrics['precision'],
            'test_recall': metrics['recall']
        },
        'training_history': history                       # Learning curves
    }

    # Save as pickle file for easy loading
    with open(FULL_PACKAGE_PATH, 'wb') as f:
        pickle.dump(model_package, f)

    print(f"💾 Model Package Saved!")
    print(f"   Location: {FULL_PACKAGE_PATH}")
    print(f"   Size: {os.path.getsize(FULL_PACKAGE_PATH) / 1e6:.1f} MB")

    # ============================================================================
    # DONE
    # ============================================================================
    print("\n" + "="*80)
    print("🎉 PIPELINE COMPLETE!")
    print("="*80)
    print(f"✅ Best validation accuracy: {best_val_acc:.2%}")
    print(f"✅ Test accuracy: {metrics['accuracy']:.2%}")
    print(f"✅ Model saved to: {MODELS_DIR}")
    print("\n💡 To make predictions on new data, use the saved model in models/")
    print("="*80)


if __name__ == "__main__":
    main()
