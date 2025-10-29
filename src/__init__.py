"""
Stock Prediction Package

This package implements an advanced deep learning system for stock price prediction
that combines technical indicators, financial news sentiment analysis, and multi-modal
neural networks with attention mechanisms.

Main Components:
    - data_loader: Load and merge stock price and news data
    - feature_engineering: Create technical indicators and prepare sequences
    - text_embeddings: Extract FinBERT embeddings from financial news
    - model: Neural network architecture with cross-modal attention
    - train: Training loop with gradient accumulation and mixed precision
    - evaluate: Model evaluation and metrics calculation
    - config: Centralized configuration management
"""
__version__ = "1.0.0"
