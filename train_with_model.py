"""
Convenience script to train with a specific model
Usage:
    python train_with_model.py simple
    python train_with_model.py advanced
    python train_with_model.py transformer
"""
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

def main():
    if len(sys.argv) < 2:
        print("Usage: python train_with_model.py <model_type>")
        print("Available models:")
        print("  - simple: SimpleHybridPredictor (LSTM-based, fast)")
        print("  - advanced: UltraAdvancedStockPredictor (Complex multi-scale)")
        print("  - transformer: TransformerStockPredictor (Pure attention)")
        sys.exit(1)
    
    model_type = sys.argv[1].lower()
    
    if model_type not in ['simple', 'advanced', 'transformer']:
        print(f"Error: Unknown model type '{model_type}'")
        print("Choose from: simple, advanced, transformer")
        sys.exit(1)
    
    # Update config
    import src.config as config
    config.MODEL_TYPE = model_type
    
    print(f"🚀 Training with model: {model_type.upper()}")
    print("="*80)
    
    # Run main
    from src.main import main as run_main
    run_main()


if __name__ == "__main__":
    main()
