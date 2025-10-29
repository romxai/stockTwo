"""
Evaluation module

Provides functions for evaluating model performance using standard
classification metrics:
- Accuracy: Overall correctness
- Precision: Of predicted positives, how many are correct?
- Recall: Of actual positives, how many did we find?
- F1 Score: Harmonic mean of precision and recall
- AUC-ROC: Area under ROC curve (measures discrimination ability)
- Confusion Matrix: Detailed breakdown of predictions
"""
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np


def evaluate(model, dataloader, criterion, device):
    """
    Evaluate model on validation/test set
    
    Runs model in eval mode (disables dropout, batch norm uses running stats)
    without computing gradients for speed and memory efficiency.

    Args:
        model: Neural network model
        dataloader: DataLoader for validation/test data
        criterion: Loss function
        device: cuda or cpu

    Returns:
        Dictionary with metrics: loss, accuracy, precision, recall, f1, auc
    """
    model.eval()  # Set to evaluation mode
    total_loss = 0
    all_preds = []
    all_probs = []
    all_labels = []

    # Disable gradient computation for faster inference
    with torch.no_grad():
        for X_num, X_text, y_dir, y_mag, y_vol in dataloader:
            # Move data to device
            X_num = X_num.to(device)
            X_text = X_text.to(device)
            y_dir = y_dir.to(device)
            y_mag = y_mag.to(device)
            y_vol = y_vol.to(device)

            # Forward pass
            dir_logits, mag_preds, vol_preds = model(X_num, X_text)
            loss, _ = criterion(dir_logits, y_dir, mag_preds, y_mag, vol_preds, y_vol)

            # Accumulate metrics
            total_loss += loss.item()
            probs = F.softmax(dir_logits, dim=-1)  # Convert logits to probabilities
            preds = probs.argmax(dim=-1).cpu().numpy()  # Get class predictions
            all_preds.extend(preds)
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(y_dir.cpu().numpy())

    # Calculate aggregate metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    # weighted average accounts for class imbalance
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Calculate AUC-ROC (only for binary classification)
    all_probs = np.array(all_probs)
    if all_probs.shape[1] == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])  # Use prob of positive class
    else:
        auc = 0.0

    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc
    }


def evaluate_full(model, dataloader, device):
    """
    Full evaluation with confidence scores using MC Dropout
    
    Monte Carlo Dropout: Run multiple forward passes with dropout enabled
    to estimate prediction uncertainty. More variance = less confident.
    
    This is useful for identifying predictions that should be treated with caution.

    Args:
        model: Neural network model
        dataloader: DataLoader for test data
        device: cuda or cpu

    Returns:
        Tuple of numpy arrays:
        - predictions: Class predictions (0 or 1)
        - probabilities: Class probabilities
        - labels: True labels
        - confidence: Confidence scores (0-1, higher = more confident)
    """
    model.eval()
    test_preds = []
    test_probs = []
    test_labels = []
    test_confidence = []

    with torch.no_grad():
        for X_num, X_text, y_dir, _, _ in dataloader:
            X_num = X_num.to(device)
            X_text = X_text.to(device)

            # MC Dropout for confidence estimation (10 stochastic forward passes)
            preds, probs, confidence = model.predict_with_confidence(X_num, X_text, mc_samples=10)

            test_preds.extend(preds.cpu().numpy())
            test_probs.extend(probs.cpu().numpy())
            test_labels.extend(y_dir.numpy())
            test_confidence.extend(confidence.cpu().numpy())

    test_preds = np.array(test_preds)
    test_probs = np.array(test_probs)
    test_labels = np.array(test_labels)
    test_confidence = np.array(test_confidence)

    return test_preds, test_probs, test_labels, test_confidence


def calculate_metrics(labels, predictions, probabilities):
    """
    Calculate comprehensive classification metrics
    
    Args:
        labels: True labels (N,)
        predictions: Predicted labels (N,)
        probabilities: Class probabilities (N, 2)
        
    Returns:
        Dictionary with all metrics including confusion matrix
    """
    accuracy = accuracy_score(labels, predictions)
    precision = precision_score(labels, predictions, average='weighted', zero_division=0)
    recall = recall_score(labels, predictions, average='weighted', zero_division=0)
    f1 = f1_score(labels, predictions, average='weighted', zero_division=0)
    auc = roc_auc_score(labels, probabilities[:, 1])  # Use probability of positive class
    cm = confusion_matrix(labels, predictions)  # 2x2 matrix for binary classification

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': auc,
        'confusion_matrix': cm
    }


def print_evaluation_results(metrics, confidence):
    """
    Print formatted evaluation results
    
    Displays:
    - Classification metrics (accuracy, precision, recall, F1, AUC)
    - Average prediction confidence
    - Confusion matrix showing prediction breakdown
    
    Args:
        metrics: Dictionary with metric values
        confidence: Average confidence score
    """
    print("\n🎯 TEST SET RESULTS:")
    print("="*80)
    print(f"   Accuracy:   {metrics['accuracy']:.4f} ({metrics['accuracy']:.2%})")
    print(f"   Precision:  {metrics['precision']:.4f}")
    print(f"   Recall:     {metrics['recall']:.4f}")
    print(f"   F1-Score:   {metrics['f1']:.4f}")
    print(f"   AUC-ROC:    {metrics['auc']:.4f}")
    print(f"   Avg Confidence: {confidence:.4f}")
    print("="*80)

    # Confusion matrix breakdown
    print(f"\n📊 Confusion Matrix:")
    print(f"                Predicted")
    print(f"              Down   Up")
    print(f"Actual Down   {metrics['confusion_matrix'][0,0]:4d}  {metrics['confusion_matrix'][0,1]:4d}")
    print(f"       Up     {metrics['confusion_matrix'][1,0]:4d}  {metrics['confusion_matrix'][1,1]:4d}")
    print("="*80)
