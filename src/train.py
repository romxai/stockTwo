"""
Training module

Implements the training loop with advanced techniques:
- Mixed precision training (FP16) for speed and memory efficiency
- Gradient accumulation (simulate larger batch sizes)
- Gradient clipping (prevent exploding gradients)
- Learning rate scheduling
- Early stopping

These techniques are crucial for training deep networks on GPUs with limited memory.
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                gradient_accumulation_steps, max_grad_norm, device):
    """
    Train model for one epoch
    
    Gradient accumulation allows training with larger effective batch sizes
    by accumulating gradients over multiple small batches before updating weights.
    Effective batch size = batch_size * gradient_accumulation_steps
    
    Args:
        model: Neural network model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Parameter optimizer (AdamW)
        scheduler: Learning rate scheduler
        scaler: GradScaler for mixed precision training
        gradient_accumulation_steps: Number of batches to accumulate before update
        max_grad_norm: Gradient clipping threshold
        device: cuda or cpu
        
    Returns:
        avg_loss: Average training loss for epoch
        accuracy: Training accuracy
    """
    model.train()  # Enable dropout, batch norm updates
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()  # Clear gradients from previous epoch

    for batch_idx, (X_num, X_text, y_dir, y_mag, y_vol) in enumerate(dataloader):
        # Move batch to GPU/CPU
        X_num = X_num.to(device)
        X_text = X_text.to(device)
        y_dir = y_dir.to(device)
        y_mag = y_mag.to(device)
        y_vol = y_vol.to(device)

        # Mixed precision training: use FP16 for forward/backward, FP32 for weights
        # Reduces memory usage and speeds up training on modern GPUs
        if scaler:
            with autocast():  # Automatic mixed precision context
                dir_logits, mag_preds, vol_preds = model(X_num, X_text)
                loss, _ = criterion(dir_logits, y_dir, mag_preds, y_mag, vol_preds, y_vol)
                # Scale loss by accumulation steps (so accumulated gradient has correct magnitude)
                loss = loss / gradient_accumulation_steps

            # Scale loss to prevent underflow in FP16, then backprop
            scaler.scale(loss).backward()
        else:
            # Standard FP32 training
            dir_logits, mag_preds, vol_preds = model(X_num, X_text)
            loss, _ = criterion(dir_logits, y_dir, mag_preds, y_mag, vol_preds, y_vol)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        # Gradient accumulation: only update weights every N batches
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)  # Unscale gradients before clipping
                # Clip gradients to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)      # Update weights
                scaler.update()             # Update loss scale for next iteration
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()  # Clear gradients for next accumulation
            # Update learning rate scheduler (if exists)
            if scheduler:
                scheduler.step()

        # Track metrics for epoch summary
        total_loss += loss.item() * gradient_accumulation_steps
        preds = dir_logits.argmax(dim=-1).cpu().numpy()  # Get class predictions
        labels = y_dir.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def setup_training(model, dataloaders, config, device):
    """
    Setup training components: loss, optimizer, scheduler, scaler
    
    Args:
        model: Neural network to train
        dataloaders: Data loaders for calculating total steps
        config: Training configuration dictionary
        device: cuda or cpu
        
    Returns:
        criterion: Loss function (MultiTaskLoss)
        optimizer: AdamW optimizer
        scheduler: Learning rate scheduler (or None for fixed LR)
        scaler: GradScaler for mixed precision (or None for CPU)
    """
    from model import MultiTaskLoss

    # Multi-task loss with uncertainty weighting
    criterion = MultiTaskLoss().to(device)

    # AdamW: Adam with decoupled weight decay (better than L2 regularization)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],  # L2 penalty on weights
        betas=(0.9, 0.999)                     # Exponential decay rates for moments
    )

    total_steps = len(dataloaders['train']) * config['epochs']
    
    # --- SCHEDULER DISABLED ---
    # OneCycleLR was causing instability - comment out for fixed learning rate
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-4,
    #     total_steps=total_steps,
    #     pct_start=config.get('warmup_ratio', 0.3),  # Warmup for 30% of training
    #     anneal_strategy='cos',                       # Cosine annealing
    #     div_factor=25.0,                             # Initial LR = max_lr/25
    #     final_div_factor=10000.0                     # Final LR = max_lr/10000
    # )
    
    scheduler = None  # Use fixed learning rate for stability

    # GradScaler for automatic mixed precision (only if GPU available)
    scaler = GradScaler() if torch.cuda.is_available() else None

    return criterion, optimizer, scheduler, scaler


def train_model(model, dataloaders, config, device, save_path):
    """
    Main training loop with early stopping and checkpointing
    
    Training process:
    1. Train for one epoch
    2. Validate on validation set
    3. Save model if validation accuracy improves
    4. Stop if no improvement for 'patience' epochs
    5. Stop if target accuracy reached
    
    Tracks training history for later visualization and analysis.

    Args:
        model: Neural network model to train
        dataloaders: Dictionary with train/val/test dataloaders
        config: Training configuration (epochs, patience, target_accuracy, etc.)
        device: Device to train on (cuda or cpu)
        save_path: Path to save best model checkpoint

    Returns:
        history: Dictionary with training/validation metrics per epoch
        best_val_acc: Best validation accuracy achieved
    """
    print("🚀 Starting training...")
    print("="*80)

    criterion, optimizer, scheduler, scaler = setup_training(model, dataloaders, config, device)

    # Initialize training history for tracking progress
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0       # Track best validation accuracy
    best_epoch = 0           # Track which epoch had best accuracy
    patience_counter = 0     # Count epochs without improvement

    from evaluate import evaluate

    # Main training loop
    for epoch in range(config['epochs']):
        epoch_start = datetime.now()

        # Train for one epoch
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, scheduler,
            scaler, config['gradient_accumulation_steps'],
            config['max_grad_norm'], device
        )

        # Evaluate on validation set (no gradient computation)
        val_metrics = evaluate(model, dataloaders['val'], criterion, device)

        # Record metrics for this epoch
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['learning_rate'].append(optimizer.param_groups[0]['lr'])

        # Print progress
        epoch_time = (datetime.now() - epoch_start).total_seconds()

        print(f"Epoch {epoch+1:3d}/{config['epochs']} | "
              f"Time: {epoch_time:5.1f}s | "
              f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
              f"Train Loss: {train_loss:.4f} | "
              f"Train Acc: {train_acc:.4f} | "
              f"Val Loss: {val_metrics['loss']:.4f} | "
              f"Val Acc: {val_metrics['accuracy']:.4f} | "
              f"Val F1: {val_metrics['f1']:.4f}")

        # Save checkpoint if validation accuracy improved
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0  # Reset patience counter

            # Save complete checkpoint (can resume training if needed)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                'best_val_acc': best_val_acc,
                'config': config,
                'history': history
            }, save_path)

            print(f"   ✅ New best model saved! Accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1  # Increment if no improvement

        # Early stopping: stop if no improvement for 'patience' epochs
        if patience_counter >= config['patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

        # Stop early if target accuracy reached
        if val_metrics['accuracy'] >= config['target_accuracy']:
            print(f"\n🎯 Target accuracy {config['target_accuracy']:.1%} reached!")
            print(f"   Validation accuracy: {val_metrics['accuracy']:.4f}")
            break

    print("\n" + "="*80)
    print("🎉 Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"   Model saved to: {save_path}")

    return history, best_val_acc
