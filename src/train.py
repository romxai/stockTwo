"""
Training module
"""
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from sklearn.metrics import accuracy_score
import numpy as np
from datetime import datetime


def train_epoch(model, dataloader, criterion, optimizer, scheduler, scaler,
                gradient_accumulation_steps, max_grad_norm, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []

    optimizer.zero_grad()

    for batch_idx, (X_num, X_text, y_dir, y_mag, y_vol) in enumerate(dataloader):
        X_num = X_num.to(device)
        X_text = X_text.to(device)
        y_dir = y_dir.to(device)
        y_mag = y_mag.to(device)
        y_vol = y_vol.to(device)

        # Mixed precision forward pass
        if scaler:
            with autocast():
                dir_logits, mag_preds, vol_preds = model(X_num, X_text)
                loss, _ = criterion(dir_logits, y_dir, mag_preds, y_mag, vol_preds, y_vol)
                loss = loss / gradient_accumulation_steps

            scaler.scale(loss).backward()
        else:
            dir_logits, mag_preds, vol_preds = model(X_num, X_text)
            loss, _ = criterion(dir_logits, y_dir, mag_preds, y_mag, vol_preds, y_vol)
            loss = loss / gradient_accumulation_steps
            loss.backward()

        # Gradient accumulation
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

            optimizer.zero_grad()
            # --- ADD THIS CHECK ---
            if scheduler:
                scheduler.step()

        # Track metrics
        total_loss += loss.item() * gradient_accumulation_steps
        preds = dir_logits.argmax(dim=-1).cpu().numpy()
        labels = y_dir.cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels)

    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)

    return avg_loss, accuracy


def setup_training(model, dataloaders, config, device):
    """Setup training components"""
    from model import MultiTaskLoss

    criterion = MultiTaskLoss().to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config['learning_rate'],
        weight_decay=config['weight_decay'],
        betas=(0.9, 0.999)
    )

    total_steps = len(dataloaders['train']) * config['epochs']
    
    # --- COMMENT OUT THESE LINES ---
    # scheduler = torch.optim.lr_scheduler.OneCycleLR(
    #     optimizer,
    #     max_lr=1e-4, # This was 1e-3, you changed it to 1e-4
    #     total_steps=total_steps,
    #     pct_start=config.get('warmup_ratio', 0.3),
    #     anneal_strategy='cos',
    #     div_factor=25.0,
    #     final_div_factor=10000.0
    # )
    
    # --- ADD THIS LINE INSTEAD ---
    scheduler = None # Use a fixed LR

    scaler = GradScaler() if torch.cuda.is_available() else None

    return criterion, optimizer, scheduler, scaler


def train_model(model, dataloaders, config, device, save_path):
    """
    Main training loop

    Args:
        model: Neural network model
        dataloaders: Dictionary with train/val/test dataloaders
        config: Training configuration
        device: Device to train on
        save_path: Path to save best model

    Returns:
        history: Training history
        best_val_acc: Best validation accuracy
    """
    print("🚀 Starting training...")
    print("="*80)

    criterion, optimizer, scheduler, scaler = setup_training(model, dataloaders, config, device)

    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'learning_rate': []
    }

    best_val_acc = 0.0
    best_epoch = 0
    patience_counter = 0

    from evaluate import evaluate

    for epoch in range(config['epochs']):
        epoch_start = datetime.now()

        # Train
        train_loss, train_acc = train_epoch(
            model, dataloaders['train'], criterion, optimizer, scheduler,
            scaler, config['gradient_accumulation_steps'],
            config['max_grad_norm'], device
        )

        # Validate
        val_metrics = evaluate(model, dataloaders['val'], criterion, device)

        # Update history
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

        # Save best model
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch + 1
            patience_counter = 0

            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                # --- CHANGE THIS BLOCK BELOW ---
                'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
                # -------------------------------
                'best_val_acc': best_val_acc,
                'config': config,
                'history': history
            }, save_path)

            print(f"   ✅ New best model saved! Accuracy: {best_val_acc:.4f}")
        else:
            patience_counter += 1

        # Early stopping
        if patience_counter >= config['patience']:
            print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
            print(f"   Best validation accuracy: {best_val_acc:.4f} at epoch {best_epoch}")
            break

        # Check target
        if val_metrics['accuracy'] >= config['target_accuracy']:
            print(f"\n🎯 Target accuracy {config['target_accuracy']:.1%} reached!")
            print(f"   Validation accuracy: {val_metrics['accuracy']:.4f}")
            break

    print("\n" + "="*80)
    print("🎉 Training complete!")
    print(f"   Best validation accuracy: {best_val_acc:.4f} (epoch {best_epoch})")
    print(f"   Model saved to: {save_path}")

    return history, best_val_acc
