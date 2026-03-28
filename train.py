# train.py

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim

import config
from utils import (
    AverageMeter, EarlyStopping, TrainingHistory,
    save_checkpoint, format_time
)


def train_one_epoch(model, train_loader, criterion, optimizer, device, epoch):
    """Train for one epoch."""
    model.train()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    total_batches = len(train_loader)

    for batch_idx, (images, labels) in enumerate(train_loader):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient clipping to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

        optimizer.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs.data, 1)
        batch_acc = (predicted == labels).sum().item() / labels.size(0) * 100

        # Update meters
        loss_meter.update(loss.item(), labels.size(0))
        acc_meter.update(batch_acc, labels.size(0))

        # Print progress
        if (batch_idx + 1) % 50 == 0 or (batch_idx + 1) == total_batches:
            print(f"  Epoch [{epoch}] Batch [{batch_idx+1}/{total_batches}] "
                  f"Loss: {loss_meter.avg:.4f} | Acc: {acc_meter.avg:.2f}%")

    return loss_meter.avg, acc_meter.avg


def validate(model, val_loader, criterion, device):
    """Validate the model."""
    model.eval()

    loss_meter = AverageMeter()
    acc_meter = AverageMeter()

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            _, predicted = torch.max(outputs.data, 1)
            batch_acc = (predicted == labels).sum().item() / labels.size(0) * 100

            loss_meter.update(loss.item(), labels.size(0))
            acc_meter.update(batch_acc, labels.size(0))

    return loss_meter.avg, acc_meter.avg


def train_model(model, train_loader, valid_loader, device,
                epochs=config.EPOCHS, learning_rate=config.LEARNING_RATE):
    """
    Full training pipeline.

    Args:
        model: PyTorch model
        train_loader: Training data loader
        valid_loader: Validation data loader
        device: Device to train on
        epochs: Number of training epochs
        learning_rate: Initial learning rate

    Returns:
        model: Trained model
        history: TrainingHistory object
    """
    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}")
    print(f"Device: {device}")
    print(f"Epochs: {epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"{'='*60}\n")

    # Move model to device
    model = model.to(device)

    # Loss function
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=learning_rate,
        weight_decay=1e-4
    )

    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=True
    )

    # Early stopping
    early_stopping = EarlyStopping(
        patience=config.EARLY_STOPPING_PATIENCE,
        mode='min',
        verbose=True
    )

    # Training history
    history = TrainingHistory()

    # Track best model
    best_val_acc = 0.0
    best_val_loss = float('inf')
    total_train_time = 0

    for epoch in range(1, epochs + 1):
        epoch_start = time.time()

        print(f"\n{'─'*60}")
        print(f"Epoch {epoch}/{epochs}")
        print(f"{'─'*60}")

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        print(f"  Learning Rate: {current_lr:.2e}")

        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )

        # Validate
        val_loss, val_acc = validate(model, valid_loader, criterion, device)

        # Calculate epoch time
        epoch_time = time.time() - epoch_start
        total_train_time += epoch_time

        # Update history
        history.update(train_loss, train_acc, val_loss, val_acc, current_lr, epoch_time)

        # Print epoch results
        print(f"\n  {'─'*40}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Valid Loss: {val_loss:.4f} | Valid Acc: {val_acc:.2f}%")
        print(f"  Time: {format_time(epoch_time)}")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_val_loss = val_loss
            save_checkpoint(
                model, optimizer, scheduler, epoch, history,
                config.BEST_MODEL_PATH
            )
            print(f"  >>> New best model saved! (Val Acc: {val_acc:.2f}%)")

        # Update learning rate
        scheduler.step(val_loss)

        # Check early stopping
        if early_stopping(val_loss):
            print(f"\n  Early stopping triggered at epoch {epoch}")
            break

    print(f"\n{'='*60}")
    print(f"Training Complete!")
    print(f"{'='*60}")
    print(f"Total Training Time: {format_time(total_train_time)}")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Best Validation Loss: {best_val_loss:.4f}")
    print(f"Model saved to: {config.BEST_MODEL_PATH}")
    print(f"{'='*60}\n")

    # Save training history
    history.save(os.path.join(config.RESULTS_DIR, 'training_history.json'))

    return model, history