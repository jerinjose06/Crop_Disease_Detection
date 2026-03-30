# utils.py

import os
import json
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

import torch

import config


class AverageMeter:
    """Computes and stores the average and current value."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""

    def __init__(self, patience=config.EARLY_STOPPING_PATIENCE, min_delta=0.0,
                 mode='min', verbose=True):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, score):
        if self.best_score is None:
            self.best_score = score
            return False

        if self.mode == 'min':
            improved = score < self.best_score - self.min_delta
        else:
            improved = score > self.best_score + self.min_delta

        if improved:
            self.best_score = score
            self.counter = 0
        else:
            self.counter += 1
            if self.verbose:
                print(f"  EarlyStopping counter: {self.counter}/{self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
                return True

        return False


class TrainingHistory:
    """Track training metrics."""

    def __init__(self):
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        self.learning_rates = []
        self.epoch_times = []

    def update(self, train_loss, train_acc, val_loss, val_acc, lr, epoch_time):
        self.train_losses.append(train_loss)
        self.train_accuracies.append(train_acc)
        self.val_losses.append(val_loss)
        self.val_accuracies.append(val_acc)
        self.learning_rates.append(lr)
        self.epoch_times.append(epoch_time)

    def save(self, filepath):
        """Save history to JSON file."""
        history = {
            'train_losses': self.train_losses,
            'train_accuracies': self.train_accuracies,
            'val_losses': self.val_losses,
            'val_accuracies': self.val_accuracies,
            'learning_rates': self.learning_rates,
            'epoch_times': self.epoch_times
        }
        with open(filepath, 'w') as f:
            json.dump(history, f, indent=4)

    def load(self, filepath):
        """Load history from JSON file."""
        with open(filepath, 'r') as f:
            history = json.load(f)
        self.train_losses = history['train_losses']
        self.train_accuracies = history['train_accuracies']
        self.val_losses = history['val_losses']
        self.val_accuracies = history['val_accuracies']
        self.learning_rates = history['learning_rates']
        self.epoch_times = history['epoch_times']


def plot_training_history(history, save_dir=config.PLOTS_DIR):
    """Plot training and validation metrics."""
    epochs = range(1, len(history.train_losses) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Loss plot
    axes[0, 0].plot(epochs, history.train_losses, 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].plot(epochs, history.val_losses, 'r-', label='Validation Loss', linewidth=2)
    axes[0, 0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)

    # Accuracy plot
    axes[0, 1].plot(epochs, history.train_accuracies, 'b-', label='Training Accuracy', linewidth=2)
    axes[0, 1].plot(epochs, history.val_accuracies, 'r-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Training & Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # Learning rate plot
    axes[1, 0].plot(epochs, history.learning_rates, 'g-', linewidth=2)
    axes[1, 0].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('Learning Rate')
    axes[1, 0].set_yscale('log')
    axes[1, 0].grid(True, alpha=0.3)

    # Epoch time plot
    axes[1, 1].bar(epochs, history.epoch_times, color='steelblue', alpha=0.7)
    axes[1, 1].set_title('Time per Epoch', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Time (seconds)')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'training_history.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Training history plot saved to {save_dir}/training_history.png")


def plot_confusion_matrix(cm, class_names, save_dir=config.PLOTS_DIR,
                          normalize=True, figsize=(20, 18)):
    """Plot confusion matrix."""
    if normalize:
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        cm_plot = cm_normalized
        fmt = '.2f'
        title = 'Normalized Confusion Matrix'
    else:
        cm_plot = cm
        fmt = 'd'
        title = 'Confusion Matrix'

    # Shorten class names for display
    short_names = [name.replace('___', '\n').replace('_', ' ')[:30] for name in class_names]

    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm_plot, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=short_names, yticklabels=short_names,
                ax=ax, annot_kws={'size': 6})

    ax.set_title(title, fontsize=16, fontweight='bold')
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    plt.xticks(rotation=45, ha='right', fontsize=6)
    plt.yticks(rotation=0, fontsize=6)

    plt.tight_layout()
    filename = 'confusion_matrix_normalized.png' if normalize else 'confusion_matrix.png'
    plt.savefig(os.path.join(save_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved to {save_dir}/{filename}")


def plot_per_class_accuracy(class_accuracies, class_names, save_dir=config.PLOTS_DIR):
    """Plot per-class accuracy bar chart."""
    # Sort by accuracy
    sorted_pairs = sorted(zip(class_names, class_accuracies), key=lambda x: x[1])
    sorted_names, sorted_accs = zip(*sorted_pairs)

    # Shorten names
    short_names = [name.replace('___', '\n').replace('_', ' ')[:35] for name in sorted_names]

    colors = ['red' if acc < 80 else 'orange' if acc < 90 else 'green' for acc in sorted_accs]

    fig, ax = plt.subplots(figsize=(12, 16))
    bars = ax.barh(range(len(sorted_accs)), sorted_accs, color=colors, alpha=0.8)
    ax.set_yticks(range(len(short_names)))
    ax.set_yticklabels(short_names, fontsize=7)
    ax.set_xlabel('Accuracy (%)', fontsize=12)
    ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    ax.axvline(x=90, color='gray', linestyle='--', alpha=0.5, label='90% threshold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='x')

    # Add value labels
    for bar, acc in zip(bars, sorted_accs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{acc:.1f}%', va='center', fontsize=6)

    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'per_class_accuracy.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Per-class accuracy plot saved to {save_dir}/per_class_accuracy.png")


def plot_sample_predictions(images, true_labels, pred_labels, class_names,
                            save_dir=config.PLOTS_DIR, num_samples=16):
    """Plot sample predictions."""
    # Denormalize images
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

    num_samples = min(num_samples, len(images))
    nrows = int(np.ceil(num_samples / 4))

    fig, axes = plt.subplots(nrows, 4, figsize=(16, 4 * nrows))
    axes = axes.flatten() if num_samples > 4 else [axes] if num_samples == 1 else axes

    for i in range(num_samples):
        img = images[i].cpu() * std + mean
        img = img.permute(1, 2, 0).numpy()
        img = np.clip(img, 0, 1)

        true_name = class_names[true_labels[i]].replace('___', '\n')
        pred_name = class_names[pred_labels[i]].replace('___', '\n')
        correct = true_labels[i] == pred_labels[i]

        axes[i].imshow(img)
        axes[i].axis('off')

        color = 'green' if correct else 'red'
        title = f"True: {true_name}\nPred: {pred_name}"
        axes[i].set_title(title, fontsize=7, color=color, fontweight='bold')

    # Hide unused axes
    for j in range(num_samples, len(axes)):
        axes[j].axis('off')

    plt.suptitle('Sample Predictions (Green=Correct, Red=Wrong)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'sample_predictions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Sample predictions saved to {save_dir}/sample_predictions.png")


def format_time(seconds):
    """Format time in seconds to readable string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        return f"{seconds/60:.1f}m"
    else:
        return f"{seconds/3600:.1f}h"


def save_checkpoint(model, optimizer, scheduler, epoch, history, filepath):
    """Save training checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
        'history': {
            'train_losses': history.train_losses,
            'train_accuracies': history.train_accuracies,
            'val_losses': history.val_losses,
            'val_accuracies': history.val_accuracies,
            'learning_rates': history.learning_rates,
            'epoch_times': history.epoch_times
        }
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(model, optimizer=None, scheduler=None, filepath=config.BEST_MODEL_PATH_GENERAL, device=None):
    """Load training checkpoint."""
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])

    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    if scheduler and checkpoint.get('scheduler_state_dict'):
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])

    epoch = checkpoint.get('epoch', 0)
    return model, optimizer, scheduler, epoch