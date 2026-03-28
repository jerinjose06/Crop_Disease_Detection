# evaluate.py

import os
import numpy as np
import torch
import torch.nn as nn
from collections import defaultdict

import config
from utils import (
    AverageMeter, plot_confusion_matrix, plot_per_class_accuracy,
    plot_sample_predictions
)


def evaluate_model(model, test_loader, device, class_names=None, idx_to_class=None):
    """
    Comprehensive model evaluation on test set.

    Args:
        model: Trained PyTorch model
        test_loader: Test data loader
        device: Device
        class_names: List of class names
        idx_to_class: Dictionary mapping index to class name

    Returns:
        results: Dictionary with evaluation metrics
    """
    model.eval()
    model = model.to(device)

    if class_names is None:
        class_names = config.CLASS_NAMES

    num_classes = len(class_names)

    criterion = nn.CrossEntropyLoss()
    loss_meter = AverageMeter()

    all_predictions = []
    all_labels = []
    all_probs = []
    sample_images = []
    sample_true = []
    sample_pred = []

    print(f"\n{'='*60}")
    print(f"Evaluating Model on Test Set")
    print(f"{'='*60}\n")

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(images)
            loss = criterion(outputs, labels)

            # Get probabilities and predictions
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            loss_meter.update(loss.item(), labels.size(0))

            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

            # Collect samples for visualization
            if len(sample_images) < 16:
                remaining = 16 - len(sample_images)
                sample_images.extend(images[:remaining].cpu())
                sample_true.extend(labels[:remaining].cpu().numpy())
                sample_pred.extend(predicted[:remaining].cpu().numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Calculate overall metrics
    overall_accuracy = (all_predictions == all_labels).sum() / len(all_labels) * 100

    # Calculate per-class metrics
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    class_tp = defaultdict(int)
    class_fp = defaultdict(int)
    class_fn = defaultdict(int)

    for true, pred in zip(all_labels, all_predictions):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1
            class_tp[true] += 1
        else:
            class_fn[true] += 1
            class_fp[pred] += 1

    # Per-class accuracy, precision, recall, f1
    per_class_metrics = {}
    class_accuracies = []

    for i in range(num_classes):
        total = class_total.get(i, 0)
        correct = class_correct.get(i, 0)
        tp = class_tp.get(i, 0)
        fp = class_fp.get(i, 0)
        fn = class_fn.get(i, 0)

        accuracy = (correct / total * 100) if total > 0 else 0
        precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0

        class_accuracies.append(accuracy)
        per_class_metrics[class_names[i]] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'total_samples': total,
            'correct': correct
        }

    # Confusion matrix
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for true, pred in zip(all_labels, all_predictions):
        confusion_matrix[true][pred] += 1

    # Calculate macro and weighted averages
    precisions = [m['precision'] for m in per_class_metrics.values()]
    recalls = [m['recall'] for m in per_class_metrics.values()]
    f1_scores = [m['f1_score'] for m in per_class_metrics.values()]
    totals = [m['total_samples'] for m in per_class_metrics.values()]

    macro_precision = np.mean(precisions)
    macro_recall = np.mean(recalls)
    macro_f1 = np.mean(f1_scores)

    total_samples = sum(totals)
    weighted_precision = sum(p * t for p, t in zip(precisions, totals)) / total_samples
    weighted_recall = sum(r * t for r, t in zip(recalls, totals)) / total_samples
    weighted_f1 = sum(f * t for f, t in zip(f1_scores, totals)) / total_samples

    # Print results
    print(f"\n{'='*60}")
    print(f"EVALUATION RESULTS")
    print(f"{'='*60}")
    print(f"\nOverall Test Loss:     {loss_meter.avg:.4f}")
    print(f"Overall Test Accuracy: {overall_accuracy:.2f}%")
    print(f"\nMacro Precision:    {macro_precision:.2f}%")
    print(f"Macro Recall:       {macro_recall:.2f}%")
    print(f"Macro F1-Score:     {macro_f1:.2f}%")
    print(f"\nWeighted Precision: {weighted_precision:.2f}%")
    print(f"Weighted Recall:    {weighted_recall:.2f}%")
    print(f"Weighted F1-Score:  {weighted_f1:.2f}%")

    print(f"\n{'='*60}")
    print(f"PER-CLASS RESULTS")
    print(f"{'='*60}")
    print(f"{'Class':<50} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'Samples':>8}")
    print(f"{'─'*90}")

    for class_name in sorted(per_class_metrics.keys()):
        m = per_class_metrics[class_name]
        short_name = class_name[:48]
        print(f"{short_name:<50} {m['accuracy']:>7.2f}% {m['precision']:>7.2f}% "
              f"{m['recall']:>7.2f}% {m['f1_score']:>7.2f}% {m['total_samples']:>7d}")

    print(f"{'='*60}\n")

    # Generate plots
    print("Generating evaluation plots...")

    # Confusion Matrix
    plot_confusion_matrix(confusion_matrix, class_names, normalize=True)
    plot_confusion_matrix(confusion_matrix, class_names, normalize=False)

    # Per-class accuracy
    plot_per_class_accuracy(class_accuracies, class_names)

    # Sample predictions
    if sample_images:
        sample_images_tensor = torch.stack(sample_images)
        plot_sample_predictions(
            sample_images_tensor, sample_true, sample_pred, class_names
        )

    # Compile results
    results = {
        'overall_accuracy': overall_accuracy,
        'overall_loss': loss_meter.avg,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'weighted_precision': weighted_precision,
        'weighted_recall': weighted_recall,
        'weighted_f1': weighted_f1,
        'per_class_metrics': per_class_metrics,
        'confusion_matrix': confusion_matrix.tolist()
    }

    # Save results
    import json
    results_save = {k: v for k, v in results.items() if k != 'confusion_matrix'}
    results_save['confusion_matrix'] = confusion_matrix.tolist()
    with open(os.path.join(config.RESULTS_DIR, 'evaluation_results.json'), 'w') as f:
        json.dump(results_save, f, indent=4, default=str)

    print(f"\nResults saved to {config.RESULTS_DIR}/evaluation_results.json")

    return results


def find_misclassified(model, test_loader, device, class_names, num_samples=20):
    """Find and display misclassified samples."""
    model.eval()

    misclassified = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs.data, 1)

            mask = predicted != labels
            if mask.any():
                for i in range(mask.sum().item()):
                    idx = torch.where(mask)[0][i]
                    misclassified.append({
                        'image': images[idx].cpu(),
                        'true_label': labels[idx].item(),
                        'pred_label': predicted[idx].item(),
                        'confidence': probs[idx][predicted[idx]].item(),
                        'true_class': class_names[labels[idx].item()],
                        'pred_class': class_names[predicted[idx].item()]
                    })

            if len(misclassified) >= num_samples:
                break

    print(f"\nTop Misclassified Samples:")
    print(f"{'─'*80}")
    for i, m in enumerate(misclassified[:num_samples]):
        print(f"  {i+1}. True: {m['true_class']:<40} → Predicted: {m['pred_class']:<40} "
              f"(Conf: {m['confidence']:.2f})")

    return misclassified