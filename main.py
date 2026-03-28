# main.py

import os
import sys
import time
import torch
import argparse

import config
from data_loader import get_data_loaders, set_seed
from model import get_model, model_summary, count_parameters
from train import train_model
from evaluate import evaluate_model, find_misclassified
from predict import CropDiseasePredictor
from utils import (
    plot_training_history, load_checkpoint, TrainingHistory
)


def setup_device():
    """Setup and configure the device."""
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"Using GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    else:
        device = torch.device('cpu')
        print("Using CPU")

    config.DEVICE = device
    return device


def print_banner():
    """Print application banner."""
    banner = """
    ╔══════════════════════════════════════════════════════════╗
    ║         CROP DISEASE IDENTIFICATION USING CNN           ║
    ║                                                          ║
    ║   Dataset: New Plant Diseases Dataset                    ║
    ║   38 Classes | 14 Crop Species | CNN Architecture        ║
    ╚══════════════════════════════════════════════════════════╝
    """
    print(banner)


def run_training(device, model_type='full'):
    """Run the complete training pipeline."""
    print("\n" + "="*60)
    print("PHASE 1: DATA LOADING")
    print("="*60)

    # Load data
    train_loader, valid_loader, test_loader, class_to_idx, idx_to_class = get_data_loaders()

    # Update class names from dataset
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    print(f"\nClasses found: {num_classes}")
    for i, name in enumerate(class_names):
        print(f"  {i:2d}: {name}")

    print("\n" + "="*60)
    print("PHASE 2: MODEL CREATION")
    print("="*60)

    # Create model
    model = get_model(model_type=model_type, num_classes=num_classes)
    model_summary(model)

    print("\n" + "="*60)
    print("PHASE 3: TRAINING")
    print("="*60)

    # Train model
    model, history = train_model(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        device=device,
        epochs=config.EPOCHS,
        learning_rate=config.LEARNING_RATE
    )

    # Plot training history
    plot_training_history(history)

    print("\n" + "="*60)
    print("PHASE 4: EVALUATION ON TEST SET")
    print("="*60)

    # Load best model for evaluation
    best_model = get_model(model_type=model_type, num_classes=num_classes)
    best_model, _, _, _ = load_checkpoint(best_model, filepath=config.BEST_MODEL_PATH)
    best_model = best_model.to(device)

    # Evaluate
    results = evaluate_model(
        model=best_model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    # Find misclassified samples
    print("\n" + "="*60)
    print("PHASE 5: MISCLASSIFICATION ANALYSIS")
    print("="*60)

    find_misclassified(
        model=best_model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    return results


def run_prediction(device, image_path, model_type='full'):
    """Run prediction on a single image."""
    predictor = CropDiseasePredictor(
        model_path=config.BEST_MODEL_PATH,
        model_type=model_type,
        device=device
    )

    predictions = predictor.predict_and_display(image_path)
    return predictions


def run_evaluation_only(device, model_type='full'):
    """Run evaluation only (requires trained model)."""
    _, _, test_loader, class_to_idx, idx_to_class = get_data_loaders()
    class_names = [idx_to_class[i] for i in range(len(idx_to_class))]
    num_classes = len(class_names)

    model = get_model(model_type=model_type, num_classes=num_classes)
    model, _, _, _ = load_checkpoint(model, filepath=config.BEST_MODEL_PATH)
    model = model.to(device)

    results = evaluate_model(
        model=model,
        test_loader=test_loader,
        device=device,
        class_names=class_names
    )

    return results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Crop Disease Identification using CNN')

    parser.add_argument('--mode', type=str, default='train',
                        choices=['train', 'evaluate', 'predict'],
                        help='Mode: train, evaluate, or predict')

    parser.add_argument('--model_type', type=str, default='full',
                        choices=['full', 'light'],
                        help='Model type: full or light')

    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs')

    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size')

    parser.add_argument('--lr', type=float, default=None,
                        help='Learning rate')

    parser.add_argument('--image', type=str, default=None,
                        help='Image path for prediction mode')

    parser.add_argument('--dataset_dir', type=str, default=None,
                        help='Dataset directory path')

    args = parser.parse_args()

    # Update config based on arguments
    if args.epochs:
        config.EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    if args.lr:
        config.LEARNING_RATE = args.lr
    if args.dataset_dir:
        config.DATASET_DIR = args.dataset_dir
        config.TRAIN_DIR = os.path.join(args.dataset_dir, "train")
        config.VALID_DIR = os.path.join(args.dataset_dir, "valid")

    # Print banner
    print_banner()

    # Set seed
    set_seed()

    # Setup device
    device = setup_device()

    # Run selected mode
    start_time = time.time()

    if args.mode == 'train':
        results = run_training(device, model_type=args.model_type)

    elif args.mode == 'evaluate':
        if not os.path.exists(config.BEST_MODEL_PATH):
            print(f"Error: No trained model found at {config.BEST_MODEL_PATH}")
            print("Please train a model first using --mode train")
            sys.exit(1)
        results = run_evaluation_only(device, model_type=args.model_type)

    elif args.mode == 'predict':
        if args.image is None:
            print("Error: Please provide an image path using --image <path>")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"Error: Image not found at {args.image}")
            sys.exit(1)
        if not os.path.exists(config.BEST_MODEL_PATH):
            print(f"Error: No trained model found at {config.BEST_MODEL_PATH}")
            sys.exit(1)
        predictions = run_prediction(device, args.image, model_type=args.model_type)

    total_time = time.time() - start_time
    print(f"\nTotal execution time: {total_time/60:.1f} minutes")


if __name__ == '__main__':
    main()