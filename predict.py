# predict.py

import os
import json
import torch
import numpy as np
from PIL import Image
from torchvision import transforms

import config
from model import get_model
from utils import load_checkpoint


class CropDiseasePredictor:
    """Predictor class for crop disease identification."""

    def __init__(self, model_path=config.BEST_MODEL_PATH, model_type='full',
                 device=None):
        """
        Initialize predictor.

        Args:
            model_path: Path to saved model
            model_type: 'full' or 'light'
            device: Device to use for inference
        """
        self.device = device or torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu'
        )

        # Load model
        self.model = get_model(model_type=model_type)
        self.model, _, _, _ = load_checkpoint(self.model, filepath=model_path)
        self.model = self.model.to(self.device)
        self.model.eval()

        # Define transform
        self.transform = transforms.Compose([
            transforms.Resize(config.IMAGE_SIZE),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        self.class_names = config.CLASS_NAMES

        # Load remedies
        self.remedies = {}
        remedies_path = os.path.join(os.path.dirname(__file__), 'remedies.json')
        if os.path.exists(remedies_path):
            with open(remedies_path, 'r', encoding='utf-8') as f:
                self.remedies = json.load(f)

        print(f"Model loaded from {model_path}")
        print(f"Using device: {self.device}")

    def predict(self, image_path, top_k=5):
        """
        Predict disease from a single image.

        Args:
            image_path: Path to image file
            top_k: Number of top predictions to return

        Returns:
            predictions: List of (class_name, probability) tuples
        """
        # Load and transform image
        image = Image.open(image_path).convert('RGB')
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)

        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)

        # Get top-k predictions
        top_probs, top_indices = torch.topk(probabilities, top_k)
        top_probs = top_probs.cpu().numpy()[0]
        top_indices = top_indices.cpu().numpy()[0]

        predictions = []
        for prob, idx in zip(top_probs, top_indices):
            class_name = self.class_names[idx]
            # Parse plant and disease
            parts = class_name.split('___')
            plant = parts[0].replace('_', ' ')
            disease = parts[1].replace('_', ' ') if len(parts) > 1 else 'Unknown'

            pred_data = {
                'class': class_name,
                'plant': plant,
                'disease': disease,
                'probability': float(prob),
                'percentage': float(prob * 100)
            }
            if class_name in self.remedies:
                pred_data['remedy_info'] = self.remedies[class_name]

            predictions.append(pred_data)

        return predictions

    def predict_batch(self, image_paths, top_k=3):
        """Predict diseases for multiple images."""
        results = []
        for path in image_paths:
            try:
                preds = self.predict(path, top_k=top_k)
                results.append({
                    'image': path,
                    'predictions': preds,
                    'status': 'success'
                })
            except Exception as e:
                results.append({
                    'image': path,
                    'predictions': [],
                    'status': f'error: {str(e)}'
                })
        return results

    def predict_and_display(self, image_path, top_k=5):
        """Predict and print formatted results."""
        predictions = self.predict(image_path, top_k=top_k)

        print(f"\n{'='*60}")
        print(f"Prediction for: {os.path.basename(image_path)}")
        print(f"{'='*60}")

        for i, pred in enumerate(predictions):
            bar = '█' * int(pred['percentage'] / 2)
            print(f"\n  {i+1}. {pred['plant']} - {pred['disease']}")
            print(f"     Confidence: {pred['percentage']:.2f}% {bar}")

        print(f"\n{'='*60}")

        # Determine if healthy or diseased
        top_pred = predictions[0]
        is_healthy = 'healthy' in top_pred['disease'].lower()
        if 'remedy_info' in top_pred:
            is_healthy = top_pred['remedy_info'].get('is_healthy', is_healthy)

        if is_healthy:
            print(f"  ✓ Plant appears HEALTHY ({top_pred['plant']})")
            if 'remedy_info' in top_pred and top_pred['remedy_info'].get('remedies', {}).get('maintenance'):
                print("\n  Maintenance Tips:")
                for tip in top_pred['remedy_info']['remedies']['maintenance']:
                    print(f"    - {tip}")
            if 'remedy_info' in top_pred and top_pred['remedy_info'].get('prevention_tips'):
                print("\n  Prevention Tips:")
                for tip in top_pred['remedy_info']['prevention_tips']:
                    print(f"    - {tip}")
        else:
            print(f"  ⚠ Disease detected: {top_pred['disease']} on {top_pred['plant']}")
            print(f"    Confidence: {top_pred['percentage']:.2f}%")
            
            if 'remedy_info' in top_pred:
                info = top_pred['remedy_info']
                print(f"\n  Description: {info.get('description', 'N/A')}")
                
                remedies = info.get('remedies', {})
                if remedies:
                    print("\n  Recommended Remedies:")
                    for category, items in remedies.items():
                        if items:
                            category_name = category.replace('_', ' ').title()
                            print(f"    {category_name}:")
                            for item in items:
                                print(f"      - {item}")
                
                prev_tips = info.get('prevention_tips', [])
                if prev_tips:
                    print("\n  Prevention Tips:")
                    for tip in prev_tips:
                        print(f"    - {tip}")

        print(f"{'='*60}\n")

        return predictions

if __name__ == '__main__':
    import argparse
    import tkinter as tk
    from tkinter import filedialog
    import sys

    parser = argparse.ArgumentParser(description="Crop Disease Predictor")
    parser.add_argument('--image', type=str, help='Path to the image to predict')
    parser.add_argument('--browse', action='store_true', help='Open a file dialog to browse for an image')
    
    args = parser.parse_args()
    
    image_path = args.image
    
    if args.browse or not image_path:
        print("Opening file dialog to select an image...")
        root = tk.Tk()
        root.withdraw()  # Hide the main window
        # Bring dialog to front
        root.attributes('-topmost', True)
        
        image_path = filedialog.askopenfilename(
            title="Select Crop Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
                ("All files", "*.*")
            ]
        )
        root.destroy()
        
        if not image_path:
            print("No image selected. Exiting.")
            sys.exit(0)
            
    print(f"Loading model and preparing to predict for: {image_path}")
    try:
        predictor = CropDiseasePredictor()
        predictor.predict_and_display(image_path)
    except Exception as e:
        print(f"Error during prediction: {e}")