import os
import tempfile
from flask import Flask, render_template, request, jsonify
from werkzeug.utils import secure_filename

import config
from predict import CropDiseasePredictor

app = Flask(__name__)

# Initialize predictor at startup to avoid reloading model per request
print("Loading model for the web application...")
try:
    general_predictor = CropDiseasePredictor(
        model_path=config.BEST_MODEL_PATH_GENERAL,
        model_type='full',
        class_names=config.CLASS_NAMES_GENERAL
    )
    banana_predictor = CropDiseasePredictor(
        model_path=config.BEST_MODEL_PATH_BANANA,
        model_type='light',
        class_names=config.CLASS_NAMES_BANANA
    )
    model_loaded = True
except Exception as e:
    print(f"Warning: Failed to load models. Error: {e}")
    general_predictor = None
    banana_predictor = None
    model_loaded = False

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_endpoint():
    if not model_loaded:
        return jsonify({'error': 'Model is not loaded properly. Ensure model exists at configuration path.'}), 500

    if 'image' not in request.files:
        return jsonify({'error': 'No image provided in request.'}), 400
    
    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file.'}), 400

    if file:
        filename = secure_filename(file.filename)
        # Create a temporary file to save the uploaded image
        fd, temp_path = tempfile.mkstemp(suffix=".jpg")
        os.close(fd)
        
        try:
            file.save(temp_path)
            # Run prediction on both models
            results_general = general_predictor.predict(temp_path, top_k=3)
            results_banana = banana_predictor.predict(temp_path, top_k=3)
            
            # Compare the top prediction from both models
            top_general = results_general[0]
            top_banana = results_banana[0]
            
            if top_banana['probability'] > top_general['probability']:
                final_results = results_banana
                print(f"[Model Selector] Chose Banana Model ({top_banana['percentage']:.2f}% vs {top_general['percentage']:.2f}%)")
            else:
                final_results = results_general
                print(f"[Model Selector] Chose General Model ({top_general['percentage']:.2f}% vs {top_banana['percentage']:.2f}%)")

            return jsonify({
                'success': True,
                'predictions': final_results
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    # Add port configuration if needed
    app.run(debug=True, host='0.0.0.0', port=5000)
