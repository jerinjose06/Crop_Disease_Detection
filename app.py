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
    predictor = CropDiseasePredictor(model_path=config.BEST_MODEL_PATH, model_type='full')
    model_loaded = True
except Exception as e:
    print(f"Warning: Failed to load model. Error: {e}")
    predictor = None
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
            # Run prediction logic using existing Python function
            results = predictor.predict(temp_path, top_k=3)
            return jsonify({
                'success': True,
                'predictions': results
            })
        except Exception as e:
            return jsonify({'success': False, 'error': str(e)}), 500
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

if __name__ == '__main__':
    # Add port configuration if needed
    app.run(debug=True, host='0.0.0.0', port=5000)
