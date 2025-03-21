from flask import Flask, request, jsonify, render_template, send_from_directory
import os
from keras.models import load_model  # type: ignore
from keras.preprocessing.image import load_img, img_to_array  # type: ignore
import numpy as np
import tensorflow as tf
import logging

app = Flask(__name__)

# Load the trained model
model = load_model('model.h5')

# Define folder paths
UPLOAD_FOLDER = 'static/uploads'
MODEL_DIRECTORY = 'static/models'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Define class labels
class_labels = ['EOSINOPHIL', 'LYMPHOCYTE', 'MONOCYTE', 'NEUTROPHIL']

# Preprocess function
def preprocess_image(image_path):
    logging.info(f'Preprocessing image: {image_path}')
    image = load_img(image_path, target_size=(224, 224))  # Match model input size
    image = img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)  # Preprocessing for MobileNetV2
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# Home route
@app.route('/')
def index():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No image selected for uploading'})

    # Save the image
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)
    logging.info(f'Image uploaded to: {filepath}')

    # Preprocess and predict
    image = preprocess_image(filepath)
    prediction = model.predict(image)
    predicted_class = class_labels[np.argmax(prediction)]
    logging.info(f'Predicted class: {predicted_class}')

    # Send prediction result back as JSON
    return jsonify({'prediction': predicted_class})

# Serve 3D models
@app.route('/models/<path:filename>')
def send_model(filename):
    return send_from_directory(MODEL_DIRECTORY, filename)

if __name__ == '__main__':
    app.run(debug=True)
