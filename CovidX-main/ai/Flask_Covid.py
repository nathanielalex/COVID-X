from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import joblib
import pandas as pd

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = './uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

try:
    model = load_model('./covid_resnet.keras')
    print("COVID model loaded successfully.")
except Exception as e:
    print(f"Error loading COVID model: {e}")
    raise e

model_classification = joblib.load('knn_model.pkl')

def preprocess_image(image_path):
    try:
        img = cv2.imread(image_path)
        img = cv2.resize(img, (224, 224))
        img = img / 255.0
        img = np.array(img)
        img = np.expand_dims(img, axis=0)
        print(f"Image preprocessed successfully, shape: {img.shape}")
        return img
    except Exception as e:
        print(f"Error in preprocessing image: {e}")
        raise e

def predict_covid(image_path):
    try:
        img = preprocess_image(image_path)
        prediction = model.predict(img)
        print(prediction)
        
        classes = ['BACTERIAL-PNEUMONIA', 'COVID19', 'NORMAL', 'VIRAL-PNEUMONIA']
        predicted_label=classes[np.argmax(prediction)]
        return predicted_label
    except Exception as e:
        print(f"Error in prediction: {e}")
        raise e

@app.route('/predict', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400
    
    if file:
        filename = file.filename
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)  

        try:
            prediction = predict_covid(file_path)
            return jsonify({"message": f"Prediction: {prediction}"})
        except Exception as e:
            return jsonify({"message": f"Error during prediction: {e}"}), 500

@app.route('/covid-prediction', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    
    breathing = 1 if data['answer']['breathing'] == 'Yes' else 0
    fever = 1 if data['answer']['fever'] == 'Yes' else 0
    cough = 1 if data['answer']['cough'] == 'Yes' else 0
    soreThroat = 1 if data['answer']['soreThroat'] == 'Yes' else 0
    hyperTension = 1 if data['answer']['hyperTension'] == 'Yes' else 0
    abroad = 1 if data['answer']['abroad'] == 'Yes' else 0
    contact = 1 if data['answer']['contact'] == 'Yes' else 0
    gathering = 1 if data['answer']['gathering'] == 'Yes' else 0
    exposed = 1 if data['answer']['exposed'] == 'Yes' else 0
    family = 1 if data['answer']['family'] == 'Yes' else 0
    
    input_features = pd.DataFrame([[breathing, fever, cough, soreThroat, hyperTension, abroad, contact, gathering, exposed, family]],
                               columns=['Breathing Problem', 'Fever', 'Dry Cough', 'Sore throat', 'Hyper Tension', 'Abroad travel', 'Contact with COVID Patient', 'Attended Large Gathering', 'Visited Public Exposed Places', 'Family working in Public Exposed Places'])

    prediction = model_classification.predict(input_features)

    predicted_probabilities = model_classification.predict_proba(input_features)
    
    predicted_class = prediction[0]
    confidence_percentage = predicted_probabilities[0][predicted_class] * 100
    
    return jsonify({'prediction': prediction[0].item(), 'confidence' : confidence_percentage}) 

if __name__ == '__main__':
    app.run(debug=True, port=5000)
