from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pickle
import numpy as np
import os

app = Flask(__name__)
CORS(app)

# Load model
MODEL_PATH = 'diabetes_model.pkl'

try:
    with open(MODEL_PATH, 'rb') as f:
        model = pickle.load(f)
    print("Model loaded successfully!")
except FileNotFoundError:
    print("Model not found. Please train the model first by running train_model.py")
    model = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if model is None:
            return jsonify({
                'error': 'Model not loaded. Please train the model first.'
            }), 500
        
        # Get data from request
        data = request.get_json()
        # Extract features in the correct order
        features = [
            float(data.get('Pregnancies', 0)),
            float(data.get('Glucose', 0)),
            float(data.get('BloodPressure', 0)),
            float(data.get('SkinThickness', 0)),
            float(data.get('Insulin', 0)),
            float(data.get('BMI', 0)),
            float(data.get('DiabetesPedigreeFunction', 0.5)),  # Default value
            float(data.get('Age', 0))
        ]

        features_array = np.array([features])
        prediction = model.predict(features_array)[0]
        probability = model.predict_proba(features_array)[0]

        risk_percentage = round(probability[1] * 100, 2)
        
        if risk_percentage < 30:
            risk_level = "Low"
        elif risk_percentage < 60:
            risk_level = "Moderate"
        else:
            risk_level = "High"
        
        return jsonify({
            'prediction': int(prediction),
            'risk_percentage': risk_percentage,
            'risk_level': risk_level,
            'message': 'Diabetic' if prediction == 1 else 'Non-Diabetic'
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )

