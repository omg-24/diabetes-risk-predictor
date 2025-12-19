# Diabetes Risk Predictor

A web application that predicts diabetes risk using a Logistic Regression machine learning model.

## Features

- **Machine Learning Model**: Logistic Regression trained on the Pima Indians Diabetes dataset
- **Real-time Predictions**: Get instant risk assessment based on health metrics
- **Risk Categorization**: Low, Moderate, or High risk levels
- **User-friendly Interface**: Clean and responsive web design

## Installation

1. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

2. **Train the Model**:
```bash
python train_model.py
```

This will:
- Download the diabetes dataset
- Train a Logistic Regression model
- Save the model as `diabetes_model.pkl`

3. **Run the Application**:
```bash
python app.py
```

4. **Open in Browser**:
Navigate to `http://localhost:5000`

## How to Use

1. Fill in the required health metrics:
   - **Glucose Level** (mg/dL): Blood glucose concentration
   - **Blood Pressure** (mm Hg): Diastolic blood pressure
   - **BMI**: Body Mass Index
   - **Age**: Age in years
   - **Pregnancies** (optional): Number of pregnancies
   - **Skin Thickness** (optional): Triceps skin fold thickness
   - **Insulin** (optional): Insulin level

2. Click "Predict Risk"

3. View your risk assessment:
   - Risk percentage
   - Risk level (Low/Moderate/High)
   - Prediction result

## Model Information

- **Algorithm**: Logistic Regression
- **Dataset**: Pima Indians Diabetes Database
- **Features**: 8 health metrics
- **Accuracy**: ~75-80% (varies based on data split)

## Important Notes

⚠️ **Disclaimer**: This is a prediction tool based on machine learning. It should NOT replace professional medical advice. Always consult with healthcare professionals for proper diagnosis and treatment.

## Project Structure

```
diabetes-app/
├── app.py                 # Flask backend server
├── train_model.py         # Model training script
├── requirements.txt       # Python dependencies
├── diabetes_model.pkl     # Trained model (generated)
└── frontend-static/
    ├── index.html         # Web interface
    └── styles.css         # Styling
```

## Technologies Used

- **Backend**: Flask, Python
- **Machine Learning**: Scikit-learn (Logistic Regression)
- **Frontend**: HTML, CSS, JavaScript
- **Data Processing**: Pandas, NumPy
