import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle

# Load the Pima Indians Diabetes dataset
# You can download from: https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database
# Or we'll create a function to download it

def download_diabetes_data():
    """Download diabetes dataset from UCI repository"""
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                    'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        df = pd.read_csv(url, names=column_names)
        print("Dataset downloaded successfully!")
        return df
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        print("Creating sample dataset for demonstration...")
        # Create a small sample dataset for demonstration
        np.random.seed(42)
        n_samples = 500
        data = {
            'Pregnancies': np.random.randint(0, 15, n_samples),
            'Glucose': np.random.randint(70, 200, n_samples),
            'BloodPressure': np.random.randint(50, 120, n_samples),
            'SkinThickness': np.random.randint(10, 60, n_samples),
            'Insulin': np.random.randint(0, 300, n_samples),
            'BMI': np.random.uniform(18, 50, n_samples),
            'DiabetesPedigreeFunction': np.random.uniform(0.1, 2.5, n_samples),
            'Age': np.random.randint(21, 80, n_samples),
            'Outcome': np.random.randint(0, 2, n_samples)
        }
        return pd.DataFrame(data)

def train_logistic_regression_model():
    print("="*60)
    print("Diabetes Risk Predictor - Model Training")
    print("="*60)
    
    # Load data
    print("\n1. Loading dataset...")
    df = download_diabetes_data()
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    
    # Check for missing values
    print(f"\nMissing values:\n{df.isnull().sum()}")
    
    # Replace zeros with median (for certain features, 0 is not realistic)
    features_to_replace_zeros = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
    for feature in features_to_replace_zeros:
        df[feature] = df[feature].replace(0, df[feature].median())
    
    # Separate features and target
    X = df.drop('Outcome', axis=1)
    y = df['Outcome']
    
    print(f"\nClass distribution:")
    print(f"Non-Diabetic (0): {sum(y==0)} ({sum(y==0)/len(y)*100:.2f}%)")
    print(f"Diabetic (1): {sum(y==1)} ({sum(y==1)/len(y)*100:.2f}%)")
    
    # Split the data
    print("\n2. Splitting dataset into train and test sets (80-20 split)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Scale the features
    print("\n3. Scaling features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train Logistic Regression model
    print("\n4. Training Logistic Regression model...")
    model = LogisticRegression(
        random_state=42,
        max_iter=1000,
        solver='lbfgs'
    )
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    print("\n5. Evaluating model...")
    y_pred_train = model.predict(X_train_scaled)
    y_pred_test = model.predict(X_test_scaled)
    
    # Calculate accuracy
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    print(f"\nTraining Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    # Confusion Matrix
    print(f"\nConfusion Matrix (Test Set):")
    print(confusion_matrix(y_test, y_pred_test))
    
    # Classification Report
    print(f"\nClassification Report (Test Set):")
    print(classification_report(y_test, y_pred_test, 
                                target_names=['Non-Diabetic', 'Diabetic']))
    
    # Feature importance (coefficients)
    print("\nFeature Importance (Coefficients):")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Coefficient': model.coef_[0]
    }).sort_values('Coefficient', ascending=False)
    print(feature_importance)
    
    # Create a pipeline that includes scaling
    from sklearn.pipeline import Pipeline
    final_model = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    
    # Save the model
    print("\n6. Saving model...")
    with open('diabetes_model.pkl', 'wb') as f:
        pickle.dump(final_model, f)
    
    print("\n" + "="*60)
    print("Model trained and saved successfully as 'diabetes_model.pkl'!")
    print("="*60)
    
    return final_model

if __name__ == "__main__":
    train_logistic_regression_model()
