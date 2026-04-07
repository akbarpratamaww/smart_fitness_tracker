import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, r2_score
import joblib
import os

MODEL_PATH = 'models/calorie_predictor.pkl'
ENCODER_PATH = 'models/label_encoder.pkl'

def train_calorie_prediction_model():
    """
    Train a machine learning model to predict calories burned during exercise.
    Uses Random Forest Regressor for accurate predictions.
    """
    try:
        # Load exercise dataset
        df = pd.read_csv('data/exercise_dataset.csv')
    except FileNotFoundError:
        # Create sample dataset if not exists
        np.random.seed(42)
        n_samples = 2000
        
        df = pd.DataFrame({
            'Gender': np.random.choice(['Male', 'Female'], n_samples),
            'Age': np.random.randint(18, 70, n_samples),
            'Height_cm': np.random.normal(170, 10, n_samples),
            'Weight_kg': np.random.normal(75, 15, n_samples),
            'Duration_min': np.random.randint(10, 120, n_samples),
            'Heart_Rate_bpm': np.random.randint(80, 180, n_samples),
            'Body_Temp_C': np.random.normal(37.5, 0.5, n_samples),
        })
        
        # Calculate calories burned (formula based)
        df['Calories_Burned'] = (
            0.5 * df['Duration_min'] + 
            0.8 * df['Weight_kg'] + 
            0.3 * df['Heart_Rate_bpm'] - 
            0.2 * df['Age'] +
            np.where(df['Gender'] == 'Male', 50, 0) +
            np.random.normal(0, 20, n_samples)
        )
        df['Calories_Burned'] = df['Calories_Burned'].clip(lower=50, upper=1200).round()
        
        os.makedirs('data', exist_ok=True)
        df.to_csv('data/exercise_dataset.csv', index=False)
    
    # Prepare features
    X = df.drop('Calories_Burned', axis=1)
    y = df['Calories_Burned']
    
    # Encode categorical variables
    label_encoders = {}
    categorical_cols = X.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        label_encoders[col] = le
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"Model trained! MAE: {mae:.2f} calories, R²: {r2:.3f}")
    
    # Save model
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    joblib.dump(label_encoders, ENCODER_PATH)
    
    return model, label_encoders, mae, r2

def load_model():
    """Load the pre-trained model."""
    try:
        model = joblib.load(MODEL_PATH)
        label_encoders = joblib.load(ENCODER_PATH)
        return model, label_encoders
    except:
        return train_calorie_prediction_model()

def predict_calories_burned(gender, age, height_cm, weight_kg, duration_min, heart_rate_bpm, body_temp_c):
    """Predict calories burned using the ML model."""
    model, label_encoders = load_model()
    
    # Prepare input data
    input_data = pd.DataFrame({
        'Gender': [gender],
        'Age': [age],
        'Height_cm': [height_cm],
        'Weight_kg': [weight_kg],
        'Duration_min': [duration_min],
        'Heart_Rate_bpm': [heart_rate_bpm],
        'Body_Temp_C': [body_temp_c]
    })
    
    # Encode categorical features
    if 'Gender' in label_encoders:
        input_data['Gender'] = label_encoders['Gender'].transform([gender])[0]
    
    # Predict
    prediction = model.predict(input_data)[0]
    return round(prediction, 0)