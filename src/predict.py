import joblib
import numpy as np
import pandas as pd
from pathlib import Path


# Locate the saved model
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / "models" / "car_price_model.pkl"


def load_model():
    """Load the trained model and preprocessing objects."""
    return joblib.load(MODEL_PATH)


def predict_car_price(car_details):
    """Predict the price of a used car."""

    bundle = load_model()

    model = bundle['model']
    numeric_features = bundle['numeric_features']
    categorical_features = bundle['categorical_features']

    car_df = pd.DataFrame([car_details])

    # Handle missing numeric values
    numeric_data = bundle['numeric_imputer'].transform(
        car_df[numeric_features]
    )

    # Handle categorical values
    categorical_data = bundle['categorical_imputer'].transform(
        car_df[categorical_features]
    )

    # Encode categorical features
    encoded_data = bundle['encoder'].transform(categorical_data)

    # Combine numerical and categorical features
    processed_data = np.hstack([
        numeric_data,
        encoded_data
    ])

    # Predict car price
    predicted_price = model.predict(processed_data)

    return predicted_price[0]