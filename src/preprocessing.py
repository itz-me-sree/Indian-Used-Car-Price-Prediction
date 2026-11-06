import numpy as np
import pandas as pd


def extract_brand(name):
    """Extract and standardize the car brand."""

    if pd.isna(name) or not str(name).strip():
        return np.nan

    name = str(name).strip()

    if name.lower().startswith("land rover"):
        return "Land Rover"

    brand = name.split()[0].title()

    # Preserve correct brand capitalization
    brand_mapping = {
        "Bmw": "BMW",
        "Isuzu": "Isuzu"
    }

    return brand_mapping.get(brand, brand)


def preprocess_data(df):
    """Clean and prepare the used car dataset."""

    clean_df = df.copy()

    # Remove unnecessary index column
    if 'Unnamed: 0' in clean_df.columns:
        clean_df.drop(columns=['Unnamed: 0'], inplace=True)

    # Extract car brands
    clean_df['Brand'] = clean_df['Name'].apply(extract_brand)

    # Convert Mileage into numeric format
    clean_df['Mileage'] = pd.to_numeric(
        clean_df['Mileage']
        .str.replace(' kmpl', '', regex=False)
        .str.replace(' km/kg', '', regex=False),
        errors='coerce'
    )

    # Convert Engine into numeric format
    clean_df['Engine'] = pd.to_numeric(
        clean_df['Engine'].str.replace(' CC', '', regex=False),
        errors='coerce'
    )

    # Convert Power into numeric format
    clean_df['Power'] = pd.to_numeric(
        clean_df['Power'].str.replace(' bhp', '', regex=False),
        errors='coerce'
    )

    # Replace invalid zero values with NaN
    clean_df['Mileage'] = clean_df['Mileage'].replace(0, np.nan)
    clean_df['Seats'] = clean_df['Seats'].replace(0, np.nan)

    # Remove New_Price due to excessive missing values
    if 'New_Price' in clean_df.columns:
        clean_df.drop(columns=['New_Price'], inplace=True)

    return clean_df