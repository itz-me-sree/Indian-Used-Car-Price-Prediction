# Indian Used Car Price Prediction & Market Analysis

A machine learning project that explores the Indian used-car market and predicts the estimated resale price of a car using vehicle specifications and ownership details.

The project combines exploratory data analysis (EDA), data cleaning, feature engineering, regression modelling, model evaluation, and a reusable prediction module.

## Project Overview

Buying or selling a used car can be difficult because resale prices depend on several factors, including the vehicle's age, mileage, engine capacity, power, fuel type, transmission, brand, and location.

This project analyses a dataset of used cars in India and builds a regression model to estimate their resale prices. The target variable, `Price`, is measured in **Indian lakhs**.

## Objectives

- Understand the structure and characteristics of the used-car dataset.
- Explore relationships between vehicle features and resale prices.
- Clean inconsistent and missing data.
- Build and compare regression models.
- Evaluate prediction errors across different price ranges.
- Identify the features that contribute most to model predictions.
- Save the trained model and reuse it to predict the price of a new car.

## Dataset

**Source:** [Used Cars Price Prediction – Kaggle](https://www.kaggle.com/datasets/avikasliwal/used-cars-price-prediction)

The original training dataset contains **6,019 records and 14 columns**. The project uses the training dataset for analysis and model development. The original raw files are retained in `data/raw/`.

Important features include:

| Feature | Description |
|---|---|
| `Name` | Car name and model |
| `Location` | City where the car is listed |
| `Year` | Manufacturing year |
| `Kilometers_Driven` | Distance driven by the car |
| `Fuel_Type` | Petrol, Diesel, CNG, LPG, or Electric |
| `Transmission` | Manual or Automatic |
| `Owner_Type` | Ownership history |
| `Mileage` | Vehicle mileage |
| `Engine` | Engine capacity in CC |
| `Power` | Engine power in bhp |
| `Seats` | Seating capacity |
| `Price` | Used-car price in lakhs (target variable) |

`New_Price` is excluded from the initial modelling dataset because approximately 86% of its values are missing.

## Project Structure

```text
Indian-Used-Car-Price-Prediction/
│
├── data/
│   ├── raw/
│   │   ├── train-data.csv
│   │   └── test-data.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_Data_Understanding.ipynb
│   └── 02_Data_Preprocessing.ipynb
│
├── src/
│   ├── preprocessing.py
│   └── predict.py
│
├── models/
│   └── car_price_model.pkl
│
├── outputs/
│   ├── baseline_results.csv
│   └── model_comparison.csv
│
├── requirements.txt
└── README.md
```

## Workflow

### 1. Exploratory Data Analysis

Performed in `notebooks/01_Data_Understanding.ipynb`.

- Inspected dataset dimensions, data types, summary statistics, and missing values.
- Examined duplicate records and unusual values.
- Analysed price distributions and relationships with vehicle age, kilometres driven, fuel type, transmission, ownership type, and location.
- Explored brand-wise and model-wise price patterns.
- Studied numerical feature distributions and correlations.

### 2. Data Preprocessing

Performed in `notebooks/02_Data_Preprocessing.ipynb` and reusable functions in `src/preprocessing.py`.

- Removed the unnecessary `Unnamed: 0` index column.
- Extracted and standardized car brands from vehicle names.
- Converted `Mileage`, `Engine`, and `Power` from text with units into numeric values.
- Replaced invalid zero values in `Mileage` and `Seats` with missing values.
- Excluded `New_Price` from the initial model because of its high percentage of missing values.
- Used median imputation for numerical features and most-frequent imputation for categorical features.
- Applied one-hot encoding to categorical features.

A Porsche Cayenne record with a highly unusual recorded price was identified during error analysis and excluded from the modelling dataset as a suspected data-quality issue. This decision is documented in the preprocessing notebook; it is not treated as a proven data-entry error.

### 3. Model Development

The project experimented with:

- Linear Regression as a baseline comparison.
- Random Forest Regression as the main model.
- RandomizedSearchCV for Random Forest hyperparameter tuning.

The baseline Random Forest model performed better on the held-out test set than the tuned model, so the baseline model was retained as the final model candidate.

### 4. Model Evaluation

Evaluation included:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- R-squared (R²)
- Actual vs. predicted price plot
- Residual analysis
- Error analysis across different price categories
- Feature importance and permutation importance

## Results

### Model Comparison

| Model | Test MAE (lakhs) | Test RMSE (lakhs) | Test R² |
|---|---:|---:|---:|
| Linear Regression | 3.12 | 7.20 | 0.5788 |
| Random Forest (Baseline) | 1.51 | 4.50 | 0.8574 |
| Random Forest (Tuned) | 1.68 | 4.68 | 0.8458 |

The baseline Random Forest achieved the best test-set results among the compared models.

### Key Findings

- `Power` and `Year` were the two most influential features in the feature-importance analysis.
- The model performed better for budget and mid-range vehicles than for luxury vehicles.
- Mean Absolute Error increased across higher actual-price categories.
- Expensive cars were often underestimated, indicating a limitation in predicting high-value vehicles.
- Feature importance describes predictive contribution in this model; it does not establish causation.

**Note:** These metrics are based on the project's current train-test split and preprocessing choices. They should not be interpreted as a guarantee of prediction accuracy for every car or market listing.

## Sample Prediction

The saved model was tested with the following example vehicle:

| Input | Value |
|---|---:|
| Brand | Hyundai |
| Year | 2020 |
| Kilometers Driven | 35,000 |
| Mileage | 18.5 |
| Engine | 1197 CC |
| Power | 82 bhp |
| Seats | 5 |
| Location | Kochi |
| Fuel Type | Petrol |
| Transmission | Manual |
| Owner Type | First |

**Estimated price: ₹6.79 lakhs**

This is an example output from the trained model, not a verified market valuation.

## Getting Started

### 1. Clone the repository

```bash
git clone <your-github-repository-url>
cd Indian-Used-Car-Price-Prediction
```

Replace `<your-github-repository-url>` with the URL of your GitHub repository.

### 2. Create and activate a virtual environment (recommended)

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the notebooks

Open the project in Jupyter Notebook or VS Code and run:

1. `notebooks/01_Data_Understanding.ipynb`
2. `notebooks/02_Data_Preprocessing.ipynb`

Run the notebooks in order to follow the analysis and modelling workflow.

### 5. Predict a car price using the saved model

The trained model bundle should be available at `models/car_price_model.pkl`.

From the project root, run Python with the following example:

```python
from src.predict import predict_car_price

car = {
    "Year": 2020,
    "Kilometers_Driven": 35000,
    "Mileage": 18.5,
    "Engine": 1197,
    "Power": 82,
    "Seats": 5,
    "Location": "Kochi",
    "Fuel_Type": "Petrol",
    "Transmission": "Manual",
    "Owner_Type": "First",
    "Brand": "Hyundai"
}

estimated_price = predict_car_price(car)
print(f"Estimated Car Price: ₹{estimated_price:.2f} Lakhs")
```

## Limitations

- The dataset represents a particular collection of used-car listings and may not reflect current market prices.
- Some features contain missing or inconsistent values.
- Luxury vehicles are less represented than budget and mid-range vehicles in the test set.
- The model has difficulty estimating prices for expensive cars, especially when their prices fall outside the range well represented in the training data.
- The prediction is an estimate and does not account for every factor that affects a real transaction, such as vehicle condition, accident history, service records, or local demand.
- The saved model was trained with a particular version of the preprocessing and scikit-learn pipeline. For reproducibility, use compatible library versions when loading the model.

## Future Improvements

- Evaluate additional regression algorithms and validation strategies.
- Improve handling of rare and high-value vehicle segments.
- Explore more robust feature engineering and outlier investigation.
- Add a user interface or API for interactive price estimation.
- Monitor model performance on newer used-car listings.

## Technologies Used

- **Language:** Python
- **Data analysis:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-learn
- **Model persistence:** Joblib
- **Development:** Jupyter Notebook, VS Code

## Author

**Sreerag S**

GitHub: [itz-me-sree](https://github.com/itz-me-sree)

---

If you find an issue or have a suggestion, feel free to open an issue or contribute.
