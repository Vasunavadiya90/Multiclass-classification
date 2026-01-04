# Credit Card Default Prediction - Multiclass Classification

A comprehensive machine learning project for predicting credit card default risk using multiclass classification. The project includes end-to-end ML pipeline implementation with data ingestion, transformation, model training, and deployment via FastAPI.

##  Project Overview

This project predicts credit scores (Poor, Standard, Good) (0,1,2) based on various financial and credit-related features. It implements a complete MLOps pipeline with:

- **Data Ingestion**: Automated data loading and train-test splitting
- **Data Transformation**: Feature engineering with preprocessing pipelines
- **Model Training**: Multiple classification algorithms with automatic model selection
- **REST API**: FastAPI-based prediction service
- **Custom Exception Handling**: Comprehensive error tracking and logging

## 📊 Features

The model predicts credit scores based on 20 input features:

| Feature | Type | Description |
|---------|------|-------------|
| `Delay_from_due_date` | Integer | Number of days delayed from due date |
| `Num_of_Delayed_Payment` | Integer | Total number of delayed payments |
| `Num_Credit_Inquiries` | Integer | Number of credit inquiries |
| `Credit_Utilization_Ratio` | Float | Credit utilization percentage |
| `Credit_History_Age` | Integer | Age of credit history in months |
| `Payment_of_Min_Amount` | String | Whether minimum amount is paid (Yes/No) |
| `Amount_invested_monthly` | Float | Monthly investment amount |
| `Monthly_Balance` | Float | Average monthly balance |
| `Credit_Mix` | String | Type of credit mix |
| `Payment_Behaviour` | String | Payment behavior pattern |
| `Age` | Integer | Customer age |
| `Annual_Income` | Float | Annual income |
| `Num_Bank_Accounts` | Integer | Number of bank accounts |
| `Num_Credit_Card` | Integer | Number of credit cards |
| `Interest_Rate` | Float | Average interest rate |
| `Num_of_Loan` | Integer | Number of active loans |
| `Monthly_Inhand_Salary` | Float | Monthly take-home salary |
| `Changed_Credit_Limit` | Float | Recent credit limit changes |
| `Outstanding_Debt` | Float | Total outstanding debt |
| `Total_EMI_per_month` | Float | Total EMI payments per month |

## 🏗️ Project Structure

```
Classification_project/
│
├── artifacts/                           # Generated artifacts (not in git)
│   ├── data.csv                        # Raw data
│   ├── train.csv                       # Training data
│   ├── test.csv                        # Testing data
│   ├── model.pkl                       # Trained model
│   ├── x_transformer.pkl               # Feature transformer
│   └── y_transformer.pkl               # Target transformer
│
├── logs/                                # Application logs (not in git)
│
├── notebooks/                           # Jupyter notebooks for EDA
│   └── data/
│       └── data_no_outliers.csv        # Cleaned dataset
│
├── src/                                 # Source code
│   ├── __init__.py
│   ├── logger.py                       # Logging configuration
│   ├── exceptions.py                   # Custom exception handling
│   ├── utils.py                        # Utility functions
│   │
│   ├── components.py/                  # ML pipeline components
│   │   ├── data_ingestion.py          # Data loading and splitting
│   │   ├── data_transformation.py     # Feature engineering
│   │   └── model_trainer.py           # Model training and evaluation
│   │
│   └── pipelines/                      # Prediction and training pipelines
│       ├── predict_pipeline.py        # Inference pipeline
│       └── training_pipeline.py       # Training orchestration
│
├── fast_api.py                         # FastAPI application
├── frontend_stramlit.py                # Streamlit frontend (optional)
├── requirements.txt                    # Python dependencies
├── setup.py                            # Package setup configuration
└── README.md                           # This file
```

## 🚀 Installation & Setup

### Prerequisites
- Python 3.8+
- pip package manager

### 1. Clone the Repository
```bash
git clone https://github.com/Vasunavadiya90/Multiclass-classification.git
cd Multiclass-classification
```

### 2. Create Virtual Environment
```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Install Package in Editable Mode
```bash
pip install -e .
```

## 🎓 Training the Model

To train the model from scratch:

```bash
python src/components.py/data_ingestion.py
```

This will:
1. Load data from `notebooks/data/data_no_outliers.csv`
2. Split into train/test sets (80/20)
3. Apply feature transformations
4. Train multiple classification models:
   - Logistic Regression
   - AdaBoost Classifier
   - Gradient Boosting Classifier
   - Random Forest Classifier
   - K-Neighbors Classifier
   - Decision Tree Classifier
   - XGBoost Classifier
5. Select the best performing model
6. Save artifacts to `artifacts/` folder

## 🌐 Running the API

### Start the FastAPI Server

```bash
uvicorn fast_api:app --reload
```

The API will be available at: `http://127.0.0.1:8000`

### API Documentation

Once the server is running, access:
- **Swagger UI**: `http://127.0.0.1:8000/docs`
- **ReDoc**: `http://127.0.0.1:8000/redoc`

### API Endpoints

#### POST `/predict`
Predicts credit score based on input features.

**Request Body Example:**
```json
{
  "Delay_from_due_date": 12,
  "Num_of_Delayed_Payment": 7,
  "Num_Credit_Inquiries": 2,
  "Credit_Utilization_Ratio": 24.366,
  "Credit_History_Age": 258,
  "Payment_of_Min_Amount": "No",
  "Amount_invested_monthly": 80.587,
  "Monthly_Balance": 455,
  "Credit_Mix": "Good",
  "Payment_Behaviour": "High_spent_Medium_value_payments",
  "Age": 26,
  "Annual_Income": 81817,
  "Num_Bank_Accounts": 5,
  "Num_Credit_Card": 5,
  "Interest_Rate": 4,
  "Num_of_Loan": 4,
  "Monthly_Inhand_Salary": 80471,
  "Changed_Credit_Limit": 11.37,
  "Outstanding_Debt": 809.88,
  "Total_EMI_per_month": 49.575854
}
```

**Response Example:**
```json
{
  "prediction:": 2.0
}
```

**Prediction Values:**
- `0.0` - Poor Credit Score
- `1.0` - Standard Credit Score
- `2.0` - Good Credit Score

### Testing the API

Using `curl`:
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "Delay_from_due_date": 12,
    "Num_of_Delayed_Payment": 7,
    "Num_Credit_Inquiries": 2,
    "Credit_Utilization_Ratio": 24.366,
    "Credit_History_Age": 258,
    "Payment_of_Min_Amount": "No",
    "Amount_invested_monthly": 80.587,
    "Monthly_Balance": 455,
    "Credit_Mix": "Good",
    "Payment_Behaviour": "High_spent_Medium_value_payments",
    "Age": 26,
    "Annual_Income": 81817,
    "Num_Bank_Accounts": 5,
    "Num_Credit_Card": 5,
    "Interest_Rate": 4,
    "Num_of_Loan": 4,
    "Monthly_Inhand_Salary": 80471,
    "Changed_Credit_Limit": 11.37,
    "Outstanding_Debt": 809.88,
    "Total_EMI_per_month": 49.575854
  }'
```

## 🔧 Components Explained

### 1. Data Ingestion (`data_ingestion.py`)
- Loads raw data from CSV
- Splits data into training and testing sets
- Saves processed data to artifacts folder

### 2. Data Transformation (`data_transformation.py`)
- **Numerical Pipeline**: Imputation + Standard Scaling
- **Categorical Pipeline**: Imputation + One-Hot Encoding + Scaling
- **Target Encoding**: Ordinal encoding for credit scores (Poor < Standard < Good)
- Saves preprocessing objects as pickle files

### 3. Model Trainer (`model_trainer.py`)
- Trains multiple classification algorithms
- Evaluates models using accuracy score
- Selects the best performing model
- Saves the best model to artifacts

### 4. Prediction Pipeline (`predict_pipeline.py`)
- Loads saved model and transformers
- Accepts input data via `CustomData` class
- Applies preprocessing transformations
- Returns predictions

### 5. Exception Handling (`exceptions.py`)
- Custom exception class with detailed error messages
- Captures file name, line number, and error message
- Integrates with logging system

### 6. Logging (`logger.py`)
- Automatic log file generation with timestamps
- Logs saved in `logs/` folder
- INFO level logging for tracking pipeline execution

## 📦 Dependencies

Core libraries used:
- **pandas**: Data manipulation
- **numpy**: Numerical operations
- **scikit-learn**: ML algorithms and preprocessing
- **fastapi**: REST API framework
- **uvicorn**: ASGI server
- **seaborn/matplotlib**: Data visualization (for EDA)


## Streamlit
- The app is model to user by making userunterface frontend using streamlit .

## 👤 Author

**Vasu Navadiya**
- GitHub: [@Vasunavadiya90](https://github.com/Vasunavadiya90)
- Email: Vasunavadiya933@gmail.com

## 🙏 Acknowledgments

- Dataset source: Credit card customer data
- Built with scikit-learn and FastAPI
- Inspired by real-world credit risk assessment systems

---

**Note**: The `artifacts/` and `logs/` folders are not committed to version control. Run the training pipeline to generate these files locally.
