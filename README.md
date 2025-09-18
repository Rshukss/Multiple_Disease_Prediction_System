# Multiple Disease Prediction System Using Machine Learning

Multiple Disease Prediction System is an enhanced machine learning-based web application for predicting multiple diseases including Diabetes, Heart Disease, Parkinson's Disease, and Breast Cancer. It provides an integrated platform for healthcare predictions using optimized ML models with hyperparameter tuning and advanced feature engineering.

## Features

- **Multi-Disease Prediction:** Supports 4 major diseases with optimized ML models.
- **Interactive Web Interface:** User-friendly Streamlit application with real-time predictions.
- **Advanced Model Selection:** GridSearchCV with hyperparameter tuning for optimal performance.
- **Feature Engineering:** Correlation-based feature selection and PCA dimensionality reduction.
- **Model Persistence:** Trained models saved with comprehensive metadata.
- **Production Ready:** Scalable architecture with proper validation and error handling.

## Prerequisites

- **Python 3.8+** (recommended)
- **pip** (Python package installer)

## Installation

1. **Clone the Repository**
```
git clone https://github.com/Rshukss/Multiple_Disease_Prediction_System
```

3. **Install Required Python Packages**
```
pip install -r requirements.txt
```

4. **Run the Application**
```
streamlit run webapp.py
```

## Project Structure
```
multiple-disease-prediction-system/
│
├── webapp.py                          # Main Streamlit application
├── config.toml                        # Configuration settings
├── requirements.txt                   # Python dependencies
├── Breast_Cancer_Prediction/
│   ├── breastCancer.sav              # Trained breast cancer model
│   └── breastCancer_metadata.json    # Model metadata and parameters
├── Diabetes_Prediction/
│   ├── diabetes_metadata.json       # Model metadata
│   └── JupyterDP.ipynb              # Development notebook
├── Heart_Disease_Prediction/
│   ├── data.csv                     # Heart disease dataset
│   ├── heart_model.sav              # Trained heart disease model
│   └── heart_metadata.json         # Model metadata with PCA info
├── Parkinson's_Prediction/
│   ├── data.csv                     # Parkinson's dataset
│   ├── parkinsons.sav               # Trained Parkinson's model
│   └── parkinsons_metadata.json     # Model metadata
└── Saved Models/                     # Centralized model storage
    ├── breastCancer.sav
    ├── heart_model.sav
    └── parkinsons.sav
```


## Usage
- **Disease Selection:** Choose from 4 available disease prediction models.
- **Feature Input:** Enter patient data through intuitive form interfaces.
- **Real-time Prediction:** Get instant predictions with confidence scores.
- **Model Information:** View feature importance and model specifications.

## Supported Diseases

### 1. Diabetes Prediction
- **Enhanced Model:** Random Forest with GridSearchCV optimization
- **Features:** 6 selected features via correlation analysis
- **Interface:** User-friendly 3-column layout with validation

### 2. Heart Disease Prediction
- **Enhanced Model:** SVM with hyperparameter tuning
- **Preprocessing:** StandardScaler + PCA (90.3% variance retained)
- **Features:** 10 principal components from 13 original features

### 3. Parkinson's Disease Prediction
- **Enhanced Model:** Optimized SVM with RBF kernel
- **Features:** 16 acoustic and phonation parameters
- **Accuracy:** Improved through systematic hyperparameter tuning

### 4. Breast Cancer Prediction
- **Enhanced Model:** SVM with Linear kernel (GridSearchCV optimized)
- **Features:** 23 selected from 30 original features (correlation threshold = 0.3)
- **Performance:** Near-perfect accuracy with enhanced preprocessing

## Model Performance Enhancement

### ⚠️ Important Note on Results

This repository contains **enhanced versions** of the models described in our published paper: *"Multiple Disease Prediction System Using Machine Learning"*. 

**Key Enhancement:** The implementation now includes **GridSearchCV hyperparameter tuning** and advanced feature selection, which **may produce different results** than those reported in the original paper.

| Disease | Original Paper Results | Enhanced Implementation |
|---------|----------------------|------------------------|
| **Diabetes** | Logistic Regression (77%) | Random Forest (~82-85%)* |
| **Heart Disease** | SVM (80%) | Optimized SVM (~85-88%)* |
| **Breast Cancer** | SVM (97%) | Optimized SVM (~97-98%)* |
| **Parkinson's** | SVM (92%) | Optimized SVM (~94-96%)* |

*Expected performance with hyperparameter optimization and enhanced preprocessing

### Enhancements Made:
- **GridSearchCV:** Systematic hyperparameter optimization for all models
- **Cross-Validation:** StratifiedKFold (k=5) for robust evaluation
- **Feature Engineering:** Advanced correlation-based feature selection
- **Model Comparison:** Automated selection of best-performing algorithms
- **Pipeline Integration:** Proper preprocessing and scaling workflows

**Note:** While the original paper demonstrated proof-of-concept for multi-disease prediction, this enhanced version provides optimized performance through rigorous ML practices.

## Acknowledgements

- [Scikit-learn](https://scikit-learn.org/) for machine learning algorithms and preprocessing tools.
- [Streamlit](https://streamlit.io/) for the interactive web application framework.
- [Pandas](https://pandas.pydata.org/) and [NumPy](https://numpy.org/) for data manipulation.
- Original datasets from UCI Machine Learning Repository and Kaggle.

