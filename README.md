# Car-Price-Prediction-Using-Multiple-Linear-Regression



## Overview  
This repository contains a machine-learning pipeline for predicting car prices based on the multiple independent features. I explored baseline linear regression, scaled, encoded features, and demonstrated how hyperparameter-tuned Gradient Boosting can explain up to 90% of the variance in price.

## Dataset  
- **Source**: https://www.kaggle.com/datasets/hellbuoy/car-price-prediction
- **Key features**:  
  - Numerical: wheelbase, carlength, carwidth, carheight, curbweight, enginesize, boreratio, stroke, compressionratio, horsepower, peakrpm, citympg, highwaympg  
  - Categorical: CarName, fueltype, aspiration, doornumber, carbody, drivewheel, enginelocation, enginetype, cylindernumber, fuelsystem  
- **Target**: `price` (continuous, log-transformed & standardized)

Model Performance
Baseline Linear Regression

R² (test): 0.843

RMSE (test): 0.414

Tuned Gradient Boosting Regressor

R² (test): 0.900

Key hyperparameters:

n_estimators: 200

max_depth: 5

learning_rate: 0.1

Insights & Key Findings
The chosen features explain the majority of price variance, with engine size, horsepower and curb weight being the strongest predictors.

Label encoding of categorical variables and standardisation of numerical features yielded stable training despite the small sample size.

Gradient Boosting’s non-linear modeling lifted R² from 0.843 to 0.900, demonstrating significant performance gains.

Future Work
Additional ensemble models: Evaluate Random Forest, XGBoost and LightGBM against the 0.843 baseline.

Feature engineering: One-hot encode high-cardinality categories (e.g. CarName), add interaction/polynomial terms, and experiment with a log transform on price.

Robust validation: Implement k-fold cross-validation, residual and heteroscedasticity analysis.

Pipeline & deployment: Wrap preprocessing and the final estimator into a sklearn.Pipeline for reproducibility and production readiness.
