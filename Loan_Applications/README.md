
# Loan Application Prediction Project

## Overview

This project aims to predict the success of loan applications based on various features extracted from the applicants' credit data and loan details. Using machine learning techniques, specifically an artificial neural network (ANN), the model predicts whether a loan application will be approved or not based on factors such as the applicant's credit history, employment type, loan amount, and more.

## Dataset

The project uses two datasets:

1. **Credit Features Dataset (`credit_features_subset.csv`)**: Contains information on the applicant's credit history, including age of oldest account, number of active accounts, defaults, and more.
2. **Loan Applications Dataset (`loan_applications.csv`)**: Contains details about the loan application, including loan amount, term, employment type, loan purpose, and whether the loan was successfully approved or not.

These two datasets are merged on the `UID` column to create a comprehensive dataset.

## Data Preprocessing

The following preprocessing steps were applied to the data:

1. **String Cleaning**: The `LoanPurpose` column was cleaned by converting all values to lowercase and stripping any leading/trailing spaces.
2. **Categorical Encoding**: The `LoanPurpose` and `EmploymentType` columns were one-hot encoded to convert categorical variables into numerical form, suitable for machine learning algorithms.
3. **Rare Categories Handling**: Rare categories in the `LoanPurpose` column were combined into a new category called `rare_categories` to prevent overfitting due to low-frequency values.

## Model Architecture

An artificial neural network (ANN) was built to predict loan approval success. The model has the following architecture:

- **Input Layer**: Takes in all the preprocessed features.
- **Hidden Layer**: 1 hidden layer with 32 units and ReLU activation function.
- **Output Layer**: A single output node with sigmoid activation function to predict the binary outcome (approved or not approved).

### Model Training

The model was trained using the following parameters:
- **Optimizer**: Adam
- **Loss Function**: Binary Cross-Entropy
- **Metrics**: Accuracy

An early stopping mechanism was used to prevent overfitting, with the training stopping if the validation loss did not improve for 3 consecutive epochs.

## Results

The model achieved the following results during training:

- **Training Accuracy**: ~89.5%
- **Validation Accuracy**: ~89.7%

The early stopping mechanism helped the model converge quickly and prevent overfitting, achieving a stable accuracy on both training and validation data.

## Dependencies

- `pandas`
- `numpy`
- `scikit-learn`
- `tensorflow`
- `matplotlib`
- `seaborn`

## Future Work

- Hyperparameter tuning to improve model performance.
- Implement other machine learning algorithms (e.g., Random Forest, XGBoost) for comparison.
- Enhance feature engineering for better model accuracy.
- Deploy the model as a web application or API.

