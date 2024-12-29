

---

# Artificial Neural Network for Bank Customer Churn Prediction

This project implements an Artificial Neural Network (ANN) to predict whether a customer will leave the bank based on their profile and activity data. The dataset, preprocessing steps, model structure, training, and evaluation are described below.

---

## Project Overview

The goal is to predict customer churn using features such as credit score, geography, age, balance, etc. We preprocess the dataset, build an ANN model with Keras, and evaluate its performance.

---

## Dataset

The dataset is stored in a CSV file, **`Churn_Modelling.csv`**, containing customer information, such as:
- Geography
- Credit Score
- Gender
- Age
- Tenure
- Account Balance
- Number of Products
- Active Membership Status
- Estimated Salary
- Whether the customer has churned (label column)

---

## Workflow

### 1. **Importing Libraries**

The following libraries were used:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
```

---

### 2. **Data Loading and Feature Selection**
```python
dataset = pd.read_csv('Churn_Modelling.csv')
X = dataset.iloc[:, 3:-1].values  # Features
y = dataset.iloc[:, -1].values   # Labels
```
- **`X`**: Selects all columns except the first three and the last one as input features.
- **`y`**: Selects the last column as the target variable.

---

### 3. **Encoding Categorical Variables**

#### Encoding Gender
```python
le = LabelEncoder()
X[:, 2] = le.fit_transform(X[:, 2])
```
- Transforms the **Gender** column into numerical values: `Male = 1`, `Female = 0`.

#### Encoding Geography
```python
ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [1])], remainder='passthrough')
X = np.array(ct.fit_transform(X))
```
- One-hot encodes the **Geography** column, converting countries like `France`, `Spain`, `Germany` into binary columns.

---

### 4. **Splitting the Dataset**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
- Splits the dataset into training (80%) and testing (20%) sets.

---

### 5. **Feature Scaling**
```python
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```
- Standardizes the features to ensure all have similar ranges, improving model training.

---

### 6. **Building the ANN**
```python
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1, activation='sigmoid'))
```
- **Input Layer**: Automatically derived from the shape of `X_train`.
- **Two Hidden Layers**: Each has 6 neurons and ReLU activation.
- **Output Layer**: One neuron with sigmoid activation for binary classification.

---

### 7. **Compiling the Model**
```python
ann.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
```
- **Optimizer**: `adam` for adaptive learning.
- **Loss Function**: `binary_crossentropy` for binary classification.
- **Metric**: `accuracy` to track model performance during training.

---

### 8. **Training the Model**
```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```
- **Batch Size**: Processes 32 samples at a time.
- **Epochs**: Iterates 100 times over the dataset.

---

### 9. **Prediction**

#### Predicting for a New Customer
```python
print(ann.predict(sc.transform([[1, 0, 0, 600, 1, 40, 3, 60000, 2, 1, 1, 50000]])) > 0.5)
```
- Predicts churn for a specific customer.
- Returns `False`, indicating the customer is unlikely to leave.

#### Testing Predictions on the Test Set
```python
y_pred = ann.predict(X_test)
y_pred = (y_pred > 0.5)
```
- Predicts churn for the test set.
- Converts probabilities into binary values (`True` or `False`).

---

### 10. **Model Evaluation**
```python
cm = confusion_matrix(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
```
- **Confusion Matrix**: Shows the number of correct and incorrect predictions.
- **Accuracy Score**: Calculates overall model accuracy.

---

## Results

- **Confusion Matrix**:
  ```
  [[1504   91]
   [ 188  217]]
  ```
- **Accuracy**: `86.05%`

---
