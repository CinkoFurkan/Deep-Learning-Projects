---

# Artificial Neural Network for Power Plant Energy Prediction

This project implements an Artificial Neural Network (ANN) to predict the energy output of a power plant based on environmental parameters. The dataset, preprocessing steps, model structure, training, and evaluation are described below.

---

## Project Overview

The goal is to predict the **Power Output (PE)** of a power plant using features such as:
- **AT**: Temperature
- **V**: Exhaust Vacuum
- **AP**: Ambient Pressure
- **RH**: Relative Humidity

Using these features, we build and train a regression model using an ANN.

---

## Dataset

The dataset is stored in an Excel file, **`Folds5x2_pp.xlsx`**, which contains:
- **Features (Input)**: `AT`, `V`, `AP`, `RH`
- **Target (Output)**: `PE` (Power Output)

---

## Workflow

### 1. **Importing Libraries**

The following libraries were used:
```python
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
```

---

### 2. **Data Loading and Feature Selection**
```python
dataset = pd.read_excel('Folds5x2_pp.xlsx')
X = dataset.iloc[:, :-1].values  # Features
y = dataset.iloc[:, -1].values   # Target
```
- **`X`**: Includes columns `AT`, `V`, `AP`, and `RH`.
- **`y`**: Includes the target column `PE`.

---

### 3. **Exploratory Data Analysis**
```python
print(dataset.head())
```
This displays the first 5 rows of the dataset for a quick overview of the data.

Example output:
```
    AT      V       AP     RH      PE
0  14.96  41.76  1024.07  73.17  463.26
1  25.18  62.96  1020.04  59.08  444.37
2   5.11  39.40  1012.16  92.14  488.56
3  20.86  57.32  1010.24  76.64  446.48
4  10.82  37.50  1009.23  96.62  473.90
```

---

### 4. **Splitting the Dataset**
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
```
- Splits the dataset into training (80%) and testing (20%) sets.

---

### 5. **Building the ANN**
```python
ann = tf.keras.models.Sequential()
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=6, activation='relu'))
ann.add(tf.keras.layers.Dense(units=1))
```
- **Input Layer**: Automatically derived from the shape of `X_train`.
- **Two Hidden Layers**: Each has 6 neurons and ReLU activation.
- **Output Layer**: One neuron without activation, as this is a regression problem.

---

### 6. **Compiling the Model**
```python
ann.compile(optimizer='adam', loss='mean_squared_error')
```
- **Optimizer**: `adam` for adaptive learning.
- **Loss Function**: `mean_squared_error` for regression.

---

### 7. **Training the Model**
```python
ann.fit(X_train, y_train, batch_size=32, epochs=100)
```
- **Batch Size**: Processes 32 samples at a time.
- **Epochs**: Iterates 100 times over the dataset.

---

### 8. **Prediction**
```python
y_pred = ann.predict(X_test)
```
- Predicts the power output (`PE`) for the test set.

---

### 9. **Model Evaluation**
```python
np.set_printoptions(precision=2)
print(np.concatenate((y_pred.reshape(len(y_pred),1), y_test.reshape(len(y_test),1)),1))
```
- Displays predictions alongside actual values for comparison.

Example output:
```
[[434.72 433.26]
 [457.52 463.28]
 [435.12 431.23]
 ...
 [456.34 458.96]
 [462.15 466.65]
 [439.89 438.73]]
```

---

## Results

The predictions from the ANN are compared to the actual values using Mean Squared Error (MSE) during training, ensuring the model performs well in predicting the power output.

---
