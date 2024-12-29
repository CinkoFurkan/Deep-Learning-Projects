# Google Stock Price Prediction with LSTM

This project demonstrates the use of Long Short-Term Memory (LSTM) networks to predict Google stock prices based on historical data. The model is built using Python and key libraries such as Keras, NumPy, and Matplotlib.

## Table of Contents
1. [Overview](#overview)
2. [Dataset](#dataset)
3. [Requirements](#requirements)
4. [Implementation](#implementation)
5. [Results](#results)
6. [How to Run](#how-to-run)
7. [License](#license)

## Overview
LSTMs are a type of recurrent neural network (RNN) capable of learning long-term dependencies, which makes them well-suited for time series forecasting. This project uses an LSTM to predict Google stock prices.

## Dataset
- **Training Data**: `Google_Stock_Price_Train.csv`
- **Test Data**: `Google_Stock_Price_Test.csv`

The training set contains historical Google stock prices, and the test set is used for evaluation.

## Requirements
Ensure you have the following Python libraries installed:

- `numpy`
- `matplotlib`
- `pandas`
- `scikit-learn`
- `tensorflow` (or `keras`)

To install the required libraries, run:
```bash
pip install numpy matplotlib pandas scikit-learn tensorflow
```

## Implementation

### 1. Preprocessing
- The `Open` column from the dataset is used as the feature.
- Data is scaled using MinMaxScaler to normalize it between 0 and 1.
- The model uses 60 previous time steps to predict the next stock price.

### 2. Model Architecture
The LSTM model is built using the Keras library with the following layers:
- Four LSTM layers with 50 units each.
- Dropout layers with a rate of 0.2 to prevent overfitting.
- A dense layer with 1 unit for the output.

### 3. Training
The model is compiled with the Adam optimizer and Mean Squared Error (MSE) as the loss function. It is trained for 100 epochs with a batch size of 32.

### 4. Prediction
The model predicts the stock prices for the test set, which are then compared against the real stock prices.

## Results
The predictions are visualized using Matplotlib. The red line represents the real stock prices, and the blue line represents the predicted prices.

![Prediction Results](image.png)

## How to Run
1. Clone this repository:
   ```bash
   git clone <repository_url>
   ```
2. Navigate to the project directory:
   ```bash
   cd google-stock-prediction
   ```
3. Place the datasets (`Google_Stock_Price_Train.csv` and `Google_Stock_Price_Test.csv`) in the project directory.
4. Run the script:
   ```bash
   python stock_price_prediction.py
   ```
5. View the prediction results in a plot.

