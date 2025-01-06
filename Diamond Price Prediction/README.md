# Diamond Price Prediction using Neural Networks

This project involves predicting the price of diamonds based on various features such as carat weight, dimensions, clarity, and other attributes using a neural network model built with Keras.

## Dataset

The dataset used for this project is the **Diamonds Dataset**, which includes the following features:

- **Shape**: Shape of the diamond (e.g., Cushion, Pear, Oval, etc.)
- **Cut**: Quality of the cut (e.g., Ideal, Very Good, etc.)
- **Color**: Color of the diamond (e.g., D, E, F, etc.)
- **Clarity**: Clarity grade of the diamond (e.g., VS1, VVS2, etc.)
- **Carat Weight**: The weight of the diamond in carats
- **Length/Width Ratio**: Ratio of the diamond's length to its width
- **Depth %**: Depth percentage of the diamond
- **Table %**: Table percentage of the diamond
- **Polish**: Polish quality (e.g., Excellent, Very Good, etc.)
- **Symmetry**: Symmetry quality (e.g., Excellent, Very Good, etc.)
- **Girdle**: Girdle thickness of the diamond
- **Culet**: Culet size (dropped during preprocessing)
- **Fluorescence**: Fluorescence intensity (dropped during preprocessing)
- **Length, Width, Height**: Physical dimensions of the diamond
- **Price**: Target variable (price of the diamond)

### Preprocessing

- Missing values were handled by filling categorical columns with the mode (most frequent value) and numerical columns with the median.
- One-hot encoding was applied to categorical features like **Shape**, **Color**, **Clarity**, etc.
- The data was scaled using **StandardScaler** to normalize the features before training the model.

### Model Architecture

A **neural network** was built using Keras with the following layers:
- **Input Layer**: 64 units with ReLU activation
- **Hidden Layers**: 
  - 32 units with ReLU activation
  - 16 units with ReLU activation
- **Output Layer**: 1 unit with linear activation (suitable for regression)

The model was compiled using:
- **Optimizer**: Adam
- **Loss Function**: Mean Squared Error (MSE)
- **Metrics**: Mean Absolute Error (MAE)

### Training & Evaluation

The model was trained on the training set with a batch size of 32 and up to 100 epochs, using early stopping to prevent overfitting. The evaluation on the test set showed:

- **Test Loss (MSE)**: 1908992.375
- **Test MAE**: 547.1697

### Libraries Used

- **Pandas**: Data manipulation and analysis
- **Scikit-learn**: Preprocessing (OneHotEncoder, StandardScaler) and train-test split
- **Keras**: Building and training the neural network model
- **TensorFlow**: Backend for Keras

### How to Run

1. **Clone the repository**:

   ```bash
   git clone https://github.com/yourusername/diamond-price-prediction.git
   cd diamond-price-prediction
   ```

2. **Jupyter Notebook**:
   
   If you'd like to run the code in a Jupyter notebook, you can find the notebook in the repository (if applicable). Open it and run the cells step by step.

### Results

The model achieves a test MAE of around **$547** for predicting the price of diamonds. This can be further improved with hyperparameter tuning, feature engineering, or trying different regression algorithms.
