# Heart Disease Classification Project

This project predicts whether a patient is likely to have heart disease based on several medical attributes. The model is built using TensorFlow's Keras API and achieves an accuracy of approximately **85.87%** on the test set.

## Dataset Description

The dataset contains medical information about 918 patients, with the goal of predicting the presence of heart disease (`HeartDisease` column). Below is a detailed description of the columns:

| Column          | Description                                                                 |
|------------------|-----------------------------------------------------------------------------|
| `Age`           | Age of the patient (in years).                                              |
| `Sex`           | Gender of the patient (`M` for Male, `F` for Female).                       |
| `ChestPainType` | Type of chest pain experienced (`ATA`, `NAP`, `ASY`, `TA`).                 |
| `RestingBP`     | Resting blood pressure (in mm Hg).                                          |
| `Cholesterol`   | Serum cholesterol level (in mg/dl).                                         |
| `FastingBS`     | Fasting blood sugar (`0` if < 120 mg/dl, `1` if > 120 mg/dl).               |
| `RestingECG`    | Results of resting electrocardiogram (`Normal`, `ST`, `LVH`).               |
| `MaxHR`         | Maximum heart rate achieved during exercise.                               |
| `ExerciseAngina`| Exercise-induced angina (`Y` for Yes, `N` for No).                          |
| `Oldpeak`       | Depression in ST segment during exercise relative to rest.                 |
| `ST_Slope`      | Slope of the ST segment (`Up`, `Flat`, `Down`).                             |
| `HeartDisease`  | Target variable (`0` for No, `1` for Yes).                                  |

## Project Workflow

1. **Data Preprocessing**:
   - Converted categorical features (`Sex`, `ChestPainType`, `RestingECG`, `ExerciseAngina`, `ST_Slope`) into numerical format using `OneHotEncoder`.
   - Scaled numerical columns to improve model performance.

2. **Model Architecture**:
   - The model is built using TensorFlow's `Sequential` API with the following layers:
     - Dense layer with 16 neurons and ReLU activation.
     - Another Dense layer with 16 neurons and ReLU activation.
     - Output Dense layer with 1 neuron and Sigmoid activation for binary classification.
   - Compiled with the Adam optimizer and Binary Crossentropy loss.

3. **Evaluation**:
   - The model was evaluated using a test dataset, achieving an accuracy of **85.87%**.
   - Confusion Matrix:
     ```
     [[64 13]
      [13 94]]
     ```

## Key Libraries Used

- `TensorFlow` for building the classification model.
- `pandas` for data manipulation.
- `scikit-learn` for preprocessing and evaluation.

## Conclusion

This project demonstrates a basic binary classification task using machine learning to predict heart disease. Future improvements may include hyperparameter tuning, feature selection, and testing on a larger dataset.

## Acknowledgements

The dataset used in this project is publicly available and used for educational purposes.

