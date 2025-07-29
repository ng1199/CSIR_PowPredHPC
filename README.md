# Power Consumption Forecasting in HPC Clusters using LSTM

## Project Overview

This project focuses on forecasting power consumption across high-performance computing (HPC) cluster nodes using LSTM-based deep learning. The goal is to provide accurate short-term predictions (5-minute intervals) using historical power usage data for multiple nodes, helping optimize energy efficiency and resource allocation in HPC environments.

## Problem Statement

Power consumption in HPC environments is highly dynamic and hard to forecast due to varying computational loads and complex node-level behaviors. Traditional statistical models fall short in capturing temporal dependencies. Therefore, this project applies LSTM, a type of Recurrent Neural Network (RNN), to learn temporal patterns and make multistep future predictions.

## Dataset

- **Source**: Real-world HPC node-wise power consumption data (collected over 5 months).
- **Structure**: Multiple node-wise columns like `1_1-Pow-consumption`, `2_11-Pow-consumption`, each sampled at 5-minute intervals.
- **Duration**: Historical data used up to June 10th for training. Model predicts power from **June 10 to June 17**.

## Project Features

- End-to-end data processing: cleaning, scaling, sequence generation
- Univariate LSTM-based time series forecasting
- Multi-step prediction (24 hours = 288 future time steps at 5-minute intervals)
- Model evaluation on test and validation sets
- Visualizations for predictions vs actuals
- Evaluation metrics: MAE, RMSE, R² Score

## Methodology

1. **Data Preprocessing**
   - Missing value handling
   - Feature scaling using `MinMaxScaler`
   - Sequence creation for time series input

2. **Model Architecture**
   - Sequential LSTM model
   - 1 LSTM layer with 50 units
   - Dense output layer
   - Optimized with `Adam`, loss = `MSE`

3. **Training & Validation**
   - Train-test-validation split
   - Model trained on multiple node columns independently
   - Performance tracked on both validation and unseen test data

4. **Multistep Prediction**
   - Autoregressive forecasting for 288 time steps (24 hours)
   - Rolling input windows for recursive prediction

5. **Evaluation Metrics**
   - Root Mean Square Error (RMSE)
   - Mean Absolute Error (MAE)
   - R² Score

## Model Performance

| Dataset     | MAE   | RMSE  | R² Score |
|-------------|-------|-------|----------|
| Test Set    | ~1.50 | ~1.98 | ~0.91    |
| Validation  | ~1.43 | ~1.85 | ~0.92    |

> Note: Performance may vary slightly by node, depending on load variability and signal pattern.

## Project File

- `Power_pred_HPC_LSTM.ipynb`: Jupyter notebook with full code for preprocessing, training, prediction, and visualization.
- Dataset should be placed in the same directory as `hardware_pow_data.xlsx`.

## How to Run

1. Clone this repository
2. Install dependencies:
    ```bash
    pip install pandas numpy matplotlib scikit-learn tensorflow openpyxl
    ```
3. Run the notebook:
    ```
    jupyter notebook Power_pred_HPC_LSTM.ipynb
    ```

## Limitations

- The model is trained independently per node, not using spatial correlation between nodes.
- Forecasting accuracy may degrade with higher forecast horizons due to error accumulation.
- Some models (like ARIMA and Prophet) were initially explored but did not handle multivariate multistep sequences well, so were replaced by LSTM.

## Future Improvements

- Extend to multivariate LSTM or Graph Neural Networks to capture spatial-temporal dependencies.
- Deploy as a real-time forecasting microservice using Flask or FastAPI.
- Integrate anomaly detection to flag unusual power patterns.

## License

This project is released without any license and is strictly for academic or research purposes during my internship. Reuse of the code is permitted with proper credit.

## Author

**Naman Gupta**  
Intern, CSIR (Council of Scientific and Industrial Research)  
Project: Power Forecasting using LSTM in HPC Environments
