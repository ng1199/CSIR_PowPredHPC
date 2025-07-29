# Power Consumption Prediction in HPC Clusters using LSTM

## Project Overview

This project focuses on forecasting power consumption across high-performance computing (HPC) cluster nodes using LSTM-based deep learning. The goal is to provide accurate short-term predictions (5-minute intervals) using historical power usage data for multiple nodes, helping optimize energy efficiency and resource allocation in HPC environments.

## Problem Statement

Power consumption in HPC environments is highly dynamic and hard to forecast due to varying computational loads and complex node-level behaviors. Traditional statistical models fall short in capturing temporal dependencies. Therefore, this project applies LSTM, a type of Recurrent Neural Network (RNN), to learn temporal patterns and make multistep future predictions.

## Dataset

- **Source**: Real-world HPC node-wise power consumption data collected internally during a CSIR research internship.
- **Structure**: Multiple node-wise columns like `1_1-Pow-consumption`, `2_11-Pow-consumption`, each sampled at 5-minute intervals.
- **Duration**: Historical data used up to June 10th for training. The model predicts power from **June 10 to June 17**.
- **Availability**: *Due to confidentiality agreements with CSIR laboratories, the dataset is not provided in this repository.*

## Project Features

- End-to-end data processing: cleaning, scaling, sequence generation
- Univariate LSTM-based time series forecasting
- Multi-step prediction (24 hours = 288 future time steps at 5-minute intervals)
- Model evaluation on test and validation sets
- Visualizations for predictions vs actuals
- Evaluation metrics: MAE, MSE, RMSE, R² Score

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
   - Mean Absolute Error (MAE)
   - Mean Squared Error (MSE)
   - Root Mean Squared Error (RMSE)
   - R² Score

## Model Performance

| Dataset     | MAE     | MSE        | RMSE    | R² Score |
|-------------|---------|------------|---------|----------|
| Test Set    | 1893.86 | 7695219.13 | 2774.03 | 0.9899   |
| Validation  | 1574.44 | 5238448.74 | 2288.77 | 0.9924   |

> Note: Performance may vary slightly by node, depending on load variability and signal patterns.

## Project File

- `Power_pred_HPC_LSTM.ipynb`: Jupyter notebook with complete code for data processing, model training, evaluation, and forecasting.
- **Dataset not included** due to confidentiality.

## How to Run

1. Clone this repository
2. Install dependencies:
    ```bash
    pip install pandas numpy matplotlib scikit-learn tensorflow openpyxl
    ```
3. Place your own compatible dataset as `hardware_pow_data.xlsx` in the working directory (if available).
4. Run the notebook:
    ```bash
    jupyter notebook Power_pred_HPC_LSTM.ipynb
    ```

## Limitations

- The model is trained independently per node, without leveraging spatial relationships across the cluster.
- Accuracy may degrade for longer forecasting windows due to error propagation.
- Baseline models like ARIMA and Prophet were evaluated but could not handle high-resolution multistep forecasting as efficiently as LSTM.

## Future Improvements

- Extend to multivariate LSTM or Graph Neural Networks for spatial-temporal modeling.
- Deploy model in real-time using Flask/FastAPI for HPC cluster monitoring.
- Integrate anomaly detection to flag unusual power usage.

## License

This project does not carry any open-source license and is strictly for academic or research use as part of an internship at CSIR. Reuse is allowed with proper attribution.

## Author

**Naman Gupta**  
Intern, CSIR (Council of Scientific and Industrial Research)  
Project: Power Forecasting using LSTM in HPC Environments
