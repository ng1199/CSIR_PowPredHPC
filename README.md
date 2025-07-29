Power Consumption Prediction in HPC Systems using LSTM
Overview
This project focuses on building a deep learning-based predictive model using Long Short-Term Memory (LSTM) networks to forecast node-level power consumption in High-Performance Computing (HPC) systems. The time series nature of power usage data across different HPC nodes makes LSTM a suitable choice for capturing temporal dependencies and fluctuations.

Objectives
Predict future power consumption for individual HPC nodes at 5-minute intervals

Improve energy efficiency and workload scheduling in HPC environments

Explore and compare multiple time series forecasting techniques

Dataset Description
The dataset contains node-wise power consumption values over 5 months, recorded at 5-minute intervals.

Features:

Columns: Example - 1_1-Pow-consumption, 2_11-Pow-consumption, etc.

Format: Timestamps as rows, each column representing a node's power in watts

Time Span:

February to June (~5 months)

Around 40,000 rows per node

Methodology
Preprocessing
Handled missing values using forward/backward fill

Converted string timestamps to datetime format

Normalized power values using MinMaxScaler

Created time window sequences for supervised LSTM training

Models and Techniques Tried
Technique	Outcome/Issue
ARIMA	Not suitable for multi-node time series
Basic LSTM	Underfitting; failed to capture complex patterns
GRU	Lower accuracy than LSTM on evaluation metrics
Deep LSTM	Better convergence and trend learning
LSTM Multistep	Final model; predicted 288 steps ahead (24 hours)

Final LSTM Model Architecture
Input shape: (samples, 60 time steps, 1 feature)

Architecture:

LSTM(128), Dropout(0.2)

LSTM(64), Dropout(0.2)

Dense(288)

Loss Function: Mean Squared Error (MSE)

Optimizer: Adam

Evaluation Results
Metric	Value (Node 2_11)
RMSE	~5.67
MAE	~4.32
Visual Trend	Matches ground truth well with minor peak deviations

Visualization Insights
The predicted power curve over 24 hours closely follows the actual pattern. Daily cycles are well captured, with accurate peak and drop estimations beyond 30 time steps.

Setup Instructions
bash
Copy
Edit
pip install numpy pandas matplotlib seaborn scikit-learn tensorflow
Usage
bash
Copy
Edit
# Train the model
python train_lstm.py

# Generate 24-hour future predictions
python predict_future.py
Sample Code Snippet (Appendix)
python
Copy
Edit
def create_dataset(series, look_back=60, forecast_horizon=288):
    X, y = [], []
    for i in range(len(series) - look_back - forecast_horizon):
        X.append(series[i:(i + look_back)])
        y.append(series[(i + look_back):(i + look_back + forecast_horizon)])
    return np.array(X), np.array(y)
Challenges and Fixes
ARIMA and basic LSTM did not perform well on multiple correlated time series

Introduced deeper LSTM layers with dropout to prevent overfitting

Tuned batch size, epochs, and learning rate to improve convergence

Increased input window (look-back) and output size (forecast horizon) for more stable predictions

References
HPC Dataset Source: CSIR Internal Logs

Deep Learning for Power Prediction: https://arxiv.org/abs/2106.08584

LSTM Theory: https://colah.github.io/posts/2015-08-Understanding-LSTMs/

Author
Naman Gupta
Email: namangupta@example.com

License
This project is licensed under the MIT License.
