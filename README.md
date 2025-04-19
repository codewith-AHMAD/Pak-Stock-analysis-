# Stock Market Closing Price Prediction (Pakistan, 2008-2021)

[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
![Python](https://img.shields.io/badge/python-3.x-blue.svg)
![TensorFlow](https://img.shields.io/badge/TensorFlow-%E2%80%8F2.x-orange.svg)
![Keras](https://img.shields.io/badge/Keras-%E2%80%8F2.x-red.svg)
![NumPy](https://img.shields.io/badge/NumPy-1.x-green.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.x-purple.svg)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.x-brightgreen.svg)
![Matplotlib](https://img.shields.io/badge/Matplotlib-3.x-blueviolet.svg)
![Seaborn](https://img.shields.io/badge/Seaborn-0.x-lightcoral.svg)

## Overview

This project focuses on predicting the closing price of the Pakistani stock market using historical data from 2008 to 2021. A Long Short-Term Memory (LSTM) neural network model was employed to capture the temporal dependencies within the stock price data. The project includes data preprocessing, exploratory data visualization, model building, training, and evaluation.

## Project Goals

* To develop a robust model capable of predicting the closing price of the Pakistani stock market.
* To understand the application of LSTM networks in financial time series forecasting.
* To implement standard data preprocessing and visualization techniques for stock market analysis.

## Repository Contents

* `stock_analysis.ipynb`: Jupyter Notebook containing the complete code for data loading, preprocessing, visualization, model building, training, and evaluation.
* `data/`: Directory to store the stock market dataset (e.g., `stock_pak.csv`). **Note:** The actual data file is not included in this repository for privacy/size reasons. You will need to provide your own dataset in this format.
* `requirements.txt`: List of Python libraries required to run the notebook.

## Technical Details

### Long Short-Term Memory (LSTM) Model

The core of this project is an LSTM neural network. LSTMs are a type of Recurrent Neural Network (RNN) capable of learning long-term dependencies in sequential data. This makes them particularly well-suited for time series forecasting tasks like stock price prediction.

The model architecture used is as follows:

model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(LSTM(units=50))
model.add(Dense(1))

LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])): This is the first LSTM layer with 50 memory units. return_sequences=True indicates that this layer will output a sequence of values for each input time step, which is necessary for the subsequent LSTM layer. The input_shape specifies the shape of the input data for each time step (number of look-back days, number of features).

### LSTM(units=50): 
This is the second LSTM layer, also with 50 memory units. Since return_sequences is False (by default), this layer will output only the last output in the sequence.

### Dense(1): 
This is a dense (fully connected) output layer with a single neuron, which outputs the predicted closing price.

## Look-Back Period
The look_back parameter is a crucial concept in time series forecasting with models like LSTMs. It defines the number of past data points (in this case, the number of previous trading days) that the model considers when making a prediction for the next time step.
In this project, the look_back period was set to 60

This means that for each prediction, the LSTM model looks at the closing prices and potentially other features from the preceding 60 trading days to learn patterns and make an informed forecast for the subsequent day's closing price.

#### The data was prepared using a sliding window approach based on this look_back value:
X_list = []
y_list = []
for i in range(len(X) - look_back - 1):
    X_list.append(X.iloc[i:(i+look_back)].values)
    y_list.append(y.iloc[i+look_back])

## Results
The trained LSTM model achieved the following performance metrics on the test set:

Mean Absolute Error (MAE): 0.0072
Mean Squared Error (MSE): 0.00011
Root Mean Squared Error (RMSE): 0.0105
R-squared: 0.9987
Additionally, the Direction Accuracy of the model was found to be 0.9905, indicating a high accuracy in predicting the direction (up or down) of the closing price movement.

These metrics suggest that the LSTM model performed exceptionally well in capturing the patterns in the historical stock market data and making accurate predictions.

## Setup and Installation
git clone [https://github.com/your_username/your_repo_name.git](https://github.com/your_username/your_repo_name.git)
cd your_repo_name

## Install the required Python libraries:
pip install -r requirements.txt

You can generate the requirements.txt file using:
pip freeze > requirements.txt

## License
This project is licensed under the MIT License.

## Acknowledgements
CHISEL
Zeeshan-ul-hassan Usmani
