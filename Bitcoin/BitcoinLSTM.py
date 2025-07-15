import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import math
import datetime as dt
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.layers import LSTM
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import datetime
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import layers
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import *
import researchpy as rp



Bitcoin_Prices = pd.read_csv('BTC-USD.csv')
Bitcoin_Prices = Bitcoin_Prices[['Date', 'Close']]


def str_to_datetime(s):
    split = s.split('-')
    year, month, day = int(split[0]), int(split[1]), int(split[2])
    return datetime.datetime(year=year, month=month, day=day)

Bitcoin_Prices['Date'] = Bitcoin_Prices['Date'].apply(str_to_datetime)

Bitcoin_Prices.index = Bitcoin_Prices.pop('Date')
Bitcoin_Prices = Bitcoin_Prices.dropna()

#Plots a diagram for stock close prices
#plt.plot(FTSE_Prices.index, FTSE_Prices['Log_Return'], color= 'red', label= 'FTSE Log Returns')
#plt.plot(FTSE_Prices.index, FTSE_Prices['Close'], color= 'black', label= 'FTSE Closing Prices')
#plt.title('FTSE')
#plt.legend(loc = 'lower right')
#plt.show()

# Convert the data into displaying the current value at a time and the last 3 values in the previous days
def df_to_windowed_df(dataframe, first_date_str, last_date_str, n=3):
    first_date = str_to_datetime(first_date_str)
    last_date = str_to_datetime(last_date_str)

    target_date = first_date

    dates = []
    X, Y = [], []

    last_time = False
    while True:
        df_subset = dataframe.loc[:target_date].tail(n + 1)

        if len(df_subset) != n + 1:
            print(f'Error: Window of size {n} is too large for date {target_date}')
            return

        values = df_subset['Close'].to_numpy()
        x, y = values[:-1], values[-1]

        dates.append(target_date)
        X.append(x)
        Y.append(y)

        next_week = dataframe.loc[target_date:target_date + datetime.timedelta(days=7)]
        next_datetime_str = str(next_week.head(2).tail(1).index.values[0])
        next_date_str = next_datetime_str.split('T')[0]
        year_month_day = next_date_str.split('-')
        year, month, day = year_month_day
        next_date = datetime.datetime(day=int(day), month=int(month), year=int(year))

        if last_time:
            break

        target_date = next_date

        if target_date == last_date:
            last_time = True

    ret_df = pd.DataFrame({})
    ret_df['Target Date'] = dates

    X = np.array(X)
    for i in range(0, n):
        X[:, i]
        ret_df[f'Target-{n - i}'] = X[:, i]

    ret_df['Target'] = Y

    return ret_df


# Start day second time around: '2021-03-25'
windowed_df = df_to_windowed_df(Bitcoin_Prices,
                                '2018-10-12',
                                '2023-06-08',
                                n=3)

# Look through the data and create a window that can be used to supplement the LSTM model
def windowed_df_to_date_X_y(windowed_dataframe):
  df_as_np = windowed_dataframe.to_numpy()

  dates = df_as_np[:, 0]

  middle_matrix = df_as_np[:, 1:-1]
  X = middle_matrix.reshape((len(dates), middle_matrix.shape[1], 1))

  Y = df_as_np[:, -1]

  return dates, X.astype(np.float32), Y.astype(np.float32)

dates, X, y = windowed_df_to_date_X_y(windowed_df)

# Define the percentages of the dataset should be split
q_80 = int(len(dates) * .8)
q_90 = int(len(dates) * .9)

# Store the dates data, the original data set, the data set used to train and the dates for the data used in the train
# data set
dates_train, X_train, y_train = dates[:q_80], X[:q_80], y[:q_80]
dates_val, X_val, y_val = dates[q_80:q_90], X[q_80:q_90], y[q_80:q_90]
dates_test, X_test, y_test = dates[q_90:], X[q_90:], y[q_90:]

# Make the two dimensional array 3d
nsamples, nx, ny = X_train.shape
X_train = X_train.reshape((nsamples,nx*ny))

# Make the two dimensional array 3d
nsamples, nx, ny = X_val.shape
X_val = X_val.reshape((nsamples,nx*ny))

# Make the two dimensional array 3d
nsamples, nx, ny = X_test.shape
X_test = X_test.reshape((nsamples,nx*ny))

# Scale the data between 0, 1 to improve forecasting accuracy of LSTM model
sc = MinMaxScaler(feature_range = (0, 1))
X_train = sc.fit_transform(X_train)
X_val = sc.fit_transform(X_val)
X_test = sc.fit_transform(X_test)

#plt.plot(dates_train, y_train)
#plt.plot(dates_val, y_val)
#plt.plot(dates_test, y_test)

#plt.legend(['Train', 'Validation', 'Test'])
#plt.show()

# The sequential model used to forecast
model = Sequential([layers.Input((3, 1)),
                    layers.LSTM(32),
                    layers.Dropout(0.4),
                    layers.Dense(16, activation='relu'),
                    layers.Dropout(0.4),
                    layers.Dense(16, activation='relu'),
                    layers.Dense(1)])

#model.add(Dropout(0.01))
#model.add(Dropout(0.001))

# Currently the 4 2 2 split gives the best results
# Define the complication of the model
model.compile(loss='mse',
              optimizer=Adam(learning_rate=0.01),
              metrics=['mean_absolute_error'])

# Fit the model to the data needed to run the model
model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=200)

# Forecast train forecasts based on the train data and flatten it
train_predictions = model.predict(X_train).flatten()
# Plot the figure using the forecasts and the actual values
plt.plot(dates_train, train_predictions, color= 'red')
plt.plot(dates_train, y_train, color= 'black')
plt.title('Bitcoin LSTM Training Forecasted vs Actual Close Values')
plt.legend(['Training Predictions', 'Training Observations'])
plt.show()

# Forecast train forecasts based on the train data and flatten it
val_predictions = model.predict(X_val).flatten()
# Plot the figure using the forecasts and the actual values
plt.plot(dates_val, val_predictions, color= 'red')
plt.plot(dates_val, y_val, color= 'black')
plt.title('Bitcoin LSTM Validation Forecasted vs Actual Close Values')
plt.legend(['Validation Predictions', 'Validation Observations'])
plt.show()

# Forecast train forecasts based on the train data and flatten it
test_predictions = model.predict(X_test).flatten()
# Plot the figure using the forecasts and the actual values
plt.plot(dates_test, test_predictions, color= 'red')
plt.plot(dates_test, y_test, color= 'black')
plt.title('Bitcoin LSTM Testing Forecasted vs Actual Close Values')
plt.legend(['Testing Predictions', 'Testing Observations'])
plt.show()


def evaluate(observation, forecast, id):
    #Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'{id} Mean Absolute Error (MAE): {round(mae,3)}')
    # Call sklearn function to calculate MAPE
    mape = mean_absolute_percentage_error(observation, forecast)
    print(f'{id} Mean Absolute Percentage Error (MAPE): {round(mape,3)}')
    # Call sklearn function to calculate RMSE
    mse = mean_squared_error(observation, forecast)
    print(f'{id} Root Mean Squared Error (RMSE): {round(math.sqrt(mse), 3)}')
    return mae, mape, math.sqrt(mse)

# Use the evaluate function to generate the accuracy statistics for the train, validation and test data
evaluate(y_train, train_predictions, 'Train Data')
evaluate(y_val, val_predictions, 'Validation Data')
evaluate(y_test, test_predictions, 'Testing Data')

#print(train_predictions)

Train_Convert_Close_Prices = pd.DataFrame()
Train_Convert_Close_Prices.index = dates_train
Train_Convert_Close_Prices['Close'] = train_predictions
Train_Convert_Close_Prices['Train_Return'] = np.log(Train_Convert_Close_Prices['Close']) - np.log(Train_Convert_Close_Prices['Close'].shift(1))
#Train_Convert_Close_Prices['Train_Return'] = 100 * (Train_Convert_Close_Prices['Close'].pct_change())
Train_Convert_Close_Prices = Train_Convert_Close_Prices.dropna()

Val_Convert_Close_Prices = pd.DataFrame()
Val_Convert_Close_Prices.index = dates_val
Val_Convert_Close_Prices['Close'] = val_predictions
Val_Convert_Close_Prices['Val_Return'] = np.log(Val_Convert_Close_Prices['Close']) - np.log(Val_Convert_Close_Prices['Close'].shift(1))
#Val_Convert_Close_Prices['Val_Return'] = 100 * (Val_Convert_Close_Prices['Close'].pct_change())
Val_Convert_Close_Prices = Val_Convert_Close_Prices.dropna()

Test_Convert_Close_Prices = pd.DataFrame()
Test_Convert_Close_Prices.index = dates_test
Test_Convert_Close_Prices['Close'] = test_predictions
Test_Convert_Close_Prices['Test_Return'] = np.log(Test_Convert_Close_Prices['Close']) - np.log(Test_Convert_Close_Prices['Close'].shift(1))
#Test_Convert_Close_Prices['Test_Return'] = 100 * (Test_Convert_Close_Prices['Close'].pct_change())
Test_Convert_Close_Prices = Test_Convert_Close_Prices.dropna()

plt.plot(Train_Convert_Close_Prices['Train_Return'], color= 'red', label= 'Forecasted Bitcoin LSTM Training Data Returns')
plt.plot(Val_Convert_Close_Prices['Val_Return'], color= 'black', label= 'Forecasted Bitcoin LSTM Validation Data Returns Returns')
plt.plot(Test_Convert_Close_Prices['Test_Return'], color= 'blue', label= 'Forecasted Bitcoin LSTM Testing Data returns Returns')
plt.title('Bitcoin LSTM Forecasted Returns')
plt.legend(loc = 'lower left')
plt.show()

Train_Convert_Close_Prices['Date'] = Train_Convert_Close_Prices.index
Train_Convert_Close_Prices.to_csv('Bitcoin Train Data.csv')
Val_Convert_Close_Prices['Date'] = Val_Convert_Close_Prices.index
Val_Convert_Close_Prices.to_csv('Bitcoin Val Data.csv')
Test_Convert_Close_Prices['Date'] = Test_Convert_Close_Prices.index
Test_Convert_Close_Prices.to_csv('Bitcoin Test Data.csv')

Train_Convert_Close_Prices = pd.read_csv('Bitcoin Train Data.csv')
Train_Convert_Close_Prices = Train_Convert_Close_Prices[['Date', 'Close']]
Train_Convert_Close_Prices = Train_Convert_Close_Prices.dropna()

Val_Convert_Close_Prices = pd.read_csv('Bitcoin Val Data.csv')
Val_Convert_Close_Prices = Val_Convert_Close_Prices[['Date', 'Close']]
Val_Convert_Close_Prices = Val_Convert_Close_Prices.dropna()

Test_Convert_Close_Prices = pd.read_csv('Bitcoin Test Data.csv')
Test_Convert_Close_Prices = Test_Convert_Close_Prices[['Date', 'Close']]
Test_Convert_Close_Prices = Test_Convert_Close_Prices.dropna()

print(rp.summary_cont(Train_Convert_Close_Prices['Close']))
print(f"The Train Count is: {Train_Convert_Close_Prices['Close'].count()}")
print(f"The Train Max is: {Train_Convert_Close_Prices['Close'].max()}")
print(f"The Train Min is: {Train_Convert_Close_Prices['Close'].min()}")
print(f"The Train Kurtosis is: {Train_Convert_Close_Prices['Close'].kurtosis()}")
print(f"The Train Skew is: {Train_Convert_Close_Prices['Close'].skew()}")
print('----------------------------------------------------------------------------')
print(rp.summary_cont(Val_Convert_Close_Prices['Close']))
print(f"The Validation Count is: {Val_Convert_Close_Prices['Close'].count()}")
print(f"The Validation Max is: {Val_Convert_Close_Prices['Close'].max()}")
print(f"The Validation Min is: {Val_Convert_Close_Prices['Close'].min()}")
print(f"The Validation Kurtosis is: {Val_Convert_Close_Prices['Close'].kurtosis()}")
print(f"The Validation Skew is: {Val_Convert_Close_Prices['Close'].skew()}")
print('----------------------------------------------------------------------------')
print(rp.summary_cont(Test_Convert_Close_Prices['Close']))
print(f"The Test Count is: {Test_Convert_Close_Prices['Close'].count()}")
print(f"The Test Max is: {Test_Convert_Close_Prices['Close'].max()}")
print(f"The Test Min is: {Test_Convert_Close_Prices['Close'].min()}")
print(f"The Test Kurtosis is: {Test_Convert_Close_Prices['Close'].kurtosis()}")
print(f"The Test Skew is: {Test_Convert_Close_Prices['Close'].skew()}")
