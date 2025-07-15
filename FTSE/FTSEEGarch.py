from arch.__future__ import reindexing
import pandas_datareader.data as web
from datetime import datetime, timedelta
import pandas as pd
import matplotlib.pyplot as plt
from arch import arch_model
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
import yfinance as yf
from pandas_datareader import data as pdr
import sys
from numpy import cumsum, log, polyfit, sqrt, std, subtract
from numpy.random import randn
import seaborn as sns
from pylab import rcParams
import matplotlib.cm as cm
from arch import arch_model
from numpy.linalg import LinAlgError
from scipy import stats
import statsmodels.api as sm
import statsmodels.tsa.api as tsa
from statsmodels.tsa.stattools import acf, q_stat, adfuller
from sklearn.metrics import mean_squared_error
from scipy.stats import probplot, moment
from arch.univariate import ConstantMean, GARCH, Normal
from sklearn.model_selection import TimeSeriesSplit
#from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import *
import math
#import warnings

start = pd.Timestamp('2018-10-9')
end = pd.Timestamp('2023-6-9')

# Work around to allow the DataReader package to work
yf.pdr_override()

# Get stock data for Bitcoin
FTSE_Prices = pdr.DataReader('^FTSE', start, end)

# Calculate Returns as a percentage change
FTSE_Prices['Return'] = 100 * (FTSE_Prices['Close'].pct_change())
# Calculate Log returns as a number
FTSE_Prices['Log_Return'] = np.log(FTSE_Prices['Close']) - np.log(FTSE_Prices['Close'].shift(1))
# Drops rows that have incomplete/NULL data
FTSE_Prices = FTSE_Prices.dropna()

# Calculate the volatility of log returns
FTSE_Daily_Vol = FTSE_Prices['Log_Return'].std()
# Round the volatility to 2 decimal places
FTSE_Daily_Vol = round(FTSE_Daily_Vol, 2)
# Calculate the volatility of normal returns
FTSE_Daily_Vol_Normal = FTSE_Prices['Return'].std()
# Round the volatility to 2 decimal places
FTSE_Daily_Vol_Normal = round(FTSE_Daily_Vol_Normal, 2)
# Basic eGarch (1, 1) model created
basic_gm = arch_model(FTSE_Prices['Log_Return'], p = 1, q = 1, o = 2,
                      mean = 'constant', vol = 'EGARCH', dist = 'normal')

# Sets the data column index equal to the index that will be used within the rolling forecasting
index = FTSE_Prices.index
# Beginning of the forecast
start_loc = 0
# End of the forecast
end_loc = np.where(index >= '2020-1-1')[0].min()
# Dictionary to store the forecasted log returns
forecasts = {}
# Loop that conducts the rolling forecasting
for i in range(70):
    sys.stdout.write('-')
    sys.stdout.flush()
    # Get the result of the fitted egarch model
    res = basic_gm.fit(first_obs = start_loc, last_obs = i + end_loc, disp = 'off')
    # Store the variance of the daily forecasted log returns
    temp = res.forecast(horizon=1).variance
    # Relate the forecasted log returns to the index/date
    fcast = temp.iloc[i + end_loc - 1]
    # Add the temp forecast to the forecast dictionary outside of the for loop
    forecasts[fcast.name] = fcast
print(' Done!')
# Storing the contents of the dictionary inside a pandas series
variance_expandwin = pd.DataFrame(forecasts).T
# Convert the log returns to volatility
rolling_volatility = np.sqrt(variance_expandwin)

# Get the dates that are classified as a index
ind_list = rolling_volatility.index

# Plot the actual Bitcoin volatility
plt.plot(FTSE_Prices['Log_Return'].loc[ind_list].sub(FTSE_Prices['Log_Return'].loc[ind_list].mean()).pow(2),
         color = 'grey', alpha = 0.4, label = 'Actual Daily Volatility')

# Plot EGARCH  estimated volatility
plt.plot(rolling_volatility**2, color = 'red', label = 'Forecasted EGARCH Volatility')
plt.title('FTSE EGARCH Volatility Forecasting vs Actual')
#plt.yscale('linear')
plt.legend(loc = 'upper right')
plt.show()


# Get the MAE, MAPE and RMSE of the forecasted values, method is called backtesting
def evaluate(observation, forecast):
    #Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,5)}')
    # Call sklearn function to calculate MAPE
    mape = mean_absolute_percentage_error(observation, forecast)
    print(f'Mean Absolute Percentage Error (MAPE): {round(mape,5)}')
    # Call sklearn function to calculate RMSE
    mse = mean_squared_error(observation, forecast)
    print(f'Root Mean Squared Error (RMSE): {round(math.sqrt(mse), 5)}')
    return mae, mape, math.sqrt(mse)

# Backtest model with MAE, MAPE and RMSE
evaluate(FTSE_Prices['Log_Return'].loc[ind_list].sub(FTSE_Prices['Log_Return'].loc[ind_list].mean()).pow(2), rolling_volatility ** 2)



