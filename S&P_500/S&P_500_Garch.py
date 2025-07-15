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
GSPC_Prices = pdr.DataReader('^GSPC', start, end)

# Calculate Returns as a percentage change
GSPC_Prices['Return'] = 100 * (GSPC_Prices['Close'].pct_change())
# Calculate Log returns as a number
GSPC_Prices['Log_Return'] = np.log(GSPC_Prices['Close']) - np.log(GSPC_Prices['Close'].shift(1))
# Drops rows that have incomplete/NULL data
GSPC_Prices = GSPC_Prices.dropna()

# Calculate the volatility of log returns
GSPC_Daily_Vol = GSPC_Prices['Log_Return'].std()
# Round the volatility to 2 decimal places
GSPC_Daily_Vol = round(GSPC_Daily_Vol, 2)
# Calculate the volatility of normal returns
GSPC_Daily_Vol_Normal = GSPC_Prices['Return'].std()
# Round the volatility to 2 decimal places
GSPC_Daily_Vol_Normal = round(GSPC_Daily_Vol_Normal, 2)
# Basic Garch (1, 1) model created
basic_gm = arch_model(GSPC_Prices['Log_Return'], p = 1, q = 1,
                      mean = 'constant', vol = 'GARCH', dist = 'normal')

# Sets the data column index equal to the index that will be used within the rolling forecasting
index = GSPC_Prices.index
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
    # Get the result of the fitted garch model
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
plt.plot(GSPC_Prices['Log_Return'].loc[ind_list].sub(GSPC_Prices['Log_Return'].loc[ind_list].mean()).pow(2),
         color = 'grey', alpha = 0.4, label = 'Actual Daily Volatility')

# Plot EGARCH  estimated volatility
plt.plot(rolling_volatility**2, color = 'red', label = 'Forecasted GARCH Volatility')
plt.title('S&P 500 GARCH Volatility Forecasting vs Actual')
#plt.yscale('linear')
plt.legend(loc = 'upper right')
plt.show()


# Get the MAE, MAPE and RMSE of the forecasted values, method is called backtesting
def evaluate(observation, forecast):
    #Call sklearn function to calculate MAE
    mae = mean_absolute_error(observation, forecast)
    print(f'Mean Absolute Error (MAE): {round(mae,7)}')
    # Call sklearn function to calculate MAPE
    mape = mean_absolute_percentage_error(observation, forecast)
    print(f'Mean Absolute Percentage Error (MAPE): {round(mape,7)}')
    # Call sklearn function to calculate RMSE
    mse = mean_squared_error(observation, forecast)
    print(f'Root Mean Squared Error (RMSE): {round(math.sqrt(mse), 7)}')
    return mae, mape, math.sqrt(mse)

# Backtest model with MAE, MAPE and RMSE
evaluate(GSPC_Prices['Log_Return'].loc[ind_list].sub(GSPC_Prices['Log_Return'].loc[ind_list].mean()).pow(2), rolling_volatility ** 2)



