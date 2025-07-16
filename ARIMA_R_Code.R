library(quantmod)
install.packages("dplyr")
library(dplyr)
library(quantmod)
install.packages("tseries")
library(tseries)
install.packages("timeSeries")
library(timeSeries)
install.packages("forecast")
library(forecast)
install.packages("xts")
library(xts)
install.packages("rugarch")
library(rugarch)

# Clears R memory
rm(list = ls())

# Define list of tickers and get historical data between the dates mentioned
tickers = c("^FTSE", "^GSPC", "^DJI", "^IXIC", "^GDAXI", "BTC-USD", "ETH-USD", "USDT-USD", "BNB-USD", "USDC-USD")
getSymbols(tickers, from="2018-10-08", to="2023-06-09")[,5]

# Checks for missing values
colSums(is.na(FTSE))
colSums(is.na(GDAXI))

# Removes all rows with missing values
clean_FTSE <- na.omit(FTSE)
clean_GDAXI <- na.omit(GDAXI)

colSums(is.na(clean_GDAXI))
colSums(is.na(clean_FTSE))

FTSE_Close_Prices = clean_FTSE[,6]
colSums(is.na(FTSE_Close_Prices))
FTSE_Close_Prices <- na.omit(FTSE_Close_Prices)
colSums(is.na(FTSE_Close_Prices))

GDAXI_Close_Prices = clean_GDAXI[,6]
colSums(is.na(GDAXI_Close_Prices))

# Extracts the close price column in the historical price data
BNB_USD_Close_Prices = `BNB-USD`[,6]
BTC_USD_Close_Prices = `BTC-USD`[,6]
DJI_Close_Prices = DJI[,6]
ETH_USD_Close_Prices = `ETH-USD`[,6]
GSPC_Close_Prices = GSPC[,6]
IXIC_Close_Prices = IXIC[,6]
USDC_USD_Close_Prices = `USDC-USD`[,6]
USDT_USD_Close_Prices = `USDT-USD`[,6]

# Checks tickers closing price data for missing values
colSums(is.na(BNB_USD_Close_Prices))
colSums(is.na(BTC_USD_Close_Prices))
colSums(is.na(DJI_Close_Prices))
colSums(is.na(ETH_USD_Close_Prices))
colSums(is.na(GSPC_Close_Prices))
colSums(is.na(IXIC_Close_Prices))
colSums(is.na(USDC_USD_Close_Prices))
colSums(is.na(USDT_USD_Close_Prices))
colSums(is.na(FTSE_Close_Prices))
colSums(is.na(GDAXI_Close_Prices))

# Removing the original datasets, keeping only the close price vectors
rm(`BNB-USD`)
rm(`BTC-USD`)
rm(DJI)
rm(`ETH-USD`)
rm(GSPC)
rm(IXIC)
rm(`USDC-USD`)
rm(`USDT-USD`)
rm(FTSE)
rm(GDAXI)
rm(clean_FTSE)
rm(clean_GDAXI)


###========== BTC ==========###
auto.arima(BTC_Log_Returns, seasonal=FALSE)
BTC_Log_Returns = diff(log(BTC_USD_Close_Prices), lag=1)
BTC_Log_Returns = na.omit(BTC_Log_Returns)
BTC_Breakpoint = floor(nrow(BTC_Log_Returns)*(2.95/3))

# Initialzing an xts object for Actual log returns
BTC_Actual_Series = xts(0,as.Date("2018-10-09","%Y-%m-%d"))
# Initialzing a dataframe for the forecasted return series
BTC_Forecasted_Series = data.frame(Forecasted = numeric())

for (b in BTC_Breakpoint:(nrow(BTC_Log_Returns)-1)) {
  
  btc_stock_train = BTC_Log_Returns[1:b, ]
  btc_stock_test = BTC_Log_Returns[(b+1):nrow(BTC_Log_Returns), ]
  # Summary of the ARIMA model using the determined (p,d,q) parameters
  btc_fit = arima(btc_stock_train, order = c(0, 0, 2),include.mean=FALSE)
  summary(btc_fit)
  # plotting a acf plot of the residuals
  acf(btc_fit$residuals,main="Bitcoin Residuals plot")
  pacf(btc_fit$residuals,main="Bitcoin Residuals plot")
  #print(adf.test(BTC_Log_Returns))
  # Forecasting the log returns
  btc_arima.forecast = forecast(btc_fit, h = 1,level=99)
  summary(btc_arima.forecast)
  # plotting the forecast
  par(mfrow=c(1,1))
  plot(btc_arima.forecast, main = "Bitcoin ARIMA Forecast")
  # Creating a series of forecasted returns for the forecasted period
  BTC_Forecasted_Series = rbind(BTC_Forecasted_Series,btc_arima.forecast$mean[1])
  colnames(BTC_Forecasted_Series) = c("Forecasted")
  # Creating a series of actual returns for the forecasted period
  BTC_Actual_Return = BTC_Log_Returns[(b+1),]
  BTC_Actual_Series = c(BTC_Actual_Series,xts(BTC_Actual_Return))
  rm(BTC_Actual_Return)
  
  print(BTC_USD_Close_Prices[(b+1),])
  print(BTC_USD_Close_Prices[(b+2),])
  accuracy(btc_arima.forecast)
}

# Adjust the length of the Actual return series
BTC_Actual_Series = BTC_Actual_Series[-1]
# Create a time series object of the forecasted series
BTC_Forecasted_Series = xts(BTC_Forecasted_Series,index(BTC_Actual_Series))
# Create a plot of the two return series - Actual versus Forecasted
plot(BTC_Actual_Series,type='l',main='Bitcoin Actual Returns Vs ARIMA Forecasted Returns')
lines(BTC_Forecasted_Series,lwd=1.5,col='red')
legend('topleft', legend = c("Actual","Forecasted"),lty=1,lwd=1,col=c('black','red'))
# Create a table for the accuracy of the forecast
btc_comparsion = merge(BTC_Actual_Series,BTC_Forecasted_Series)
btc_comparsion$Accuracy = sign(btc_comparsion$BTC_Actual_Series)==sign(btc_comparsion$Forecasted)
print(btc_comparsion)
# Compute the accuracy percentage metric
BTC_Accuracy_percentage = sum(btc_comparsion$Accuracy == 1)*100/length(btc_comparsion$Accuracy)
print(BTC_Accuracy_percentage)
BTC_PM = accuracy(btc_arima.forecast)
accuracy(btc_arima.forecast)
print(BTC_PM)


###=========== ETH ==========###
ETH_Log_Returns = diff(log(ETH_USD_Close_Prices), lag=1)
ETH_Log_Returns = na.omit(ETH_Log_Returns)
auto.arima(ETH_Log_Returns, seasonal=FALSE)
ETH_Breakpoint = floor(nrow(ETH_Log_Returns)*(2.95/3))

# Initialzing an xts object for Actual log returns
ETH_Actual_Series = xts(0,as.Date("2018-10-09","%Y-%m-%d"))
# Initialzing a dataframe for the forecasted return series
ETH_Forecasted_Series = data.frame(Forecasted = numeric())

for (c in ETH_Breakpoint:(nrow(ETH_Log_Returns)-1)) {
  
  eth_stock_train = ETH_Log_Returns[1:c, ]
  eth_stock_test = ETH_Log_Returns[(c+1):nrow(ETH_Log_Returns), ]
  # Summary of the ARIMA model using the determined (p,d,q) parameters
  eth_fit = arima(eth_stock_train, order = c(1, 0, 1),include.mean=FALSE)
  summary(eth_fit)
  # plotting a acf plot of the residuals
  acf(eth_fit$residuals,main="Ethereum Residuals plot")
  pacf(eth_fit$residuals,main="Ethereum Residuals plot")
  #print(adf.test(ETH_Log_Returns))
  # Forecasting the log returns
  eth_arima.forecast = forecast(eth_fit, h = 1,level=99)
  summary(eth_arima.forecast)
  # plotting the forecast
  par(mfrow=c(1,1))
  plot(eth_arima.forecast, main = "Ethereum ARIMA Forecast")
  # Creating a series of forecasted returns for the forecasted period
  ETH_Forecasted_Series = rbind(ETH_Forecasted_Series,eth_arima.forecast$mean[1])
  colnames(ETH_Forecasted_Series) = c("Forecasted")
  # Creating a series of actual returns for the forecasted period
  ETH_Actual_Return = ETH_Log_Returns[(c+1),]
  ETH_Actual_Series = c(ETH_Actual_Series,xts(ETH_Actual_Return))
  rm(ETH_Actual_Return)
  
  print(ETH_USD_Close_Prices[(c+1),])
  print(ETH_USD_Close_Prices[(c+2),])
  accuracy(eth_arima.forecast)
}

# Adjust the length of the Actual return series
ETH_Actual_Series = ETH_Actual_Series[-1]
# Create a time series object of the forecasted series
ETH_Forecasted_Series = xts(ETH_Forecasted_Series,index(ETH_Actual_Series))
# Create a plot of the two return series - Actual versus Forecasted
plot(ETH_Actual_Series,type='l',main='Ethereum Actual Returns Vs Forecasted Returns')
lines(ETH_Forecasted_Series,lwd=1.5,col='red')
legend('bottomright',c("Actual","Forecasted"),lty=c(1,1),lwd=c(1.5,1.5),col=c('black','red'))
# Create a table for the accuracy of the forecast
ETH_comparsion = merge(ETH_Actual_Series,ETH_Forecasted_Series)
ETH_comparsion$Accuracy = sign(ETH_comparsion$ETH_Actual_Series)==sign(ETH_comparsion$Forecasted)
print(ETH_comparsion)
# Compute the accuracy percentage metric
ETH_Accuracy_percentage = sum(ETH_comparsion$Accuracy == 1)*100/length(ETH_comparsion$Accuracy)
print(ETH_Accuracy_percentage)
ETH_PM = accuracy(eth_arima.forecast)
accuracy(eth_arima.forecast)
print(ETH_PM)


###=========== FTSE ==========###
FTSE_Log_Returns = diff(log(FTSE_Close_Prices), lag=1)
FTSE_Log_Returns = na.omit(FTSE_Log_Returns)
auto.arima(FTSE_Log_Returns, seasonal=FALSE)
FTSE_Breakpoint = floor(nrow(FTSE_Log_Returns)*(2.95/3))

# Initialzing an xts object for Actual log returns
FTSE_Actual_Series = xts(0,as.Date("2018-10-09","%Y-%m-%d"))
# Initialzing a dataframe for the forecasted return series
FTSE_Forecasted_Series = data.frame(Forecasted = numeric())

for (c in FTSE_Breakpoint:(nrow(FTSE_Log_Returns)-1)) {
  
  FTSE_stock_train = FTSE_Log_Returns[1:c, ]
  FTSE_stock_test = FTSE_Log_Returns[(c+1):nrow(FTSE_Log_Returns), ]
  # Summary of the ARIMA model using the determined (p,d,q) parameters
  FTSE_fit = arima(FTSE_stock_train, order = c(4, 0, 1),include.mean=FALSE)
  summary(FTSE_fit)
  # plotting a acf plot of the residuals
  acf(FTSE_fit$residuals,main="FTSE Residuals plot")
  pacf(FTSE_fit$residuals,main="FTSE Residuals plot")
  #print(adf.test(FTSE_Log_Returns))
  # Forecasting the log returns
  FTSE_arima.forecast = forecast(FTSE_fit, h = 1,level=99)
  summary(FTSE_arima.forecast)
  # plotting the forecast
  par(mfrow=c(1,1))
  plot(FTSE_arima.forecast, main = "FTSE ARIMA Forecast")
  # Creating a series of forecasted returns for the forecasted period
  FTSE_Forecasted_Series = rbind(FTSE_Forecasted_Series,FTSE_arima.forecast$mean[1])
  colnames(FTSE_Forecasted_Series) = c("Forecasted")
  # Creating a series of actual returns for the forecasted period
  FTSE_Actual_Return = FTSE_Log_Returns[(c+1),]
  FTSE_Actual_Series = c(FTSE_Actual_Series,xts(FTSE_Actual_Return))
  rm(FTSE_Actual_Return)
  
  print(FTSE_Close_Prices[(c+1),])
  print(FTSE_Close_Prices[(c+2),])
  accuracy(FTSE_arima.forecast)
}

# Adjust the length of the Actual return series
FTSE_Actual_Series = FTSE_Actual_Series[-1]
# Create a time series object of the forecasted series
FTSE_Forecasted_Series = xts(FTSE_Forecasted_Series,index(FTSE_Actual_Series))
# Create a plot of the two return series - Actual versus Forecasted
plot(FTSE_Actual_Series,type='l',main='FTSE Actual Returns Vs Forecasted Returns')
lines(FTSE_Forecasted_Series,lwd=1.5,col='red')
legend('bottomright',legend=c("Actual","Forecasted"),lty=c(1,1),lwd=c(1.5,1.5),col=c('black','red'))
legend('bottomright', legend=c("Actual", "Forecasted"), col=c("red", "black"), lty=1:2, cex=0.8)
# Create a table for the accuracy of the forecast
FTSE_comparsion = merge(FTSE_Actual_Series,FTSE_Forecasted_Series)
FTSE_comparsion$Accuracy = sign(FTSE_comparsion$FTSE_Actual_Series)==sign(FTSE_comparsion$Forecasted)
print(FTSE_comparsion)
# Compute the accuracy percentage metric
FTSE_Accuracy_percentage = sum(FTSE_comparsion$Accuracy == 1)*100/length(FTSE_comparsion$Accuracy)
print(FTSE_Accuracy_percentage)
FTSE_PM = accuracy(FTSE_arima.forecast)
print(FTSE_PM)


###=========== GSPC(S&P 500) ==========###
GSPC_Log_Returns = diff(log(GSPC_Close_Prices), lag=1)
GSPC_Log_Returns = na.omit(GSPC_Log_Returns)
auto.arima(GSPC_Log_Returns, seasonal=FALSE)
GSPC_Breakpoint = floor(nrow(GSPC_Log_Returns)*(2.95/3))

# Initialzing an xts object for Actual log returns
GSPC_Actual_Series = xts(0,as.Date("2018-10-09","%Y-%m-%d"))
# Initialzing a dataframe for the forecasted return series
GSPC_Forecasted_Series = data.frame(Forecasted = numeric())

for (c in GSPC_Breakpoint:(nrow(GSPC_Log_Returns)-1)) {
  
  GSPC_stock_train = GSPC_Log_Returns[1:c, ]
  GSPC_stock_test = GSPC_Log_Returns[(c+1):nrow(GSPC_Log_Returns), ]
  # Summary of the ARIMA model using the determined (p,d,q) parameters
  GSPC_fit = arima(GSPC_stock_train, order = c(4, 0, 2),include.mean=FALSE)
  summary(GSPC_fit)
  # plotting a acf plot of the residuals
  acf(GSPC_fit$residuals,main="S&P 500 Residuals plot")
  pacf(GSPC_fit$residuals,main="S&P 500 Residuals plot")
  #print(adf.test(GSPC_Log_Returns))
  # Forecasting the log returns
  GSPC_arima.forecast = forecast(GSPC_fit, h = 1,level=99)
  summary(GSPC_arima.forecast)
  # plotting the forecast
  par(mfrow=c(1,1))
  plot(GSPC_arima.forecast, main = "S&P 500 ARIMA Forecast")
  # Creating a series of forecasted returns for the forecasted period
  GSPC_Forecasted_Series = rbind(GSPC_Forecasted_Series,GSPC_arima.forecast$mean[1])
  colnames(GSPC_Forecasted_Series) = c("Forecasted")
  # Creating a series of actual returns for the forecasted period
  GSPC_Actual_Return = GSPC_Log_Returns[(c+1),]
  GSPC_Actual_Series = c(GSPC_Actual_Series,xts(GSPC_Actual_Return))
  rm(GSPC_Actual_Return)
  
  print(GSPC_Close_Prices[(c+1),])
  print(GSPC_Close_Prices[(c+2),])
  accuracy(GSPC_arima.forecast)
}

# Adjust the length of the Actual return series
GSPC_Actual_Series = GSPC_Actual_Series[-1]
# Create a time series object of the forecasted series
GSPC_Forecasted_Series = xts(GSPC_Forecasted_Series,index(GSPC_Actual_Series))
# Create a plot of the two return series - Actual versus Forecasted
plot(GSPC_Actual_Series,type='l',main='S&P 500 Actual Returns Vs Forecasted Returns')
lines(GSPC_Forecasted_Series,lwd=1.5,col='red')
legend('bottomright',c("Actual","Forecasted"),lty=c(1,1),lwd=c(1.5,1.5),col=c('black','red'))
# Create a table for the accuracy of the forecast
GSPC_comparsion = merge(GSPC_Actual_Series,GSPC_Forecasted_Series)
GSPC_comparsion$Accuracy = sign(GSPC_comparsion$GSPC_Actual_Series)==sign(GSPC_comparsion$Forecasted)
print(GSPC_comparsion)
# Compute the accuracy percentage metric
GSPC_Accuracy_percentage = sum(GSPC_comparsion$Accuracy == 1)*100/length(GSPC_comparsion$Accuracy)
print(GSPC_Accuracy_percentage)
GSPC_PM = accuracy(GSPC_arima.forecast)
print(GSPC_PM)
