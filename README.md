# Financial Market Forecasting: Time Series vs Artificial Neural Networks

## Project Background

This project is a data-driven analysis undertaken by a data analyst working in a quantitative research capacity at a financial analytics firm. The firm focuses on applying machine learning and econometric forecasting models to support investment decisions across stock and cryptocurrency markets. The business operates in the financial services and fintech sector, offering AI-powered market analysis tools to clients.

The analysis compares the effectiveness of traditional time series models (ARIMA, GARCH, EGARCH) against artificial neural networks (LSTM) for forecasting asset returns and volatility using historical data from 2018–2023. It aims to determine which forecasting approach offers superior accuracy and insight across asset classes, helping inform product development, portfolio allocation strategies, and risk assessments.

Insights and recommendations are provided on the following key areas:

- Return Forecasting Accuracy
- Volatility Forecasting Models
- Model Comparison (Traditional vs ANN)
- Asset Class Performance (Stocks vs Crypto)

- 📂 The SQL queries used to inspect and clean the data for this analysis can be found here: **[data/sql/cleaning_queries.sql](https://github.com/Davaition/dissertation_prod/tree/main/data/sql)**  
- 📂 Targeted SQL queries regarding various business questions can be found here: **[data/sql/business_queries.sql](https://github.com/Davaition/dissertation_prod/tree/main/data/sql)**  
- 📊 An interactive Tableau dashboard used to report and explore sales trends can be found here: **[Tableau Dashboard Link Placeholder]**

## Data Structure & Initial Checks

The main dataset used for analysis was pulled from **Yahoo Finance** and consisted of daily close price data for four key assets:

- **FTSE 100 (Stock Index)**
- **S&P 500 (Stock Index)**
- **Bitcoin**
- **Ethereum**

**Database Table Overview**  
The project architecture reflects four main tables, cleaned and structured into a relational format:

- **closing_prices** – historical asset closing prices  
- **returns_log** – daily log returns for each asset  
- **forecast_results** – performance metrics for ARIMA, LSTM, GARCH, EGARCH models  
- **model_configurations** – hyperparameters and configurations used for each forecasting model

**Row Count**: ~1250–1800 records per asset.

[📊 Entity Relationship Diagram – Placeholder for ERD Image]

## Executive Summary

### Overview of Findings

As a financial data analyst, the key takeaway from this project is that traditional time series models still outperform neural network models when used with limited financial data, especially over shorter time horizons. Stocks were found to yield more accurate forecasts than cryptocurrencies across all models, and the ARIMA and EGARCH models offered superior performance in return and volatility forecasting, respectively.

**Key Takeaways for Stakeholders (e.g. Investment Strategists):**

1. **ARIMA significantly outperforms LSTM** in return forecasting.
2. **EGARCH captures volatility better than GARCH**, especially with asymmetric shocks.
3. **Stocks (FTSE & S&P 500) offer better forecast stability** compared to highly volatile cryptocurrencies.

![Forecast Summary Trend](https://github.com/Davaition/dissertation_prod/blob/main/images/summary_forecast_trend.png)

## Insights Deep Dive

### Category 1: Return Forecasting (ARIMA vs LSTM)

- **Insight 1**: ARIMA forecasts for Bitcoin achieved RMSE of 0.037 vs LSTM’s 9755 on testing data.
- **Insight 2**: ARIMA outperformed across all assets, showing 3.4x better performance than LSTM for short-term forecasts.
- **Insight 3**: LSTM models suffered from overfitting—high training accuracy but poor validation accuracy.
- **Insight 4**: S&P 500 and FTSE return forecasts had the lowest MAE and RMSE scores under ARIMA.

📉 *[Return Forecast Visual – ARIMA vs LSTM]*

### Category 2: Volatility Forecasting (GARCH vs EGARCH)

- **Insight 1**: EGARCH model provided marginally better RMSE values than GARCH across all assets.
- **Insight 2**: EGARCH captured leverage effects (asymmetric shocks) effectively, improving accuracy.
- **Insight 3**: FTSE and S&P 500 volatility forecasts had RMSE as low as 0.00005 under EGARCH.
- **Insight 4**: Ethereum produced the best volatility forecasts among cryptocurrencies, contrary to expectations.

📉 *[Volatility Forecast Comparison Chart]*

### Category 3: Asset Class Comparison

- **Insight 1**: Stocks produced significantly better forecasts than cryptocurrencies in all models.
- **Insight 2**: Cryptocurrency volatility was harder to predict due to rapid market shifts.
- **Insight 3**: Stocks had consistent MAE < 0.01, while crypto MAE exceeded 0.02–0.03 in several models.
- **Insight 4**: Historical stability in stock indices benefits forecasting models greatly.

📊 *[Stocks vs Crypto Forecast Accuracy Table]*

### Category 4: Forecasting Model Efficiency

- **Insight 1**: ARIMA offers faster computation and better short-term performance.
- **Insight 2**: LSTM requires longer training time and larger datasets for meaningful results.
- **Insight 3**: EGARCH is more robust under volatile conditions due to handling of asymmetry.
- **Insight 4**: LSTM’s effectiveness may improve with longer time horizons or feature-rich data (e.g. macroeconomics).

📊 *[Model Comparison Matrix: RMSE / MAPE by Asset]*

## Recommendations

For the **Financial Data Science & Strategy Team**, we recommend:

1. **Continue using ARIMA for short-term return forecasting** in financial indices like FTSE and S&P 500.
2. **Apply EGARCH models for risk assessments**, especially in portfolios exposed to volatility spikes.
3. **Avoid deploying LSTM models without extensive data or ensemble enhancements**.
4. **Invest in collecting macroeconomic indicators** to improve forecasting models (especially for crypto).
5. **Prioritize stock-based forecasts** for lower error and more reliable signal generation.

## Assumptions and Caveats

- **Assumption 1**: Log returns were used to normalise distributions and improve model performance.
- **Assumption 2**: Missing values and outliers were cleaned prior to modeling using ADF tests and differencing.
- **Assumption 3**: Macroeconomic indicators were not included due to dataset limitations.
- **Assumption 4**: Cryptocurrency data (2018–2023) was treated independently from external sentiment or policy shocks.
- **Assumption 5**: All forecasts were evaluated using RMSE, MAPE, and MAE consistently across models.
