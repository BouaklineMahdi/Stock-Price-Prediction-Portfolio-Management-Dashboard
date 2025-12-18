## Overview
An interactive stock analysis and visualization dashboard that combines historical market data, technical indicators, and machine learning-based price estimation to support exploratory portfolio analysis and monitoring.

## Problem
Individual investors and analysts often rely on fragmented tools to view price history, calculate indicators, track portfolios, and experiment with predictive models. This fragmentation makes it difficult to explore data, test assumptions, and monitor assets within a single, cohesive workflow.

This project consolidates data ingestion, analysis, visualization, and alerts into a unified application.

## Approach
### Architecture
- Data ingestion layer using live market data APIs
- Feature preprocessing and scaling pipeline for time-series data
- LSTM-based modeling component for sequence learning
- Interactive dashboard layer built with Dash and Plotly
- Notification module for price-based alerts
- In-memory portfolio tracking for user-selected assets

### Key Design Decisions
- Used Dash + Plotly to enable interactive, browser-based exploration without requiring frontend frameworks.
- Implemented LSTM models to experiment with sequential pattern learning on historical closing prices.
- Included technical indicators (moving averages, volume) to contextualize model outputs rather than relying solely on predictions.
- Designed the system to retrain models dynamically per selected asset to keep the pipeline simple and transparent.

### Tradeoffs
- Model training occurs at runtime, increasing latency for simplicity and clarity.
- Portfolio state is stored in memory rather than persistent storage.
- The ML component is exploratory and not optimized for predictive accuracy or production deployment.

## Tech Stack
- **Languages:** Python  
- **Data:** yfinance, pandas, NumPy  
- **Machine Learning:** TensorFlow / Keras (LSTM)  
- **Visualization:** Plotly, Dash, Dash Bootstrap Components  
- **Utilities:** scikit-learn, SMTP (email alerts)

## Results / Output
The system produces:
- Interactive candlestick charts with volume and moving averages
- Short-horizon price estimates based on recent historical data
- A simple portfolio tracker for selected tickers
- Email alerts triggered by user interaction

Outputs are intended for exploratory analysis, visualization, and experimentation rather than trading execution.

## Limitations & Next Steps
**Limitations**
- The LSTM model is trained on limited historical data and does not account for fundamentals or exogenous factors.
- No backtesting framework or performance validation pipeline is included.
- Portfolio tracking is session-based and non-persistent.
- Email credentials and alert logic are not production-hardened.

**Next Steps**
- Separate model training from the UI layer and cache trained models.
- Add backtesting and error analysis to evaluate model behavior over time.
- Incorporate additional features (returns, volatility, macro signals).
- Replace in-memory portfolio storage with a database-backed solution.
- Refactor alerting into a secure, configurable notification service.
