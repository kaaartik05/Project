# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# Function to get stock data and make predictions
def stock_price_prediction(ticker, start_date, end_date, future_days):
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Close']]
    stock_data['Days'] = np.arange(len(stock_data))

    # Prepare data for model
    X = stock_data[['Days']].values
    y = stock_data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data
    predictions = model.predict(X_test)

    # Predict future prices
    future_X = np.array(range(len(stock_data), len(stock_data) + future_days)).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.plot(stock_data['Days'], stock_data['Close'], label="Historical Prices", color="blue")
    plt.scatter(X_test, predictions, color='red', label="Test Predictions", marker='x')
    plt.plot(range(len(stock_data), len(stock_data) + future_days), future_predictions, color='orange', label="Future Predictions")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Prediction for {ticker}")
    plt.legend()
    plt.show()

    return future_predictions

# Set parameters
ticker = "AAPL"  # Example: Apple Inc.
start_date = "2020-01-01"
end_date = "2023-01-01"
future_days = 30  # Predict 30 days into the future

# Run the prediction function
predicted_prices = stock_price_prediction(ticker, start_date, end_date, future_days)
print("Predicted Future Prices:", predicted_prices)
