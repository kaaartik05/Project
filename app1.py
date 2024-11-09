
# Import necessary libraries
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from datetime import timedelta

# Function to get stock data and make predictions
def stock_price_prediction(ticker, start_date, end_date, future_days):
    # Download stock data
    print(f"Fetching data for {ticker} from {start_date} to {end_date}...")
    stock_data = yf.download(ticker, start=start_date, end=end_date)

    # Check if data is empty
    if stock_data.empty:
        print(f"No data available for {ticker} in the specified date range.")
        return None

    # If there's not enough data to split, use all for training
    if len(stock_data) <= 1:
        print("Not enough data for training and testing.")
        return None

    # Prepare the stock data and features
    stock_data = stock_data[['Close']]
    stock_data['Days'] = np.arange(len(stock_data))

    X = stock_data[['Days']].values
    y = stock_data['Close'].values

    # If there is enough data, perform a train-test split
    if len(stock_data) > 10:  # Make sure there are enough data points
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    else:
        X_train, y_train = X, y  # If too little data, skip split

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Predict on test data (if available)
    if len(X_test) > 0:
        predictions = model.predict(X_test)
        mae = mean_absolute_error(y_test, predictions)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"Model Evaluation - MAE: {mae:.2f}, RMSE: {rmse:.2f}")

    # Predict future prices
    future_X = np.array(range(len(stock_data), len(stock_data) + future_days)).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # Ensure the future_dates and future_predictions have the same length
    last_date = stock_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    # Check if lengths match
    if len(future_dates) != len(future_predictions):
        print("Mismatch between future dates and predictions!")
        return None

    # Display future predictions
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions})

    # Plot the historical and predicted prices
    plt.figure(figsize=(14, 8))
    plt.plot(stock_data['Days'], stock_data['Close'], label="Historical Prices", color="blue", linewidth=2)
    plt.scatter(X_test, predictions, color='red', label="Test Predictions", marker='x', s=50)
    future_days_range = range(len(stock_data), len(stock_data) + future_days)
    plt.plot(future_days_range, future_predictions, color='orange', linestyle='--', linewidth=2, label="Future Predictions")
    plt.axvline(x=len(stock_data)-1, color='gray', linestyle='--', linewidth=1)
    plt.text(len(stock_data)-1, stock_data['Close'].iloc[-1], ' Future Prediction Start', color='black', verticalalignment='bottom')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title(f"Detailed Stock Price Prediction for {ticker}")
    plt.legend()
    plt.show()

    return future_df

# Set parameters
ticker = "AAPL"  # Example: Apple Inc.
start_date = "2020-01-01"
end_date = "2023-01-01"
future_days = 30  # Predict 30 days into the future

# Run the prediction function
future_df = stock_price_prediction(ticker, start_date, end_date, future_days)

# Display future predictions
if future_df is not None:
    print("\nFuture Price Predictions:")
    print(future_df)
