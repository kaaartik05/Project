import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def stock_price_prediction(ticker, start_date, end_date, future_days):
    try:
        # Download stock data
        stock_data = yf.download(ticker, start=start_date, end=end_date)
        if stock_data.empty:
            print("No data found for the given ticker symbol and date range.")
            return None

        stock_data = stock_data[['Close']]
        stock_data['Days'] = np.arange(len(stock_data))

        # Prepare data for training
        X = stock_data[['Days']].values
        y = stock_data['Close'].values
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the model
        model = LinearRegression()
        model.fit(X_train, y_train)

        # Make predictions
        predictions = model.predict(X_test)

        # Predict future prices
        future_X = np.array(range(len(stock_data), len(stock_data) + future_days)).reshape(-1, 1)
        future_predictions = model.predict(future_X)

        # Plotting
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

    except Exception as e:
        print(f"An error occurred: {e}")
        return None

# Set parameters based on user input
ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")
try:
    future_days = int(input("Enter the number of future days to predict: "))
except ValueError:
    print("Please enter a valid integer for future days.")
    future_days = 0

predicted_prices = stock_price_prediction(ticker, start_date, end_date, future_days)
if predicted_prices is not None:
    print("Predicted Future Prices:", predicted_prices)
