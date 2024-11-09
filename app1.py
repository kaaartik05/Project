# Import necessary libraries
import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
from datetime import timedelta

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

    # Model evaluation
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))

    # Predict future prices
    future_X = np.array(range(len(stock_data), len(stock_data) + future_days)).reshape(-1, 1)
    future_predictions = model.predict(future_X)

    # Generate future dates
    last_date = stock_data.index[-1]
    future_dates = [last_date + timedelta(days=i) for i in range(1, future_days + 1)]

    # Display future predictions with dates
    future_df = pd.DataFrame({'Date': future_dates, 'Predicted Close Price': future_predictions})

    # Plot results
    plt.figure(figsize=(14, 8))
    # Plot historical prices
    plt.plot(stock_data['Days'], stock_data['Close'], label="Historical Prices", color="blue", linewidth=2)
    # Plot test predictions
    plt.scatter(X_test, predictions, color='red', label="Test Predictions", marker='x', s=50)
    # Plot future predictions
    future_days_range = range(len(stock_data), len(stock_data) + future_days)
    plt.plot(future_days_range, future_predictions, color='orange', linestyle='--', linewidth=2, label="Future Predictions")
    # Highlight transition point
    plt.axvline(x=len(stock_data)-1, color='gray', linestyle='--', linewidth=1)
    plt.text(len(stock_data)-1, stock_data['Close'].iloc[-1], ' Future Prediction Start', color='black', verticalalignment='bottom')
    # Add a grid and labels
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title(f"Detailed Stock Price Prediction for {ticker}")
    plt.legend()

    # Return metrics, plot, and future predictions DataFrame
    return mae, rmse, plt, future_df

# Streamlit App
st.title("Stock Price Prediction App")
st.write("This app predicts future stock prices based on historical data using linear regression.")

# User inputs
ticker = st.text_input("Enter Stock Ticker Symbol (e.g., AAPL):", value="AAPL")
start_date = st.date_input("Start Date", value=pd.to_datetime("2020-01-01"))
end_date = st.date_input("End Date", value=pd.to_datetime("2023-01-01"))
future_days = st.slider("Number of Days to Predict into the Future:", min_value=1, max_value=60, value=30)

# Run prediction if the ticker is entered
if ticker:
    st.write(f"Fetching data for {ticker} from {start_date} to {end_date}...")

    # Run the prediction function
    try:
        mae, rmse, plot, future_df = stock_price_prediction(ticker, start_date, end_date, future_days)

        # Display metrics
        st.subheader("Model Performance on Test Set")
        st.write(f"Mean Absolute Error (MAE): {mae:.2f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.2f}")

        # Display the plot
        st.subheader("Historical Prices and Future Predictions")
        st.pyplot(plot)

        # Display future predictions in a table
        st.subheader("Future Price Predictions")
        st.write(future_df)

    except Exception as e:
        st.error(f"An error occurred: {e}")
