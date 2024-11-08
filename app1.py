import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM
from sklearn.preprocessing import MinMaxScaler

def stock_price_prediction(ticker, start_date, end_date, future_days):
    # Download stock data
    stock_data = yf.download(ticker, start=start_date, end=end_date)
    stock_data = stock_data[['Close']]
    
    # Prepare data for Linear Regression
    stock_data['Days'] = np.arange(len(stock_data))
    X = stock_data[['Days']].values
    y = stock_data['Close'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Linear Regression model
    lin_reg_model = LinearRegression()
    lin_reg_model.fit(X_train, y_train)
    
    # Predict future prices with Linear Regression
    future_X = np.array(range(len(stock_data), len(stock_data) + future_days)).reshape(-1, 1)
    lin_reg_predictions = lin_reg_model.predict(future_X)
    
    # Prepare data for LSTM
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(stock_data[['Close']].values)
    
    # Create sequences for LSTM
    sequence_length = 60
    X_lstm, y_lstm = [], []
    for i in range(sequence_length, len(scaled_data)):
        X_lstm.append(scaled_data[i-sequence_length:i, 0])
        y_lstm.append(scaled_data[i, 0])
    X_lstm, y_lstm = np.array(X_lstm), np.array(y_lstm)
    X_lstm = np.reshape(X_lstm, (X_lstm.shape[0], X_lstm.shape[1], 1))
    
    # Split data for LSTM
    X_lstm_train, X_lstm_test = X_lstm[:-future_days], X_lstm[-future_days:]
    y_lstm_train, y_lstm_test = y_lstm[:-future_days], y_lstm[-future_days:]
    
    # Build and train the LSTM model
    lstm_model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(X_lstm_train.shape[1], 1)),
        LSTM(50, return_sequences=False),
        Dense(25),
        Dense(1)
    ])
    lstm_model.compile(optimizer='adam', loss='mean_squared_error')
    lstm_model.fit(X_lstm_train, y_lstm_train, epochs=10, batch_size=32, verbose=1)
    
    # Predict future prices with LSTM
    lstm_predictions = lstm_model.predict(X_lstm_test)
    lstm_predictions = scaler.inverse_transform(lstm_predictions)

    # Plot results
    plt.figure(figsize=(14, 7))
    plt.plot(stock_data['Days'], stock_data['Close'], label="Historical Prices", color="blue")
    plt.plot(range(len(stock_data), len(stock_data) + future_days), lin_reg_predictions, color='orange', label="Future Predictions (Linear Regression)")
    plt.plot(range(len(stock_data) - len(y_lstm_test), len(stock_data)), lstm_predictions, color='green', label="LSTM Predicted Prices")
    plt.xlabel("Days")
    plt.ylabel("Stock Price")
    plt.title(f"Stock Price Prediction for {ticker}")
    plt.legend()
    plt.show()

    return lin_reg_predictions, lstm_predictions

# Set parameters based on user input
ticker = input("Enter the stock ticker symbol (e.g., AAPL for Apple): ")
start_date = input("Enter the start date (YYYY-MM-DD): ")
end_date = input("Enter the end date (YYYY-MM-DD): ")
future_days = int(input("Enter the number of future days to predict: "))

# Run predictions
lin_reg_predictions, lstm_predictions = stock_price_prediction(ticker, start_date, end_date, future_days)
print("Predicted Future Prices (Linear Regression):", lin_reg_predictions)
print("Predicted Future Prices (LSTM):", lstm_predictions)
