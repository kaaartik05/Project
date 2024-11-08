import yfinance as yf
import streamlit as st
import matplotlib.pyplot as plt

# Streamlit UI Setup
st.title("Stock Price Prediction")
st.write("Enter a stock ticker to get the last few days of closing prices")

# Input: Stock Ticker
ticker = st.text_input("Stock Ticker", "AAPL")  # Default to "AAPL" for Apple Inc.

if ticker:
    # Fetch data for the given stock ticker
    data = yf.download(ticker, period="1y", interval="1d")

    if not data.empty:
        # Display last 5 closing prices
        st.write("Last 5 Closing Prices:")
        st.write(data['Close'].tail(5))

        # Plot the closing prices over time
        st.write("Stock Closing Prices Over the Last Year")
        plt.figure(figsize=(10, 5))
        plt.plot(data['Close'], color='blue', label='Closing Price')
        plt.title(f"{ticker} Stock Price")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        st.pyplot(plt)
    else:
        st.write("No data found for the ticker. Please try a different one.")
