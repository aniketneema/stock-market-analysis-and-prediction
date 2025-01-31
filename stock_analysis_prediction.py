import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from textblob import TextBlob
import feedparser

# Streamlit App Title
st.set_page_config(page_title="Stock Analysis & Prediction", layout="wide")
st.title("📈 Stock Market Analysis & Prediction")

# Sidebar for User Input
st.sidebar.header("🔍 Stock Selection")
ticker = st.sidebar.text_input("Enter Stock Ticker (NSE):", "RELIANCE")
period = st.sidebar.selectbox("Select Time Period:", ["6mo", "1y", "5y", "10y"], index=1)

# Function to fetch stock data
def fetch_stock_data(ticker, period="1y"):
    stock = yf.Ticker(ticker + ".NS")  # Appending ".NS" for NSE stocks
    data = stock.history(period=period)
    return data

# Function to calculate technical indicators
def calculate_technical_indicators(data):
    data['MA_50'] = data['Close'].rolling(window=50).mean()
    data['MA_200'] = data['Close'].rolling(window=200).mean()
    data['RSI'] = 100 - (100 / (1 + (data['Close'].diff().where(data['Close'].diff() > 0, 0).rolling(window=14).mean() / data['Close'].diff().where(data['Close'].diff() < 0, 0).rolling(window=14).mean())))
    data['MACD'] = data['Close'].ewm(span=12).mean() - data['Close'].ewm(span=26).mean()
    data['Signal_Line'] = data['MACD'].ewm(span=9).mean()
    data['Bollinger_Upper'] = data['Close'].rolling(20).mean() + (data['Close'].rolling(20).std() * 2)
    data['Bollinger_Lower'] = data['Close'].rolling(20).mean() - (data['Close'].rolling(20).std() * 2)
    return data

# Function to fetch fundamental data
def fetch_fundamental_data(ticker):
    stock = yf.Ticker(ticker + ".NS")
    info = stock.info
    return {
        "Market Cap": info.get("marketCap", "N/A"),
        "P/E Ratio": info.get("trailingPE", "N/A"),
        "EPS": info.get("trailingEps", "N/A"),
        "Dividend Yield": info.get("dividendYield", "N/A")
    }

# Function to fetch news sentiment
def fetch_news_sentiment(ticker):
    url = f"https://news.google.com/rss/search?q={ticker}+stock+India"
    feed = feedparser.parse(url)
    sentiment_scores = [TextBlob(entry.title + " " + entry.summary).sentiment.polarity for entry in feed.entries[:5]]
    return np.mean(sentiment_scores) if sentiment_scores else 0

# Function to predict stock prices
def predict_stock(data, model_type="linear_regression"):
    data.dropna(inplace=True)
    X = data[['MA_50', 'MA_200', 'RSI', 'MACD']]
    X.replace([np.inf, -np.inf], np.nan, inplace=True)
    X.dropna(inplace=True)
    y = data['Close'].loc[X.index]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression() if model_type == "linear_regression" else RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    next_day_prediction = model.predict([X.iloc[-1]])[0]
    return predictions, mean_absolute_error(y_test, predictions), next_day_prediction

# Fetch and analyze data
data = fetch_stock_data(ticker, period)
if not data.empty:
    data = calculate_technical_indicators(data)
    sentiment = fetch_news_sentiment(ticker)
    predictions, mae, next_day_pred = predict_stock(data)
    fundamentals = fetch_fundamental_data(ticker)

    # Display stock data and sentiment
    st.subheader(f"📊 Stock Data for {ticker} ({period})")
    st.write(data.tail())
    st.metric(label="Sentiment Score", value=f"{sentiment:.2f}")
    st.metric(label="Prediction MAE", value=f"{mae:.2f}")
    st.metric(label="Next-Day Predicted Price", value=f"₹{next_day_pred:.2f}")
    
    # Display fundamental data
    st.subheader("🏛 Fundamental Data")
    st.write(pd.DataFrame(fundamentals, index=[0]))

    # Multiple Visualizations
    st.subheader("📊 Visualizations")
    
    # Candlestick Chart
    st.plotly_chart(px.line(data, x=data.index, y='Close', title="📈 Closing Price Trend"))
    st.plotly_chart(px.line(data, x=data.index, y=['MA_50', 'MA_200'], title="📊 Moving Averages"))
    st.plotly_chart(px.line(data, x=data.index, y=['MACD', 'Signal_Line'], title="📉 MACD & Signal Line"))
    st.plotly_chart(px.line(data, x=data.index, y='RSI', title="📊 Relative Strength Index (RSI)"))
    st.plotly_chart(px.line(data, x=data.index, y=['Bollinger_Upper', 'Bollinger_Lower'], title="📈 Bollinger Bands"))
    
    # Recommendations
    st.subheader("💡 Stock Recommendations")
    recommendation = "HOLD"
    if next_day_pred > data['Close'].iloc[-1] * 1.02:
        recommendation = "BUY\n  The stock is expected to rise based on current trends and indicators like MACD and RSI"
    elif next_day_pred < data['Close'].iloc[-1] * 0.98:
        recommendation = "SELL \n  The stock is showing bearish signals and may decline based on technical analysis."
    st.success(f"Recommended Action: **{recommendation}**")
else:
    st.error("No data found for the given ticker. Please try another stock.")