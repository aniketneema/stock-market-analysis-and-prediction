# stock-market-analysis-and-prediction
This project provides a comprehensive stock market analysis and prediction tool using Python, Streamlit, and various libraries for financial data analysis and machine learning.
The app will open in your default web browser.

## Key Features:
Real-time Stock Data: Fetch stock data from Yahoo Finance for analysis.  
Technical Indicators:Calculate and visualize technical indicators such as moving averages (MA), Relative Strength Index (RSI), MACD, and Bollinger Bands.  
Sentiment Analysis: Analyze news sentiment based on the stock ticker using the TextBlob library to gauge the overall market sentiment.  
Predictive Modeling: Predict future stock prices using linear regression and random forest models based on technical indicators.  
Stock Recommendations: Provide buy/sell/hold recommendations based on stock predictions and technical analysis.  


## Libraries and Tools Used:
Streamlit: For creating an interactive web app interface.  
yfinance: For fetching stock market data.  
plotly: For interactive visualizations.  
scikit-learn: For machine learning (Linear Regression, Random Forest).  
TextBlob: For sentiment analysis from news data.  
pandas, numpy: For data manipulation and processing.  
feedparser: For fetching and parsing news data.  

## Usage
Enter a stock ticker symbol (e.g., RELIANCE) in the sidebar input.  
Select the time period (6 months, 1 year, 5 years, or 10 years) for stock analysis.  
View the stock's closing price data, technical indicators, and predicted next-day price.  
Get stock recommendations based on predictive modeling and technical analysis.  

## Sample Outputs:
Stock Data Table: Shows the closing price, moving averages, RSI, MACD, and Bollinger Bands.  
Sentiment Score: Shows the sentiment score based on recent news articles.  
Model Accuracy: Displays Mean Absolute Error (MAE) for stock price prediction.  
Stock Recommendation: Buy, Sell, or Hold based on prediction and analysis.  


## Visualizations:
Closing Price Trend: Line chart showing the stock's closing price over time.
Moving Averages: Plots of 50-day and 200-day moving averages.
MACD & Signal Line: Charts of MACD and Signal Line.
RSI: Line chart showing the Relative Strength Index.
Bollinger Bands: Visualizes the upper and lower Bollinger Bands.

## Contributing
Feel free to fork this repository and submit pull requests for any enhancements or bug fixes. Contributions are welcome!

