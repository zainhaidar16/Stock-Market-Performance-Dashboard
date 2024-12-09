import streamlit as st
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
import numpy as np

import plotly.graph_objects as go

# Page configuration
st.set_page_config(page_title="Stock Market Dashboard", layout="wide")
st.title("Stock Market Performance Dashboard")

# Sidebar
st.sidebar.header("Dashboard Settings")

# Date range selection
start_date = st.sidebar.date_input("Start Date", datetime.now() - timedelta(days=365))
end_date = st.sidebar.date_input("End Date", datetime.now())

# Stock selection
default_stocks = ["AAPL", "GOOGL", "MSFT", "AMZN"]
stocks = st.sidebar.multiselect("Select stocks to compare", default_stocks, default=default_stocks)

# Technical indicators
technical_indicators = st.sidebar.multiselect(
    "Select Technical Indicators",
    ["SMA", "EMA", "RSI", "MACD"],
    default=["SMA"]
)

# Function to get stock data
@st.cache_data
def get_stock_data(symbol, start, end):
    stock = yf.Ticker(symbol)
    df = stock.history(start=start, end=end)
    return df

# Function to calculate technical indicators
def calculate_indicators(df):
    if "SMA" in technical_indicators:
        df["SMA20"] = df["Close"].rolling(window=20).mean()
        df["SMA50"] = df["Close"].rolling(window=50).mean()
    
    if "EMA" in technical_indicators:
        df["EMA20"] = df["Close"].ewm(span=20).mean()
    
    if "RSI" in technical_indicators:
        delta = df["Close"].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df["RSI"] = 100 - (100 / (1 + rs))
    
    return df

# Main content
col1, col2 = st.columns([2, 1])

with col1:
    # Price comparison chart
    st.subheader("Stock Price Comparison")
    fig = go.Figure()
    
    for symbol in stocks:
        df = get_stock_data(symbol, start_date, end_date)
        df = calculate_indicators(df)
        
        # Plot stock price
        fig.add_trace(go.Scatter(x=df.index, y=df["Close"],
                                name=symbol, mode="lines"))
        
        # Plot technical indicators
        if "SMA" in technical_indicators:
            fig.add_trace(go.Scatter(x=df.index, y=df["SMA20"],
                                   name=f"{symbol} SMA20", line=dict(dash="dash")))
    
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Price",
        height=600,
        showlegend=True
    )
    st.plotly_chart(fig, use_container_width=True)

with col2:
    # Performance metrics
    st.subheader("Performance Metrics")
    
    for symbol in stocks:
        df = get_stock_data(symbol, start_date, end_date)
        
        # Calculate metrics
        total_return = ((df["Close"][-1] - df["Close"][0]) / df["Close"][0]) * 100
        daily_returns = df["Close"].pct_change()
        volatility = daily_returns.std() * np.sqrt(252) * 100
        
        # Display metrics
        st.write(f"**{symbol}**")
        st.write(f"Total Return: {total_return:.2f}%")
        st.write(f"Volatility: {volatility:.2f}%")
        st.write("---")

# Portfolio Simulation
st.subheader("Portfolio Simulation")
investment_amount = st.number_input("Enter investment amount ($)", value=10000)

if st.button("Simulate Portfolio"):
    portfolio_values = pd.DataFrame()
    
    for symbol in stocks:
        df = get_stock_data(symbol, start_date, end_date)
        shares = investment_amount / len(stocks) / df["Close"][0]
        portfolio_values[symbol] = df["Close"] * shares
    
    portfolio_values["Total"] = portfolio_values.sum(axis=1)
    
    # Plot portfolio value
    fig_portfolio = go.Figure()
    fig_portfolio.add_trace(go.Scatter(x=portfolio_values.index, 
                                     y=portfolio_values["Total"],
                                     name="Portfolio Value"))
    
    fig_portfolio.update_layout(
        xaxis_title="Date",
        yaxis_title="Portfolio Value ($)",
        height=400
    )
    st.plotly_chart(fig_portfolio, use_container_width=True)
    
    # Calculate portfolio metrics
    portfolio_return = ((portfolio_values["Total"][-1] - investment_amount) / 
                       investment_amount) * 100
    st.write(f"Portfolio Return: {portfolio_return:.2f}%")
    st.write(f"Final Portfolio Value: ${portfolio_values['Total'][-1]:.2f}")

# Add data refresh button
if st.button("Refresh Data"):
    st.experimental_rerun()