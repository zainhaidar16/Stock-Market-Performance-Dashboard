import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

# Enhanced Page Configuration
st.set_page_config(page_title="Advanced Stock Market Dashboard", layout="wide", page_icon="📈")

# Title and Introduction
st.title("🚀 Advanced Stock Market Performance Dashboard")
st.markdown("## Real-Time Market Insights and Portfolio Analysis")

# Enhanced Sidebar
with st.sidebar:
    st.header("Dashboard Controls")
    
    # Enhanced Date Range Selection
    date_range_type = st.selectbox("Date Range", 
        ["Last Year", "Year to Date", "Last 6 Months", "Last 3 Months", "Custom"])
    
    if date_range_type == "Custom":
        start_date = st.date_input("Start Date", datetime.now() - timedelta(days=365))
        end_date = st.date_input("End Date", datetime.now())
    else:
        end_date = datetime.now()
        if date_range_type == "Last Year":
            start_date = end_date - timedelta(days=365)
        elif date_range_type == "Year to Date":
            start_date = datetime(end_date.year, 1, 1)
        elif date_range_type == "Last 6 Months":
            start_date = end_date - timedelta(days=180)
        else:  # Last 3 Months
            start_date = end_date - timedelta(days=90)
    
    # Expanded Stock Selection with Market Cap Filtering
    st.subheader("Stock Selection")
    market_cap_filter = st.multiselect(
        "Filter by Market Cap", 
        ["Large Cap", "Mid Cap", "Small Cap"],
        default=["Large Cap"]
    )
    
    # Predefined stock lists
    market_cap_stocks = {
        "Large Cap": ["AAPL", "GOOGL", "MSFT", "AMZN", "NVDA"],
        "Mid Cap": ["CRM", "PYPL", "SNOW", "UBER"],
        "Small Cap": ["ROKU", "PLUG", "DKNG", "OPEN"]
    }
    
    # Combine stocks based on market cap filter
    available_stocks = [
        stock for cap in market_cap_filter 
        for stock in market_cap_stocks.get(cap, [])
    ]
    
    default_stocks = available_stocks[:4] if available_stocks else ["AAPL"]
    stocks = st.multiselect(
        "Select Stocks to Compare", 
        available_stocks, 
        default=default_stocks
    )
    
    # Enhanced Technical Indicators
    st.subheader("Technical Analysis")
    technical_indicators = st.multiselect(
        "Select Indicators",
        ["SMA", "EMA", "RSI", "MACD", "Bollinger Bands", "Volume"],
        default=["SMA", "RSI"]
    )
    
    # Risk Tolerance Slider
    risk_tolerance = st.slider(
        "Portfolio Risk Tolerance", 
        min_value=1, 
        max_value=10, 
        value=5, 
        help="Adjust for conservative (1) to aggressive (10) investment strategies"
    )

# Caching and Performance Optimization
@st.cache_data(ttl=3600)  # Cache data for 1 hour
def get_stock_data(symbol, start, end):
    """Fetch stock data with enhanced error handling"""
    try:
        stock = yf.Ticker(symbol)
        df = stock.history(start=start, end=end)
        
        # Add additional market data
        df['Market Cap'] = stock.info.get('marketCap', np.nan)
        df['Sector'] = stock.info.get('sector', 'Unknown')
        
        return df
    except Exception as e:
        st.error(f"Error fetching data for {symbol}: {e}")
        return pd.DataFrame()

def calculate_advanced_indicators(df):
    """Calculate comprehensive technical indicators"""
    indicators = {}
    
    # Simple Moving Averages
    if "SMA" in technical_indicators:
        indicators['SMA20'] = df['Close'].rolling(window=20).mean()
        indicators['SMA50'] = df['Close'].rolling(window=50).mean()
    
    # Exponential Moving Average
    if "EMA" in technical_indicators:
        indicators['EMA20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # Relative Strength Index
    if "RSI" in technical_indicators:
        delta = df['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['RSI'] = 100 - (100 / (1 + rs))
    
    # Bollinger Bands
    if "Bollinger Bands" in technical_indicators:
        indicators['BB_Middle'] = df['Close'].rolling(window=20).mean()
        indicators['BB_Upper'] = indicators['BB_Middle'] + 2 * df['Close'].rolling(window=20).std()
        indicators['BB_Lower'] = indicators['BB_Middle'] - 2 * df['Close'].rolling(window=20).std()
    
    # Add calculated indicators to dataframe
    for name, series in indicators.items():
        df[name] = series
    
    return df

def calculate_risk_metrics(returns):
    """Calculate advanced risk metrics"""
    metrics = {
        'Annual Return': returns.mean() * 252,
        'Annual Volatility': returns.std() * np.sqrt(252),
        'Sharpe Ratio': (returns.mean() * 252) / (returns.std() * np.sqrt(252)),
        'Max Drawdown': (returns.cummax() - returns).max()
    }
    return metrics

# Tabs for different views
tab1, tab2, tab3 = st.tabs(["📈 Price Analysis", "📊 Portfolio Simulation", "🔍 Risk Analysis"])

with tab1:
    # Price Comparison Chart with Enhanced Visualization
    st.subheader("Comparative Stock Performance")
    
    # Multi-axis price chart
    fig_multi_price = go.Figure()
    
    for symbol in stocks:
        df = get_stock_data(symbol, start_date, end_date)
        df = calculate_advanced_indicators(df)
        
        # Normalized price for fair comparison
        normalized_price = df['Close'] / df['Close'].iloc[0] * 100
        
        fig_multi_price.add_trace(go.Scatter(
            x=df.index, 
            y=normalized_price, 
            mode='lines', 
            name=f'{symbol} Normalized Price'
        ))
        
        # Add technical indicators
        if "SMA" in technical_indicators:
            fig_multi_price.add_trace(go.Scatter(
                x=df.index, 
                y=df['SMA20'] / df['Close'].iloc[0] * 100, 
                mode='lines', 
                name=f'{symbol} SMA20',
                line=dict(dash='dash')
            ))
    
    fig_multi_price.update_layout(
        title='Normalized Stock Performance',
        xaxis_title='Date',
        yaxis_title='Normalized Price (%)',
        height=600
    )
    st.plotly_chart(fig_multi_price, use_container_width=True)

with tab2:
    # Enhanced Portfolio Simulation
    st.subheader("Portfolio Simulation and Optimization")
    
    col1, col2 = st.columns(2)
    
    with col1:
        investment_amount = st.number_input(
            "Initial Investment ($)", 
            value=10000, 
            min_value=1000, 
            step=1000
        )
    
    with col2:
        allocation_strategy = st.selectbox(
            "Portfolio Allocation", 
            ["Equal Weights", "Market Cap Weighted", "Risk-Based"]
        )
    
    if st.button("Simulate Portfolio"):
        portfolio_values = pd.DataFrame()
        
        for symbol in stocks:
            df = get_stock_data(symbol, start_date, end_date)
            
            # Dynamic allocation strategy
            if allocation_strategy == "Equal Weights":
                shares = investment_amount / len(stocks) / df['Close'].iloc[0]
            elif allocation_strategy == "Market Cap Weighted":
                market_cap = df['Market Cap'].iloc[-1]
                total_market_cap = sum(get_stock_data(s, start_date, end_date)['Market Cap'].iloc[-1] for s in stocks)
                shares = (market_cap / total_market_cap) * investment_amount / df['Close'].iloc[0]
            else:  # Risk-Based
                returns = df['Close'].pct_change()
                volatility = returns.std()
                inv_volatility = 1 / volatility
                shares = (inv_volatility / sum(1/get_stock_data(s, start_date, end_date)['Close'].pct_change().std() for s in stocks)) * investment_amount / df['Close'].iloc[0]
            
            portfolio_values[symbol] = df['Close'] * shares
        
        portfolio_values['Total'] = portfolio_values.sum(axis=1)
        
        # Portfolio Performance Visualization
        fig_portfolio = px.line(
            portfolio_values['Total'], 
            title='Portfolio Value Over Time',
            labels={'value': 'Portfolio Value ($)', 'index': 'Date'}
        )
        st.plotly_chart(fig_portfolio, use_container_width=True)
        
        # Portfolio Performance Metrics
        portfolio_return = ((portfolio_values['Total'].iloc[-1] - investment_amount) / investment_amount) * 100
        st.metric("Portfolio Return", f"{portfolio_return:.2f}%")

with tab3:
    # Comprehensive Risk Analysis
    st.subheader("Portfolio Risk Assessment")
    
    risk_metrics_data = {}
    for symbol in stocks:
        df = get_stock_data(symbol, start_date, end_date)
        returns = df['Close'].pct_change()
        risk_metrics_data[symbol] = calculate_risk_metrics(returns)
    
    # Risk Metrics Table
    risk_df = pd.DataFrame.from_dict(risk_metrics_data, orient='index')
    st.dataframe(risk_df.style.format("{:.2%}"))
    
    # Risk Visualization
    risk_fig = go.Figure(data=[
        go.Bar(
            name='Annual Volatility', 
            x=list(risk_metrics_data.keys()), 
            y=[metrics['Annual Volatility'] for metrics in risk_metrics_data.values()]
        )
    ])
    risk_fig.update_layout(title='Stock Volatility Comparison')
    st.plotly_chart(risk_fig, use_container_width=True)

# Real-time data refresh
st.sidebar.markdown("---")
if st.sidebar.button("🔄 Refresh All Data"):
    st.experimental_rerun()