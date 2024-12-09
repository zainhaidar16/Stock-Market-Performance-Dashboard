import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objs as go
import plotly.express as px
from datetime import datetime, timedelta
import ta

class StockMarketDashboard:
    def __init__(self):
        # Set page configuration
        st.set_page_config(page_title="Stock Market Performance Dashboard", 
                            page_icon=":chart_with_upwards_trend:", 
                            layout="wide")
        
        # Initialize session state for portfolio
        if 'portfolio' not in st.session_state:
            st.session_state.portfolio = {}
    
    def fetch_stock_data(self, tickers, start_date, end_date):
        """
        Fetch historical stock data for multiple tickers
        
        Args:
            tickers (list): List of stock ticker symbols
            start_date (datetime): Start date for data retrieval
            end_date (datetime): End date for data retrieval
        
        Returns:
            dict: Dictionary of DataFrames for each ticker
        """
        stock_data = {}
        for ticker in tickers:
            try:
                df = yf.download(ticker, start=start_date, end=end_date)
                
                # Calculate technical indicators
                df['RSI'] = ta.momentum.RSIIndicator(df['Close']).rsi()
                df['MACD'] = ta.trend.MACD(df['Close']).macd()
                df['Bollinger_High'] = ta.volatility.BollingerBands(df['Close']).bollinger_hband()
                df['Bollinger_Low'] = ta.volatility.BollingerBands(df['Close']).bollinger_lband()
                
                stock_data[ticker] = df
            except Exception as e:
                st.error(f"Error fetching data for {ticker}: {e}")
        
        return stock_data
    
    def plot_stock_performance(self, stock_data):
        """
        Create interactive stock performance comparison plot
        
        Args:
            stock_data (dict): Dictionary of stock DataFrames
        """
        st.subheader("Stock Performance Comparison")
        
        # Prepare normalized data for comparison
        normalized_data = {}
        for ticker, df in stock_data.items():
            normalized_df = df['Close'] / df['Close'].iloc[0] * 100
            normalized_data[ticker] = normalized_df
        
        # Create comparison DataFrame
        comparison_df = pd.DataFrame(normalized_data)
        
        # Plotly interactive line chart
        fig = go.Figure()
        for ticker in comparison_df.columns:
            fig.add_trace(go.Scatter(
                x=comparison_df.index, 
                y=comparison_df[ticker], 
                mode='lines', 
                name=ticker
            ))
        
        fig.update_layout(
            title='Normalized Stock Performance',
            xaxis_title='Date',
            yaxis_title='Normalized Price (Base 100)',
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    
    def portfolio_simulation(self, stock_data):
        """
        Portfolio simulation and optimization
        
        Args:
            stock_data (dict): Dictionary of stock DataFrames
        """
        st.subheader("Portfolio Simulation")
        
        # Portfolio allocation input
        st.write("Portfolio Allocation")
        portfolio_allocation = {}
        for ticker in stock_data.keys():
            allocation = st.number_input(f"{ticker} Allocation (%)", 
                                         min_value=0.0, 
                                         max_value=100.0, 
                                         value=0.0,
                                         step=1.0,
                                         key=f"allocation_{ticker}")
            portfolio_allocation[ticker] = allocation / 100
        
        # Validate total allocation
        total_allocation = sum(portfolio_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            st.warning(f"Total allocation must be 100%. Current allocation: {total_allocation*100:.2f}%")
            return
        
        # Portfolio performance calculation
        portfolio_returns = pd.DataFrame()
        for ticker, weight in portfolio_allocation.items():
            portfolio_returns[ticker] = stock_data[ticker]['Close'].pct_change() * weight
        
        portfolio_returns['Portfolio_Return'] = portfolio_returns.sum(axis=1)
        cumulative_portfolio_return = (1 + portfolio_returns['Portfolio_Return']).cumprod() * 100 - 100
        
        # Portfolio performance plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=cumulative_portfolio_return.index, 
            y=cumulative_portfolio_return.values, 
            mode='lines', 
            name='Portfolio Cumulative Return'
        ))
        
        fig.update_layout(
            title='Portfolio Cumulative Performance',
            xaxis_title='Date',
            yaxis_title='Cumulative Return (%)'
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Portfolio statistics
        st.subheader("Portfolio Statistics")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Return", f"{cumulative_portfolio_return.iloc[-1]:.2f}%")
        
        with col2:
            annual_volatility = portfolio_returns['Portfolio_Return'].std() * np.sqrt(252) * 100
            st.metric("Annual Volatility", f"{annual_volatility:.2f}%")
        
        with col3:
            sharpe_ratio = (portfolio_returns['Portfolio_Return'].mean() * 252) / (portfolio_returns['Portfolio_Return'].std() * np.sqrt(252))
            st.metric("Sharpe Ratio", f"{sharpe_ratio:.2f}")
    
    def technical_indicators(self, stock_data):
        """
        Display technical indicators for selected stocks
        
        Args:
            stock_data (dict): Dictionary of stock DataFrames
        """
        st.subheader("Technical Indicators")
        
        # Select stock for detailed analysis
        selected_ticker = st.selectbox("Select Stock for Technical Analysis", 
                                       list(stock_data.keys()))
        
        # Create subplots for different indicators
        fig = go.Figure()
        
        # Candlestick chart
        fig.add_trace(go.Candlestick(
            x=stock_data[selected_ticker].index,
            open=stock_data[selected_ticker]['Open'],
            high=stock_data[selected_ticker]['High'],
            low=stock_data[selected_ticker]['Low'],
            close=stock_data[selected_ticker]['Close'],
            name='Price'
        ))
        
        # Bollinger Bands
        fig.add_trace(go.Scatter(
            x=stock_data[selected_ticker].index,
            y=stock_data[selected_ticker]['Bollinger_High'],
            mode='lines',
            name='Bollinger High',
            line=dict(color='rgba(173, 216, 230, 0.5)')
        ))
        
        fig.add_trace(go.Scatter(
            x=stock_data[selected_ticker].index,
            y=stock_data[selected_ticker]['Bollinger_Low'],
            mode='lines',
            name='Bollinger Low',
            line=dict(color='rgba(173, 216, 230, 0.5)')
        ))
        
        fig.update_layout(
            title=f'{selected_ticker} Technical Analysis',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # RSI and MACD in separate columns
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("RSI Indicator")
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(
                x=stock_data[selected_ticker].index,
                y=stock_data[selected_ticker]['RSI'],
                mode='lines',
                name='RSI'
            ))
            fig_rsi.add_hline(y=70, line_color='red', line_dash='dash')
            fig_rsi.add_hline(y=30, line_color='green', line_dash='dash')
            fig_rsi.update_layout(title='Relative Strength Index (RSI)')
            st.plotly_chart(fig_rsi, use_container_width=True)
        
        with col2:
            st.subheader("MACD Indicator")
            fig_macd = go.Figure()
            fig_macd.add_trace(go.Scatter(
                x=stock_data[selected_ticker].index,
                y=stock_data[selected_ticker]['MACD'],
                mode='lines',
                name='MACD'
            ))
            fig_macd.update_layout(title='Moving Average Convergence Divergence')
            st.plotly_chart(fig_macd, use_container_width=True)
    
    def main(self):
        """
        Main application logic
        """
        st.title("📈 Stock Market Performance Dashboard")
        
        # Sidebar for user inputs
        with st.sidebar:
            st.header("Dashboard Configuration")
            
            # Default tickers
            default_tickers = ['AAPL', 'GOOGL', 'MSFT', 'AMZN']
            tickers = st.multiselect("Select Stocks", 
                                     options=default_tickers,
                                     default=default_tickers[:2])
            
            # Date range selection
            end_date = datetime.now()
            start_date = end_date - timedelta(days=365)
            
            date_range = st.date_input("Select Date Range", 
                                       value=[start_date, end_date])
            
            if len(date_range) == 2:
                start_date, end_date = date_range
            
            # Fetch stock data
            stock_data = self.fetch_stock_data(tickers, start_date, end_date)
            
            if not stock_data:
                st.error("No stock data available. Please select valid tickers.")
                return
        
        # Dashboard tabs
        tab1, tab2, tab3 = st.tabs([
            "Performance Comparison", 
            "Portfolio Simulation", 
            "Technical Indicators"
        ])
        
        with tab1:
            self.plot_stock_performance(stock_data)
        
        with tab2:
            self.portfolio_simulation(stock_data)
        
        with tab3:
            self.technical_indicators(stock_data)

# Streamlit app entry point
def run_dashboard():
    dashboard = StockMarketDashboard()
    dashboard.main()

if __name__ == "__main__":
    run_dashboard()