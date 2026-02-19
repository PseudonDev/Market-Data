import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

# --- Setup & Page Config ---
st.set_page_config(page_title="Quant Analyzer", layout="wide")
st.title("📈 Quantitative Market Analyzer")

# --- Sidebar Inputs ---
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Ticker Symbol", value="NQ=F")
lookback_days = st.sidebar.slider("Lookback Period (Days)", 7, 730, 60)

# --- Data Engine ---
@st.cache_data(ttl=3600)
def load_data(ticker, days):
    data = yf.download(ticker, period=f"{days}d", interval="1h")
    data.index = data.index.tz_localize(None)
    return data

try:
    df = load_data(symbol, lookback_days)
    st.success(f"Loaded {len(df)} rows for {symbol}")

    # --- Calculations ---
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / 
                                  df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))

    # --- Display Metrics ---
    col1, col2, col3 = st.columns(3)
col1.metric("Current Price", f"{df['Close'].iloc[-1].item():.2f}")
col2.metric("SMA (20)", f"{df['SMA_20'].iloc[-1].item():.2f}")
col3.metric("RSI (14)", f"{df['RSI'].iloc[-1].item():.2f}")

    # --- Reversal Logic ---
    st.subheader("Midnight Reversal Analysis")
    # (Insert your specific London/Midnight logic here)
    st.line_chart(df['Close'])

except Exception as e:
    st.error(f"Error loading data: {e}")
