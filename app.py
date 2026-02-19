import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px

# --- Setup ---
st.set_page_config(page_title="Alpha Quant Terminal", layout="wide")
st.title("⚡ Alpha Quant Terminal")

# --- Sidebar ---
st.sidebar.header("Terminal Settings")
symbol = st.sidebar.text_input("Ticker Symbol", value="NQ=F")
lookback_days = st.sidebar.slider("Analysis Window (Days)", 7, 60, 30)

@st.cache_data(ttl=3600)
def load_data(ticker, days):
    data = yf.download(ticker, period=f"{days}d", interval="1h")
    data.index = data.index.tz_localize(None)
    return data

try:
    df = load_data(symbol, lookback_days)
    df['Hour'] = df.index.hour
    df['HL_Range'] = df['High'] - df['Low']
    
    # --- Session Analysis ---
    st.subheader("📊 Session Volatility Logic")
    
    # Correcting the Series vs Float error
    london = df[(df['Hour'] >= 2) & (df['Hour'] <= 5)]['HL_Range'].mean()
    ny_am = df[(df['Hour'] >= 9) & (df['Hour'] <= 12)]['HL_Range'].mean()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("London Avg Move", f"{float(london):.2f} pts")
    c2.metric("NY AM Avg Move", f"{float(ny_am):.2f} pts")
    c3.metric("Volatility Edge", f"{((ny_am/london)-1)*100:.1f}% higher")

    # --- Volatility Heatmap ---
    st.subheader("🔥 Hourly Volatility Heatmap")
    hourly_vol = df.groupby('Hour')['HL_Range'].mean().reset_index()
    fig = px.bar(hourly_vol, x='Hour', y='HL_Range', 
                 title="Average Point Move by Hour (EST)",
                 labels={'HL_Range': 'Avg Points'},
                 color='HL_Range', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_layout=True)

    # --- Main Chart ---
    st.subheader("📈 Price Action")
    st.line_chart(df['Close'])

except Exception as e:
    st.error(f"Terminal Error: {e}")
