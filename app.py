import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.express as px
from openai import OpenAI

# --- Setup ---
st.set_page_config(page_title="Alpha Quant Terminal", layout="wide")
st.title("⚡ Alpha Quant Terminal")

# --- Sidebar ---
st.sidebar.header("Terminal Settings")
symbol = st.sidebar.text_input("Ticker Symbol", value="NQ=F")
lookback_days = st.sidebar.slider("Analysis Window (Days)", 7, 60, 30)
api_key = st.sidebar.text_input("OpenAI API Key (Required for Chat)", type="password")

@st.cache_data(ttl=3600)
def load_data(ticker, days):
    # Download data
    data = yf.download(ticker, period=f"{days}d", interval="1h")
    
    # FIX: Flatten Multi-Index columns if they exist
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    
    data.index = data.index.tz_localize(None)
    return data

try:
    df = load_data(symbol, lookback_days)
    
    # Double check columns exist after flattening
    required_cols = ['Open', 'High', 'Low', 'Close']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Data for {symbol} is missing required columns. Try another ticker like 'AAPL'.")
        st.stop()

    df['Hour'] = df.index.hour
    df['HL_Range'] = df['High'] - df['Low']
    
    # --- Metrics & Heatmap ---
    london = df[(df['Hour'] >= 2) & (df['Hour'] <= 5)]['HL_Range'].mean()
    ny_am = df[(df['Hour'] >= 9) & (df['Hour'] <= 12)]['HL_Range'].mean()
    
    c1, c2, c3 = st.columns(3)
    c1.metric("London Avg Move", f"{float(london):.2f} pts")
    c2.metric("NY AM Avg Move", f"{float(ny_am):.2f} pts")
    c3.metric("Volatility Edge", f"{((ny_am/london)-1)*100:.1f}% higher")

    st.subheader("🔥 Hourly Volatility Heatmap")
    hourly_vol = df.groupby('Hour')['HL_Range'].mean().reset_index()
    fig = px.bar(hourly_vol, x='Hour', y='HL_Range', color='HL_Range', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_layout=True)

    # --- AI Chat Researcher ---
    st.divider()
    st.subheader("🤖 AI Quant Researcher")
    
    if not api_key:
        st.info("Enter your OpenAI API Key in the sidebar to unlock the Research Chat.")
    else:
        client = OpenAI(api_key=api_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about reversals or standard deviations..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Summarize for AI context
            daily_summary = df.resample('D').agg({
                'High': 'max', 'Low': 'min', 'Close': 'last', 'HL_Range': 'mean'
            }).tail(7).to_string()
            
            context = f"Analyzing {symbol}. London Avg={london:.2f}, NY Avg={ny_am:.2f}. Last 7 Days: {daily_summary}"
            
            with st.chat_message("assistant"):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"You are a Quant Strategist. Context: {context}"},
                            {"role": "user", "content": prompt}
                        ]
                    )
                    answer = response.choices[0].message.content
                    st.markdown(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                except Exception as e:
                    st.error(f"AI Error: {e}")

except Exception as e:
    st.error(f"Terminal Error: {e}")
