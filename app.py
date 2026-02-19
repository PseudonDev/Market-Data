import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from openai import OpenAI

# --- Setup ---
st.set_page_config(page_title="Quant Analyzer", layout="wide")
st.title("📈 Quantitative Market Analyzer")

# --- Sidebar ---
st.sidebar.header("Settings")
symbol = st.sidebar.text_input("Ticker Symbol", value="NQ=F")
lookback_days = st.sidebar.slider("Lookback Period (Days)", 7, 730, 60)
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

@st.cache_data(ttl=3600)
def load_data(ticker, days):
    data = yf.download(ticker, period=f"{days}d", interval="1h")
    data.index = data.index.tz_localize(None)
    return data

try:
    df = load_data(symbol, lookback_days)
    
    # Calculations
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['RSI'] = 100 - (100 / (1 + (df['Close'].diff().clip(lower=0).rolling(14).mean() / df['Close'].diff().clip(upper=0).abs().rolling(14).mean())))

    # Dashboard
    col1, col2, col3 = st.columns(3)
    col1.metric("Current Price", f"{df['Close'].iloc[-1].item():.2f}")
    col2.metric("SMA (20)", f"{df['SMA_20'].iloc[-1].item():.2f}")
    col3.metric("RSI (14)", f"{df['RSI'].iloc[-1].item():.2f}")

    st.subheader("Price Chart")
    st.line_chart(df['Close'])

    # --- AI Chat Logic Layer ---
    st.divider()
    st.subheader("🤖 AI Quant Researcher")
    
    if not api_key:
        st.warning("Please enter your OpenAI API Key in the sidebar to use the Chat Box.")
    else:
        client = OpenAI(api_key=api_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about NY vs London session volatility..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # Logic: Send a summary of the market data to the AI
            recent_data_summary = df.tail(100).to_string()
            system_msg = f"You are a Quant Trader. Analyze this data for the user: {recent_data_summary}"

            with st.chat_message("assistant"):
               response = client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_msg},
                        {"role": "user", "content": prompt}
                    ]
                )
                answer = response.choices[0].message.content
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})

except Exception as e:
    st.error(f"Error: {e}")
