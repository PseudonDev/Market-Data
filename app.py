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
api_key = st.sidebar.text_input("OpenAI API Key", type="password")

@st.cache_data(ttl=3600)
def load_data(ticker, days):
    data = yf.download(ticker, period=f"{days}d", interval="1h")
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    data.index = data.index.tz_localize(None)
    return data

try:
    df = load_data(symbol, lookback_days)
    df['Hour'] = df.index.hour
    df['HL_Range'] = df['High'] - df['Low']
    
    # --- Advanced Math: Session SD Levels ---
    # 1. Get London Session (2am-5am)
    london_data = df[(df['Hour'] >= 2) & (df['Hour'] <= 5)]
    lon_mean = london_data['Close'].mean()
    lon_std = london_data['Close'].std()
    
    # 2. Calculate the "Reversal Zones"
    sd1 = lon_mean + (1.25 * lon_std) # Midpoint of 1-1.5
    sd2 = lon_mean + (2.25 * lon_std) # Midpoint of 2-2.5
    
    c1, c2, c3 = st.columns(3)
    c1.metric("London Mean (Price)", f"{lon_mean:.2f}")
    c2.metric("1.5 SD Level", f"{lon_mean + (1.5 * lon_std):.2f}")
    c3.metric("2.5 SD Level", f"{lon_mean + (2.5 * lon_std):.2f}")

    # --- Volatility Heatmap ---
    st.subheader("🔥 Hourly Volatility Heatmap")
    hourly_vol = df.groupby('Hour')['HL_Range'].mean().reset_index()
    fig = px.bar(hourly_vol, x='Hour', y='HL_Range', color='HL_Range', color_continuous_scale='Viridis')
    st.plotly_chart(fig, use_container_layout=True)

    # --- AI Chat Researcher ---
    st.divider()
    st.subheader("🤖 AI Quant Researcher")
    
    if not api_key:
        st.info("Enter your OpenAI API Key in the sidebar.")
    else:
        client = OpenAI(api_key=api_key)
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if prompt := st.chat_input("Ask about the SD reversals..."):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)

            # FEEDING THE AI ACTUAL MATH
            context = f"""
            Ticker: {symbol}
            London Mean Price: {lon_mean:.2f}
            London StdDev: {lon_std:.2f}
            NY Open Price: {df[df['Hour'] == 9]['Open'].iloc[-1] if 9 in df['Hour'].values else 'N/A'}
            
            SD Levels anchored to London Mean:
            +1.0 SD: {lon_mean + lon_std:.2f}
            +1.5 SD: {lon_mean + (1.5 * lon_std):.2f}
            +2.0 SD: {lon_mean + (2.0 * lon_std):.2f}
            +2.5 SD: {lon_mean + (2.5 * lon_std):.2f}
            """
            
            with st.chat_message("assistant"):
                try:
                    response = client.chat.completions.create(
                        model="gpt-4o-mini",
                        messages=[
                            {"role": "system", "content": f"You are a Quant. Use these exact numbers to answer: {context}"},
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
