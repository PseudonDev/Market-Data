from fastapi import FastAPI, Query
from pydantic import BaseModel
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta

app = FastAPI(title="NQ 5m Data + AMD Detector (demo)")

SYMBOL = "NQ=F"  # Yahoo symbol for E-mini Nasdaq continuous future

def fetch_5m(symbol: str = SYMBOL, period: str = "7d", interval: str = "5m") -> pd.DataFrame:
    df = yf.download(symbol, interval=interval, period=period, progress=False, threads=False)
    if df.empty:
        raise RuntimeError("No data returned from yfinance. Check period/interval or internet.")
    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.index = pd.to_datetime(df.index)
    df = df[["open", "high", "low", "close", "volume"]]
    return df

def compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["typ"] = (df["high"] + df["low"] + df["close"]) / 3.0
    df["date"] = df.index.date
    df["vwap"] = np.nan
    for d, group in df.groupby("date"):
        vol_cum = group["volume"].cumsum()
        vp_cum = (group["typ"] * group["volume"]).cumsum()
        df.loc[group.index, "vwap"] = vp_cum / vol_cum.replace(0, np.nan)
    df["ma20"] = df["close"].rolling(window=20, min_periods=1).mean()
    df["ma50"] = df["close"].rolling(window=50, min_periods=1).mean()
    high_low = df["high"] - df["low"]
    high_prev_close = (df["high"] - df["close"].shift(1)).abs()
    low_prev_close = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([high_low, high_prev_close, low_prev_close], axis=1).max(axis=1)
    df["atr5"] = tr.rolling(window=5, min_periods=1).mean()
    df["range"] = df["high"] - df["low"]
    df.drop(columns=["date"], inplace=True)
    return df

def detect_amd(df: pd.DataFrame,
               vol_spike_mult: float = 3.0,
               range_spike_mult: float = 2.5,
               low_range_mult: float = 0.8,
               accumulation_window_bars: int = 24,
               reversal_window_bars: int = 6):
    df = df.copy()
    df["vol_med_rolling"] = df["volume"].rolling(window=accumulation_window_bars, min_periods=1).median()
    df["range_med_rolling"] = df["range"].rolling(window=accumulation_window_bars, min_periods=1).median()
    is_vol_spike = df["volume"] > (df["vol_med_rolling"] * vol_spike_mult)
    is_range_spike = df["range"] > (df["range_med_rolling"] * range_spike_mult)
    is_spike = is_vol_spike & is_range_spike
    low_range = df["range"] < (df["range_med_rolling"] * low_range_mult)
    low_range_roll_frac = low_range.rolling(window=accumulation_window_bars, min_periods=1).mean()
    is_accum_region = low_range_roll_frac > 0.75
    df["label"] = np.nan
    df.loc[is_accum_region, "label"] = "accumulation"
    df["vwap_diff"] = df["vwap"].diff()
    vwap_down = df["vwap_diff"].rolling(window=accumulation_window_bars, min_periods=1).mean() < 0
    df.loc[is_accum_region & vwap_down, "label"] = "distribution"
    manip_periods = []
    for ts in df[is_spike].index:
        i = df.index.get_indexer([ts])[0]
        end_i = min(len(df) - 1, i + reversal_window_bars)
        spike_bar = df.iloc[i]
        spike_dir = np.sign(spike_bar["close"] - spike_bar["open"])
        spike_range = spike_bar["range"]
        if spike_range <= 0:
            continue
        window = df.iloc[i:end_i + 1]
        if spike_dir > 0:
            reversal_mask = window["close"] < (spike_bar["close"] - 0.3 * spike_range)
        else:
            reversal_mask = window["close"] > (spike_bar["close"] + 0.3 * spike_range)
        if reversal_mask.any():
            rev_idx = window.index[reversal_mask.values][0]
            df.loc[ts:rev_idx, "label"] = "manipulation"
            manip_periods.append((ts, rev_idx))
    df["label"] = df["label"].ffill(limit=2).bfill(limit=2)
    return df, manip_periods

def summarize_cycles(df):
    s = df["label"].fillna("none")
    runs = []
    if s.empty:
        return []
    prev = s.iloc[0]
    start = s.index[0]
    for t, lab in s.iloc[1:].items():
        if lab != prev:
            runs.append((start, t, prev))
            start = t
            prev = lab
    runs.append((start, s.index[-1], s.iloc[-1]))
    cycles = []
    for st, en, lab in runs:
        if lab == "none":
            continue
        seg = df.loc[st:en]
        duration_min = (seg.index[-1] - seg.index[0]).total_seconds() / 60.0
        point_move = seg["close"].iloc[-1] - seg["close"].iloc[0]
        cycles.append({
            "start": str(st), "end": str(en), "label": lab,
            "duration_min": round(duration_min, 2),
            "point_move": round(point_move, 2),
            "abs_point_move": round(abs(point_move), 2),
            "avg_volume": int(seg["volume"].mean()),
            "bars": len(seg)
        })
    return cycles

@app.get("/")
def read_root():
    return {"msg": "hello from Railway 👋"}

@app.get("/raw")
def api_raw(period: str = Query("7d")):
    df = fetch_5m(period=period)
    return df.tail(1000).reset_index().to_dict(orient="records")

@app.get("/indicators")
def api_indicators(period: str = Query("7d")):
    df = fetch_5m(period=period)
    df = compute_indicators(df)
    return df.reset_index().tail(1000).to_dict(orient="records")

@app.get("/cycles")
def api_cycles(period: str = Query("7d")):
    df = fetch_5m(period=period)
    df = compute_indicators(df)
    df_labeled, manip_windows = detect_amd(df)
    cycles = summarize_cycles(df_labeled)
    return {
        "cycles": cycles,
        "manip_windows": [(str(a), str(b)) for a, b in manip_windows]
    }

@app.get("/summary")
def api_summary(period: str = Query("7d")):
    df = fetch_5m(period=period)
    df = compute_indicators(df)
    df_labeled, _ = detect_amd(df)
    cycles = summarize_cycles(df_labeled)
    manip = [c for c in cycles if c["label"] == "manipulation"]
    avg_manip = round(np.mean([c["abs_point_move"] for c in manip]), 2) if manip else None
    return {
        "period": period,
        "total_cycles": len(cycles),
        "manipulation_count": len(manip),
        "avg_manipulation_size": avg_manip
    }