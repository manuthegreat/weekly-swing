from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")
CACHE_DIR = Path("cache_daily")


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    return pd.read_json(path, typ="series").to_dict()


def artifact_dir(country: str) -> Path:
    return ARTIFACTS_DIR / country.lower()


def main() -> None:
    st.set_page_config(page_title="Weekly Swing", layout="wide")
    st.title("Weekly Swing Dashboard")

    country = st.sidebar.selectbox("Country", ["HK", "US", "SG"], index=0)
    artifacts = artifact_dir(country)
    meta = load_json(artifacts / "meta.json")

    if meta:
        st.sidebar.markdown("### Latest Run")
        st.sidebar.write(f"Run date: {meta.get('run_date', 'N/A')}")
        st.sidebar.write(f"Last cached bar: {meta.get('last_cached_bar', 'N/A')}")
        st.sidebar.write(f"Expected last bar: {meta.get('expected_last_bar', 'N/A')}")

    summary = load_parquet(artifacts / "summary_overall.parquet")
    summary_by = load_parquet(artifacts / "summary_by_setup.parquet")
    trades = load_parquet(artifacts / "trades.parquet")
    entries = load_parquet(artifacts / "entries.parquet")
    current = load_parquet(artifacts / "current_signals.parquet")
    signals = load_parquet(artifacts / "signals.parquet")

    st.header("Summary")
    if summary.empty:
        st.info("No summary data available. Run the pipeline to generate artifacts.")
    else:
        st.dataframe(summary, use_container_width=True)

    col_a, col_b = st.columns(2)
    with col_a:
        st.subheader("Summary by Setup")
        if summary_by.empty:
            st.write("No setup breakdown yet.")
        else:
            st.dataframe(summary_by, use_container_width=True)
    with col_b:
        st.subheader("Current Signals")
        if current.empty:
            st.write("No current signals.")
        else:
            st.dataframe(current, use_container_width=True)

    st.header("Trades & Entries")
    col_c, col_d = st.columns(2)
    with col_c:
        st.subheader("Recent Trades")
        if trades.empty:
            st.write("No trades yet.")
        else:
            st.dataframe(trades.tail(200), use_container_width=True)
    with col_d:
        st.subheader("Recent Entries")
        if entries.empty:
            st.write("No entries yet.")
        else:
            st.dataframe(entries.tail(200), use_container_width=True)

    st.header("Signals")
    if signals.empty:
        st.write("No signals data available.")
    else:
        st.dataframe(signals.tail(200), use_container_width=True)

    st.header("Price Chart")
    cache_path = CACHE_DIR / f"{country}_daily.parquet"
    ohlc = load_parquet(cache_path)
    if ohlc.empty:
        st.info("No cached OHLC data available yet.")
        return

    tickers = sorted(ohlc["ticker"].dropna().unique().tolist())
    if not tickers:
        st.info("No tickers in cache.")
        return

    ticker = st.selectbox("Ticker", tickers, index=0)
    series = ohlc[ohlc["ticker"] == ticker].sort_values("date")
    if series.empty:
        st.write("No data for selected ticker.")
        return

    chart = series.set_index("date")["close"]
    st.line_chart(chart, height=300)


if __name__ == "__main__":
    main()
