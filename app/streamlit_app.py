# streamlit_app.py
# ------------------------------------------------------------
# Streamlit wrapper for the golden-source backtester.
# Strategy / detection / simulation logic lives in engine.py unchanged.
# ------------------------------------------------------------

from __future__ import annotations

import streamlit as st
import pandas as pd

from engine import (
    get_hsi_universe,
    get_sp500_universe,
    get_sti_universe,
    run_country_backtest,
    default_country_configs,
    _HAS_NUMBA,
)

st.set_page_config(page_title="Weekly Swing Backtester (HK/US/SG)", layout="wide")

st.title("📈 Weekly Swing Multi-Country Backtester")
st.caption("Golden-source logic. Streamlit UI wrapper only (no strategy changes).")

with st.expander("Runtime notes", expanded=False):
    st.write(
        "- Universes: US & HK scraped from Wikipedia; SG is hardcoded.\n"
        "- Prices: yfinance (auto_adjust=False).\n"
        "- Caching: rolling parquet per country stored in app folder.\n"
        "- If Wikipedia or Yahoo changes/blocks, the run can fail (external dependency)."
    )
    st.write(f"Numba detected: `{_HAS_NUMBA}`")

# --- Controls
cfgs = default_country_configs()

colA, colB, colC, colD = st.columns([1, 1, 1, 1])
with colA:
    start_date = st.text_input("Start (YYYY-MM-DD)", value="2016-01-01")
with colB:
    end_date = st.text_input("End (blank = today)", value="")
with colC:
    n_signal_days = st.number_input("Current signals window (trading days)", min_value=1, max_value=30, value=5, step=1)
with colD:
    countries = st.multiselect("Countries", ["HK", "US", "SG"], default=["HK", "US", "SG"])

end = None if end_date.strip() == "" else end_date.strip()

run = st.button("Run backtests", type="primary")

# --- Cached universes build (so repeated UI tweaks don’t re-scrape unless needed)
@st.cache_data(show_spinner=False, ttl=60 * 60)  # 1 hour
def _load_universes():
    hk = get_hsi_universe()
    us = get_sp500_universe()
    sg = get_sti_universe()
    return hk, us, sg

if run:
    with st.spinner("Building universes..."):
        hk_uni, us_uni, sg_uni = _load_universes()

    uni_map = {"HK": hk_uni, "US": us_uni, "SG": sg_uni}

    results = {}
    for c in countries:
        with st.spinner(f"Running {c} backtest... (this can take time on first run)"):
            res = run_country_backtest(
                country=c,
                universe=uni_map[c],
                cfg=cfgs[c],
                start=start_date.strip(),
                end=end,
                n_signal_days=int(n_signal_days),
            )
            results[c] = res

    st.success("Done.")

    # --- Display
    for c in countries:
        res = results[c]
        st.header(f"{c} Results")

        top1, top2, top3 = st.columns([1, 1, 1])
        with top1:
            st.metric("Universe tickers", int(res["universe"]["Ticker"].nunique()))
        with top2:
            st.metric("Closed trades", int(res["trades"].shape[0]) if res["trades"] is not None else 0)
        with top3:
            st.metric("Signals rows", int(res["signals"].shape[0]) if res["signals"] is not None else 0)

        tabs = st.tabs([
            "Overall Summary",
            "By Setup & Entry",
            "Exit Reasons",
            "Yearly",
            "Score Buckets",
            "Market Cap Buckets",
            "Current Signals",
            "Trades",
            "Entries",
        ])

        with tabs[0]:
            st.dataframe(res["summary_overall"], use_container_width=True)

        with tabs[1]:
            st.dataframe(res["summary_by"], use_container_width=True)

        with tabs[2]:
            st.dataframe(res["exit_breakdown"], use_container_width=True)

        with tabs[3]:
            st.dataframe(res["summary_by_year"], use_container_width=True)

        with tabs[4]:
            st.dataframe(res["score_stats"], use_container_width=True)

        with tabs[5]:
            st.dataframe(res["mcap_stats"], use_container_width=True)

        with tabs[6]:
            st.dataframe(res["current_signals"], use_container_width=True)

        with tabs[7]:
            st.dataframe(res["trades"], use_container_width=True)

        with tabs[8]:
            st.dataframe(res["entries"], use_container_width=True)

        st.divider()
