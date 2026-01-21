import streamlit as st
import pandas as pd

from src.engine import (
    get_hsi_universe, get_sp500_universe, get_sti_universe,
    BacktestConfig,
    run_country_backtest,
)

st.set_page_config(page_title="Weekly Swing Multi Backtester", layout="wide")
st.title("Weekly Swing Multi Backtester (HK / US / SG)")
st.caption("Streamlit UI version of your backtester.")

st.sidebar.header("Run Settings")
start_date = st.sidebar.text_input("Start date (YYYY-MM-DD)", value="2016-01-01")
end_date = st.sidebar.text_input("End date (YYYY-MM-DD) (blank = today)", value="")
n_signal_days = st.sidebar.slider("Current signals window (trading days)", 1, 20, 5)

run_hk = st.sidebar.checkbox("Run HK", value=True)
run_us = st.sidebar.checkbox("Run US", value=True)
run_sg = st.sidebar.checkbox("Run SG", value=True)

st.sidebar.divider()
st.sidebar.subheader("Advanced")
show_raw_trades = st.sidebar.checkbox("Show raw trades table", value=False)
show_raw_signals = st.sidebar.checkbox("Show raw signals table", value=False)

cfg_hk = BacktestConfig(
    max_open_positions=20, max_entries_per_day=10, profit_target_mult=1.10, holding_days_max=30,
    turnover_baseline_days=10, tight_range_5d_max=0.13, close_pos20_min=0.58,
    pause_near_high_frac=0.965, pause_range20_max=0.24, max_hh_dist_from_close=0.25,
    turnover_expansion_min_A=1.05, turnover_expansion_min_C=1.03, breakout_buffer_pct=0.0,
)

cfg_us = BacktestConfig(
    max_open_positions=20, max_entries_per_day=10, profit_target_mult=1.10, holding_days_max=30,
    turnover_baseline_days=10, tight_range_5d_max=0.09, close_pos20_min=0.65,
    pause_near_high_frac=0.985, pause_range20_max=0.18, max_hh_dist_from_close=0.15,
    turnover_expansion_min_A=1.20, turnover_expansion_min_C=1.15, breakout_buffer_pct=0.003,
)

cfg_sg = BacktestConfig(
    max_open_positions=20, max_entries_per_day=10, profit_target_mult=1.10, holding_days_max=30,
    turnover_baseline_days=10, tight_range_5d_max=0.11, close_pos20_min=0.60,
    pause_near_high_frac=0.970, pause_range20_max=0.22, max_hh_dist_from_close=0.20,
    turnover_expansion_min_A=1.15, turnover_expansion_min_C=1.10, breakout_buffer_pct=0.001,
)

def _normalize_end(end_str: str):
    end_str = (end_str or "").strip()
    return None if end_str == "" else end_str

end_val = _normalize_end(end_date)

if st.button("Run backtests"):
    with st.spinner("Building universes..."):
        hk_uni = get_hsi_universe()
        us_uni = get_sp500_universe()
        sg_uni = get_sti_universe()

    countries = []
    if run_hk: countries.append(("HK", hk_uni, cfg_hk))
    if run_us: countries.append(("US", us_uni, cfg_us))
    if run_sg: countries.append(("SG", sg_uni, cfg_sg))

    if not countries:
        st.warning("Select at least one country in the sidebar.")
        st.stop()

    tabs = st.tabs([c[0] for c in countries])

    for tab, (country, uni, cfg) in zip(tabs, countries):
        with tab:
            st.subheader(f"{country} Results")

            with st.spinner(f"Running {country} backtest..."):
                result = run_country_backtest(
                    country=country,
                    universe=uni,
                    cfg=cfg,
                    start=start_date,
                    end=end_val,
                    n_signal_days=n_signal_days,
                )

            overall = result.get("summary_overall")
            if overall is not None and not overall.empty:
                row = overall.iloc[0].to_dict()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Trades", int(row.get("trades", 0)))
                c2.metric("Win rate %", f'{row.get("win_rate_%", float("nan")):.2f}')
                c3.metric("Expectancy %", f'{row.get("expectancy_%", float("nan")):.2f}')
                pf = row.get("profit_factor", float("nan"))
                c4.metric("Profit factor", f'{pf:.2f}' if pd.notna(pf) else "NA")

            st.markdown("### Overall (closed trades only)")
            st.dataframe(result.get("summary_overall"), use_container_width=True)

            st.markdown("### Summary by setup & entry")
            st.dataframe(result.get("summary_by"), use_container_width=True)

            st.markdown("### Exit reason breakdown")
            st.dataframe(result.get("exit_breakdown"), use_container_width=True)

            st.markdown("### Yearly stats")
            st.dataframe(result.get("summary_by_year"), use_container_width=True)

            st.markdown("### Score bucket stats")
            st.dataframe(result.get("score_stats"), use_container_width=True)

            st.markdown("### Market cap bucket stats")
            st.dataframe(result.get("mcap_stats"), use_container_width=True)

            st.markdown("### Current signals (consolidated)")
            st.dataframe(result.get("signals_grouped"), use_container_width=True)

            if show_raw_trades:
                st.markdown("### Raw trades table")
                st.dataframe(result.get("trades"), use_container_width=True)

            if show_raw_signals:
                st.markdown("### Raw signals table")
                st.dataframe(result.get("signals"), use_container_width=True)

            st.markdown("### Downloads")
            colA, colB = st.columns(2)
            with colA:
                trades_df = result.get("trades")
                if trades_df is not None and not trades_df.empty:
                    st.download_button(
                        "Download trades (CSV)",
                        data=trades_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{country}_trades.csv",
                        mime="text/csv",
                    )
            with colB:
                sig_df = result.get("signals_grouped")
                if sig_df is not None and not sig_df.empty:
                    st.download_button(
                        "Download current signals (CSV)",
                        data=sig_df.to_csv(index=False).encode("utf-8"),
                        file_name=f"{country}_current_signals.csv",
                        mime="text/csv",
                    )
else:
    st.info("Set options in the sidebar, then click **Run backtests**.")
