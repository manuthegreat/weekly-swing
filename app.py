from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ARTIFACTS_DIR = Path("artifacts")


@st.cache_data(show_spinner=False)
def load_parquet(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_parquet(path)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_json(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def artifact_dir(country: str) -> Path:
    return ARTIFACTS_DIR / country.lower()


def _format_market_cap(value: float) -> str:
    try:
        return f"{float(value):,.2f}"
    except (TypeError, ValueError):
        return ""


def _format_bucket_label(raw: str) -> str:
    cleaned = raw.strip()
    if not cleaned:
        return ""
    cleaned = cleaned.replace("(", "").replace(")", "").replace("[", "").replace("]", "")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 2:
        return raw
    try:
        low = float(parts[0])
        high = float(parts[1])
    except ValueError:
        return raw
    return f"{_format_market_cap(low)} – {_format_market_cap(high)}"


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

    current = load_parquet(artifacts / "current_signals.parquet")
    summary = load_parquet(artifacts / "summary_overall.parquet")
    summary_by = load_parquet(artifacts / "summary_by_setup.parquet")
    exit_breakdown = load_parquet(artifacts / "exit_breakdown.parquet")
    summary_by_year = load_parquet(artifacts / "summary_by_year.parquet")
    trades = load_parquet(artifacts / "trades.parquet")
    entries = load_parquet(artifacts / "entries.parquet")
    heatmap = load_parquet(artifacts / "heatmap_score_x_mcap_winrate.parquet")
    heatmap_long = load_parquet(artifacts / "heatmap_score_x_mcap_winrate_long.parquet")

    st.header("Current Signals")
    if current.empty:
        st.info("No current signals available. Run the pipeline to generate artifacts.")
    else:
        if "market_cap" in current.columns:
            styled_current = current.style.format({"market_cap": _format_market_cap})
            st.dataframe(styled_current, use_container_width=True)
        else:
            st.dataframe(current, use_container_width=True)

    st.divider()

    st.divider()

    st.header("Backtest Results")
    summary_col, setup_col = st.columns(2)
    with summary_col:
        st.subheader("Overall Summary")
        if summary.empty:
            st.info("No summary data available. Run the pipeline to generate artifacts.")
        else:
            st.dataframe(summary, use_container_width=True)

    with setup_col:
        st.subheader("Summary by Setup")
        if summary_by.empty:
            st.write("No setup breakdown yet.")
        else:
            st.dataframe(summary_by, use_container_width=True)

    exit_col, year_col = st.columns(2)
    with exit_col:
        st.subheader("Exit Reasons")
        if exit_breakdown.empty:
            st.write("No exit breakdown yet.")
        else:
            st.dataframe(exit_breakdown, use_container_width=True)

    with year_col:
        st.subheader("Summary by Year")
        if summary_by_year.empty:
            st.write("No yearly summary yet.")
        else:
            st.dataframe(summary_by_year, use_container_width=True)

    st.subheader("Trades & Entries")
    trades_tab, entries_tab = st.tabs(["Trades", "Entries"])
    with trades_tab:
        if trades.empty:
            st.write("No trades yet.")
        else:
            st.dataframe(trades, use_container_width=True)
    with entries_tab:
        if entries.empty:
            st.write("No entries yet.")
        else:
            st.dataframe(entries, use_container_width=True)

    st.divider()

    st.header("Heatmap (Score x Market Cap)")
    if heatmap.empty and heatmap_long.empty:
        st.info("Heatmap artifacts not found yet. Run the pipeline to generate them.")
    else:
        if not heatmap_long.empty:
            chart_df = heatmap_long.copy()
            chart_df["metric"] = pd.to_numeric(chart_df["metric"], errors="coerce")
            chart_df = chart_df.dropna(subset=["metric"])
            if chart_df.empty:
                st.write("No heatmap data available.")
            else:
                chart_df["mcap_bucket_label"] = chart_df["mcap_bucket"].map(_format_bucket_label)
                ordered_labels = list(dict.fromkeys(chart_df["mcap_bucket_label"].tolist()))
                chart_df["mcap_bucket_label"] = pd.Categorical(
                    chart_df["mcap_bucket_label"],
                    categories=ordered_labels,
                    ordered=True,
                )
                st.vega_lite_chart(
                    chart_df,
                    {
                        "mark": {"type": "rect"},
                        "encoding": {
                            "x": {"field": "score_bin", "type": "ordinal", "title": "Score Bin"},
                            "y": {
                                "field": "mcap_bucket_label",
                                "type": "ordinal",
                                "title": "Market Cap Bucket",
                                "axis": {"labelLimit": 300},
                            },
                            "color": {
                                "field": "metric",
                                "type": "quantitative",
                                "title": "Win Rate (%)",
                                "scale": {"scheme": "redyellowgreen"},
                            },
                            "tooltip": [
                                {"field": "score_bin", "type": "ordinal", "title": "Score Bin"},
                                {"field": "mcap_bucket_label", "type": "ordinal", "title": "Market Cap"},
                                {"field": "metric", "type": "quantitative", "title": "Win Rate (%)"},
                                {"field": "trades", "type": "quantitative", "title": "Trades"},
                            ],
                        },
                    },
                    use_container_width=True,
                )
        elif not heatmap.empty:
            st.dataframe(heatmap, use_container_width=True)


if __name__ == "__main__":
    main()
