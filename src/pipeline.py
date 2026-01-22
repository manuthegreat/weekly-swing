from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Dict

import pandas as pd

from src.backtest import run_backtest
from src.config import country_configs
from src.data import load_or_build_cache
from src.reporting import (
    expected_last_daily_bar_date,
    summarize_heatmap_score_x_mcap,
    summarize_heatmap_score_x_mcap_long,
)
from src.setups import show_grouped_signals_last_n_trading_days
from src.universes import get_universe

ARTIFACTS_DIR = Path("artifacts")


def _artifact_dir(country: str) -> Path:
    return ARTIFACTS_DIR / country.lower()


def _load_meta(path: Path) -> Dict:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except json.JSONDecodeError:
        return {}


def _write_meta(path: Path, payload: Dict) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def run_country(country: str, mode: str, n_signal_days: int = 3) -> None:
    cfg = country_configs()[country]
    universe = get_universe(country)

    df = load_or_build_cache(country, universe, cfg, start="2016-01-01", end=None)

    artifact_dir = _artifact_dir(country)
    artifact_dir.mkdir(parents=True, exist_ok=True)
    meta_path = artifact_dir / "meta.json"

    meta = _load_meta(meta_path)
    today = datetime.utcnow().date().isoformat()
    if mode == "daily" and meta.get("run_date") == today:
        print(f"[{country}] Backtest already ran today ({today}); skipping.")
        return

    trades, entries, summary_overall, summary_by, exit_breakdown, summary_by_year, signals = run_backtest(
        df, cfg, indicators_precomputed=True
    )

    trading_calendar = pd.Index(sorted(pd.to_datetime(df["date"]).dt.normalize().unique()))
    current = show_grouped_signals_last_n_trading_days(
        signals,
        trading_calendar=trading_calendar,
        n_days=n_signal_days,
    )
    if current is not None and not current.empty:
        current = current.sort_values(
            by=["signal_date", "market_cap", "score_max"],
            ascending=[False, True, False],
            na_position="last",
        ).reset_index(drop=True)

    heatmap_combined = pd.DataFrame()
    heatmap_long = pd.DataFrame(columns=["mcap_bucket", "score_bin", "metric", "trades"])
    if not trades.empty and not signals.empty:
        join_cols = ["signal_date", "ticker", "setup_tag"]
        trades_enriched = trades.copy()
        trades_enriched["signal_date"] = pd.to_datetime(trades_enriched["signal_date"]).dt.normalize()
        signals_enriched = signals[join_cols + ["score", "market_cap"]].copy()
        signals_enriched["signal_date"] = pd.to_datetime(signals_enriched["signal_date"]).dt.normalize()
        trades_enriched = trades_enriched.merge(signals_enriched, on=join_cols, how="left")
        heatmap_combined = summarize_heatmap_score_x_mcap(trades_enriched, mcap_q=10, value="win_rate_%")
        heatmap_long = summarize_heatmap_score_x_mcap_long(trades_enriched, mcap_q=10)

    trades.to_parquet(artifact_dir / "trades.parquet", index=False)
    entries.to_parquet(artifact_dir / "entries.parquet", index=False)
    summary_overall.to_parquet(artifact_dir / "summary_overall.parquet", index=False)
    summary_by.to_parquet(artifact_dir / "summary_by_setup.parquet", index=False)
    exit_breakdown.to_parquet(artifact_dir / "exit_breakdown.parquet", index=False)
    summary_by_year.to_parquet(artifact_dir / "summary_by_year.parquet", index=False)
    signals.to_parquet(artifact_dir / "signals.parquet", index=False)
    current.to_parquet(artifact_dir / "current_signals.parquet", index=False)
    heatmap_combined.to_parquet(
        artifact_dir / "heatmap_score_x_mcap_winrate.parquet",
        index=False,
    )
    heatmap_long.to_parquet(
        artifact_dir / "heatmap_score_x_mcap_winrate_long.parquet",
        index=False,
    )

    last_cached = pd.to_datetime(df["date"]).max().date().isoformat() if not df.empty else None
    expected_last = expected_last_daily_bar_date(country).date().isoformat()

    _write_meta(
        meta_path,
        {
            "run_date": today,
            "run_ts": datetime.utcnow().isoformat() + "Z",
            "country": country,
            "last_cached_bar": last_cached,
            "expected_last_bar": expected_last,
            "cache_rows": int(len(df)),
            "trades": int(len(trades)),
            "signals": int(len(signals)),
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Weekly Swing daily pipeline")
    parser.add_argument(
        "mode",
        nargs="?",
        default="daily",
        choices=["daily", "full"],
        help="daily skips repeat backtests; full forces rerun",
    )
    args = parser.parse_args()

    for country in ["HK", "US", "SG"]:
        run_country(country, args.mode)


if __name__ == "__main__":
    main()
