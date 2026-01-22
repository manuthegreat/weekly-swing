from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import BacktestConfig
from src.indicators import add_indicators
from src.reporting import (
    summarize_exit_reasons,
    summarize_trades,
    summarize_trades_by_setup_entry,
    summarize_trades_by_year,
)
from src.setups import detect_setups_fast


@dataclass
class Position:
    ticker: str
    setup_tag: str
    entry_type: str
    signal_date: pd.Timestamp
    entry_date: pd.Timestamp
    entry_price: float
    stop_level: float
    max_exit_date: pd.Timestamp


def run_backtest(
    df_raw: pd.DataFrame,
    cfg: Optional[BacktestConfig] = None,
    indicators_precomputed: bool = False,
):
    cfg = cfg or BacktestConfig()
    df = df_raw.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if not indicators_precomputed:
        df = add_indicators(df, cfg)

    country_map = {}
    if "country" in df.columns:
        tmp = df[["ticker", "country"]].dropna().drop_duplicates(subset=["ticker"])
        country_map = dict(zip(tmp["ticker"].astype(str), tmp["country"].astype(str)))

    calendar = pd.Index(sorted(df["date"].unique()))
    by_date = {d: x for d, x in df.groupby("date")}

    def nth_trading_day(d0: pd.Timestamp, n: int) -> pd.Timestamp:
        idx = calendar.get_indexer([d0])[0]
        if idx == -1:
            idx = calendar.searchsorted(d0)
        idx2 = min(idx + (n - 1), len(calendar) - 1)
        return pd.Timestamp(calendar[idx2])

    signals = detect_setups_fast(df, cfg)
    signals_by_day = {d: x for d, x in signals.groupby("signal_date")} if not signals.empty else {}

    trades: List[Dict] = []
    entries: List[Dict] = []
    positions: Dict[str, Position] = {}

    for d in calendar:
        day_df = by_date.get(d)
        if day_df is None or day_df.empty:
            continue

        if positions:
            pos_tickers = list(positions.keys())
            todays = day_df[day_df["ticker"].isin(pos_tickers)].set_index("ticker")
            to_close: List[str] = []

            for tkr, pos in list(positions.items()):
                if tkr not in todays.index:
                    continue

                row = todays.loc[tkr]
                high = float(row["high"])
                close = float(row["close"])
                target_price = pos.entry_price * cfg.profit_target_mult

                exit_reason = None
                exit_price = None

                if high >= target_price:
                    exit_reason = "TARGET_10PCT"
                    exit_price = target_price
                elif close < pos.stop_level:
                    exit_reason = "STRUCTURE_BREAK"
                    exit_price = close
                elif d >= pos.max_exit_date:
                    exit_reason = "TIME_STOP_N_DAYS"
                    exit_price = close

                if exit_reason is not None:
                    costs = (cfg.commission_bps + cfg.slippage_bps) / 10000.0
                    gross_ret = (exit_price / pos.entry_price) - 1.0
                    net_ret = gross_ret - 2 * costs

                    trades.append(
                        {
                            "ticker": tkr,
                            "country": country_map.get(tkr, "NA"),
                            "setup_tag": pos.setup_tag,
                            "entry_type": pos.entry_type,
                            "signal_date": pos.signal_date,
                            "entry_date": pos.entry_date,
                            "entry_price": pos.entry_price,
                            "exit_date": d,
                            "exit_price": exit_price,
                            "exit_reason": exit_reason,
                            "pnl_pct_gross": gross_ret * 100.0,
                            "pnl_pct_net": net_ret * 100.0,
                        }
                    )
                    to_close.append(tkr)

            for tkr in to_close:
                positions.pop(tkr, None)

        loc = calendar.get_indexer([d])[0]
        if loc <= 0:
            continue
        prev_day = pd.Timestamp(calendar[loc - 1]).normalize()

        sig_df = signals_by_day.get(prev_day)
        if sig_df is None or sig_df.empty:
            continue

        open_slots = cfg.max_open_positions - len(positions)
        if open_slots <= 0:
            continue
        intake = min(cfg.max_entries_per_day, open_slots)

        sig_df = sig_df[~sig_df["ticker"].isin(list(positions.keys()))].copy()
        if sig_df.empty:
            continue

        sig_df = sig_df.sort_values("score", ascending=False)
        sig_df = sig_df.drop_duplicates(subset=["ticker"], keep="first")

        todays_rows = day_df.set_index("ticker")

        filled = 0
        for s in sig_df.itertuples(index=False):
            if filled >= intake:
                break

            tkr = str(s.ticker)
            if tkr not in todays_rows.index:
                continue
            if tkr in positions:
                continue

            setup_tag = str(s.setup_tag)
            entry_type = str(s.expected_entry_type)

            row = todays_rows.loc[tkr]
            o = float(row["open"])
            h = float(row["high"])
            l = float(row["low"])

            stop_level = float(s.stop_level)
            entry_price = None

            if entry_type == "BREAKOUT":
                breakout = float(s.breakout_level)
                trigger = breakout * (1.0 + cfg.breakout_buffer_pct)
                if h >= trigger:
                    entry_price = max(trigger, o)

            elif entry_type == "PULLBACK":
                pb = getattr(s, "pullback_level", np.nan)
                if pd.isna(pb):
                    continue
                limit_px = float(pb)
                if l <= limit_px:
                    entry_price = min(limit_px, o) if o < limit_px else limit_px

            if entry_price is None:
                continue

            max_exit_date = nth_trading_day(d, cfg.holding_days_max)

            positions[tkr] = Position(
                ticker=tkr,
                setup_tag=setup_tag,
                entry_type=entry_type,
                signal_date=prev_day,
                entry_date=d,
                entry_price=entry_price,
                stop_level=stop_level,
                max_exit_date=max_exit_date,
            )

            entries.append(
                {
                    "ticker": tkr,
                    "country": country_map.get(tkr, "NA"),
                    "setup_tag": setup_tag,
                    "entry_type": entry_type,
                    "signal_date": prev_day,
                    "entry_date": d,
                    "entry_price": entry_price,
                }
            )
            filled += 1

    trades_df = pd.DataFrame(trades)
    if not trades_df.empty:
        trades_df = trades_df.sort_values(["entry_date", "ticker"]).reset_index(drop=True)

    entries_df = pd.DataFrame(entries)
    if not entries_df.empty:
        entries_df = entries_df.sort_values(["entry_date", "ticker"]).reset_index(drop=True)

    summary_overall = summarize_trades(trades_df)
    summary_by = summarize_trades_by_setup_entry(trades_df)
    exit_breakdown = summarize_exit_reasons(trades_df)
    summary_by_year = summarize_trades_by_year(trades_df)

    return (
        trades_df,
        entries_df,
        summary_overall,
        summary_by,
        exit_breakdown,
        summary_by_year,
        signals,
    )
