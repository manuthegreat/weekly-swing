from __future__ import annotations

from dataclasses import dataclass


@dataclass
class BacktestConfig:
    # Portfolio constraints
    max_open_positions: int = 20
    max_entries_per_day: int = 10

    # Exit
    profit_target_mult: float = 1.10
    holding_days_max: int = 30

    # Costs
    commission_bps: float = 0.0
    slippage_bps: float = 0.0

    # --- A) Vol compression
    tight_range_5d_max: float = 0.12
    close_pos20_min: float = 0.60
    sma20_slope_floor_mult: float = -0.002

    # --- C) Trend pause (split)
    pause_days_min: int = 3
    pause_days_max: int = 7
    pause_near_high_frac: float = 0.97
    pause_range20_max: float = 0.22

    # C2) 38% shallow pullback from HH
    fib_lookback: int = 10
    fib_frac: float = 0.382

    # Guardrails for pullback level realism
    pullback_min_pct_below_close: float = 0.02
    pullback_max_pct_below_close: float = 0.25

    # Guard: HH shouldn't be stale / far away
    max_hh_dist_from_close: float = 0.20

    # Ranking baseline: last N trading days turnover (daily baseline)
    turnover_baseline_days: int = 10

    # Turnover confirmation (ratio of 5d turnover vs baseline 5d turnover)
    turnover_expansion_min_A: float = 1.0
    turnover_expansion_min_C: float = 1.0

    # Breakout confirmation buffer (e.g. 0.003 = 0.3%)
    breakout_buffer_pct: float = 0.0


UNIVERSE_RULES = {
    "HK": dict(
        mcap_min=5e9,
        mcap_max=250e9,
        adv_turnover_20_min=100e6,
    ),
    "US": dict(
        mcap_min=2e9,
        mcap_max=150e9,
        adv_turnover_20_min=75e6,
    ),
    "SG": dict(
        mcap_min=1e9,
        mcap_max=80e9,
        adv_turnover_20_min=10e6,
    ),
}


MARKET_TZ = {
    "US": "America/New_York",
    "HK": "Asia/Hong_Kong",
    "SG": "Asia/Singapore",
}

MARKET_BAR_READY_TIME = {
    "US": (18, 0),
    "HK": (17, 0),
    "SG": (18, 0),
}


def country_configs() -> dict[str, BacktestConfig]:
    return {
        "HK": BacktestConfig(
            max_open_positions=20,
            max_entries_per_day=10,
            profit_target_mult=1.10,
            holding_days_max=30,
            turnover_baseline_days=10,
            tight_range_5d_max=0.13,
            close_pos20_min=0.58,
            pause_near_high_frac=0.965,
            pause_range20_max=0.24,
            max_hh_dist_from_close=0.25,
            turnover_expansion_min_A=1.05,
            turnover_expansion_min_C=1.03,
            breakout_buffer_pct=0.0,
        ),
        "US": BacktestConfig(
            max_open_positions=20,
            max_entries_per_day=10,
            profit_target_mult=1.10,
            holding_days_max=30,
            turnover_baseline_days=10,
            tight_range_5d_max=0.09,
            close_pos20_min=0.65,
            pause_near_high_frac=0.985,
            pause_range20_max=0.18,
            max_hh_dist_from_close=0.15,
            turnover_expansion_min_A=1.20,
            turnover_expansion_min_C=1.15,
            breakout_buffer_pct=0.003,
        ),
        "SG": BacktestConfig(
            max_open_positions=20,
            max_entries_per_day=10,
            profit_target_mult=1.10,
            holding_days_max=30,
            turnover_baseline_days=10,
            tight_range_5d_max=0.11,
            close_pos20_min=0.60,
            pause_near_high_frac=0.970,
            pause_range20_max=0.22,
            max_hh_dist_from_close=0.20,
            turnover_expansion_min_A=1.15,
            turnover_expansion_min_C=1.10,
            breakout_buffer_pct=0.001,
        ),
    }
