from __future__ import annotations

import numpy as np
import pandas as pd

from src.config import BacktestConfig, UNIVERSE_RULES


def _universe_filter(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    needed = [
        "sma20",
        "sma50",
        "sma20_slope5",
        "atr5",
        "atr20",
        "hh5",
        "ll5",
        "range5_pct",
        "range20_pct",
        "close_pos_20",
        "turnover_5d",
        "baseline_5d_turnover",
        "adv_turnover_20",
        "hh10_close",
        "hh_fib",
        "low_since_hh_fib",
        "market_cap",
        "country",
    ]

    d = df.dropna(subset=needed).copy()
    if d.empty:
        return d

    c = d["country"].astype(str)

    mcap_min = c.map(lambda x: UNIVERSE_RULES.get(x, {}).get("mcap_min", np.nan)).astype(float)
    mcap_max = c.map(lambda x: UNIVERSE_RULES.get(x, {}).get("mcap_max", np.nan)).astype(float)
    adv_min = c.map(lambda x: UNIVERSE_RULES.get(x, {}).get("adv_turnover_20_min", np.nan)).astype(float)

    mc = d["market_cap"].astype(float)
    adv = d["adv_turnover_20"].astype(float)

    mask = (mc >= mcap_min) & (mc <= mcap_max) & (adv >= adv_min)
    return d.loc[mask].copy()


def _score_signals_AC(out: pd.DataFrame) -> pd.DataFrame:
    out = out.copy()

    denom = out["baseline_5d_turnover"].replace(0, np.nan)
    out["turnover_expansion"] = (out["turnover_5d"] / denom).replace([np.inf, -np.inf], np.nan)

    out["range_compression_score"] = 1.0 / (out["range5_pct"].clip(lower=1e-6))
    out["dist_20dma"] = (out["close"] - out["sma20"]) / out["sma20"]

    for col in ["turnover_expansion", "range_compression_score", "dist_20dma"]:
        out[col + "_r"] = out.groupby("signal_date", observed=True)[col].rank(pct=True)

    out["score"] = (
        0.5 * out["turnover_expansion_r"]
        + 0.3 * out["range_compression_score_r"]
        + 0.2 * out["dist_20dma_r"]
    )
    return out


def detect_setups_fast(df_raw: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    df = df_raw.sort_values(["ticker", "date"]).copy()
    df = _universe_filter(df, cfg)
    if df.empty:
        return pd.DataFrame()

    denom = df["baseline_5d_turnover"].replace(0, np.nan)
    df["turnover_expansion"] = (df["turnover_5d"] / denom).replace([np.inf, -np.inf], np.nan)

    near_high = df["close"] >= (cfg.pause_near_high_frac * df["hh10_close"])
    contraction = (df["range5_pct"] < df["range20_pct"]) | (df["atr5"] < df["atr20"])
    df["pause_day"] = (near_high & contraction).astype(int)

    df["pause_count"] = (
        df.groupby("ticker", observed=True)["pause_day"]
        .transform(lambda x: x.rolling(cfg.pause_days_max, min_periods=cfg.pause_days_max).sum())
    )

    trend_up = (
        (df["close"] > df["sma20"])
        & (df["sma20"] > df["sma50"])
        & (df["sma20_slope5"] >= cfg.sma20_slope_floor_mult * df["sma20"])
    )

    vol_compress = (df["atr5"] < df["atr20"]) & (df["range5_pct"] < df["range20_pct"])
    tight = df["range5_pct"] <= cfg.tight_range_5d_max
    location = df["close_pos_20"] >= cfg.close_pos20_min
    vol_confirm_A = df["turnover_expansion"] >= cfg.turnover_expansion_min_A
    mask_A = trend_up & vol_compress & tight & location & vol_confirm_A

    A = df.loc[mask_A, :].copy()
    if not A.empty:
        A["signal_date"] = A["date"].dt.normalize()
        A["setup_tag"] = "VOL_COMPRESSION_BREAKOUT"
        A["expected_entry_type"] = "BREAKOUT"
        A["breakout_level"] = A["hh5"]
        A["pullback_level"] = np.nan
        A["stop_level"] = A["ll5"]

    mask_pause_base = (
        trend_up
        & (df["pause_count"] >= cfg.pause_days_min)
        & (df["pause_count"] <= cfg.pause_days_max)
        & (df["range20_pct"] <= cfg.pause_range20_max)
        & (df["hh_fib"] <= df["close"] * (1.0 + cfg.max_hh_dist_from_close))
    )
    PB = df.loc[mask_pause_base, :].copy()
    if not PB.empty:
        PB["signal_date"] = PB["date"].dt.normalize()

    C1 = PB.copy()
    if not C1.empty:
        vol_confirm_C = C1["turnover_expansion"] >= cfg.turnover_expansion_min_C
        C1 = C1[vol_confirm_C].copy()
        C1["setup_tag"] = "TREND_PAUSE_BREAKOUT"
        C1["expected_entry_type"] = "BREAKOUT"
        C1["breakout_level"] = C1["hh5"]
        C1["pullback_level"] = np.nan
        C1["stop_level"] = C1["ll5"]

    C2 = PB.copy()
    if not C2.empty:
        HH = C2["hh_fib"].astype(float)
        LSH = C2["low_since_hh_fib"].astype(float)
        rng = (HH - LSH).clip(lower=1e-9)
        fib38 = HH - cfg.fib_frac * rng

        C2["setup_tag"] = "TREND_PAUSE_38PULLBACK"
        C2["expected_entry_type"] = "PULLBACK"
        C2["breakout_level"] = C2["hh5"]
        C2["pullback_level"] = fib38
        C2["stop_level"] = C2["ll5"]

        close = C2["close"].astype(float)
        min_level = close * (1.0 - cfg.pullback_max_pct_below_close)
        max_level = close * (1.0 - cfg.pullback_min_pct_below_close)
        C2 = C2[(C2["pullback_level"] >= min_level) & (C2["pullback_level"] <= max_level)].copy()

    frames = [x for x in [A, C1, C2] if x is not None and not x.empty]
    if not frames:
        return pd.DataFrame()

    out = pd.concat(frames, ignore_index=True)
    out = out.sort_values(["signal_date", "ticker", "setup_tag"])
    out = out.drop_duplicates(subset=["signal_date", "ticker", "setup_tag"], keep="first")

    out = _score_signals_AC(out)

    keep_cols = [
        "signal_date",
        "ticker",
        "country",
        "setup_tag",
        "expected_entry_type",
        "score",
        "breakout_level",
        "pullback_level",
        "stop_level",
        "close",
        "sma20",
        "sma50",
        "dist_20dma",
        "range5_pct",
        "range20_pct",
        "close_pos_20",
        "atr5",
        "atr20",
        "turnover_expansion",
        "adv_turnover_20",
        "market_cap",
        "turnover_5d",
        "baseline_5d_turnover",
        "avg_turnover_10d",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return (
        out[keep_cols]
        .sort_values(["signal_date", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )


def show_grouped_signals_last_n_trading_days(
    signals: pd.DataFrame,
    trading_calendar: pd.Index,
    n_days: int = 5,
) -> pd.DataFrame:
    if signals is None or signals.empty:
        return pd.DataFrame()
    if trading_calendar is None or len(trading_calendar) == 0:
        return pd.DataFrame()

    s = signals.copy()
    s["signal_date"] = pd.to_datetime(s["signal_date"]).dt.normalize()

    cal = pd.to_datetime(pd.Index(trading_calendar)).normalize()
    last_dates = cal[-n_days:]

    window = s[s["signal_date"].isin(last_dates)].copy()
    if window.empty:
        return pd.DataFrame()

    def _join_unique(vals):
        return "; ".join(sorted(pd.Series(vals).dropna().astype(str).unique()))

    latest = (
        window.sort_values(["ticker", "signal_date", "score"], ascending=[True, True, False])
        .groupby("ticker", observed=True, sort=False)
        .tail(1)
    )

    out = (
        latest
        .set_index("ticker")
        .join(window.groupby("ticker", observed=True)["score"].max().rename("score_max"))
        .join(window.groupby("ticker", observed=True)["setup_tag"].agg(_join_unique).rename("setups"))
        .reset_index()
    )

    cols = [
        "ticker",
        "country",
        "signal_date",
        "setups",
        "score_max",
        "breakout_level",
        "pullback_level",
        "stop_level",
        "close",
        "market_cap",
    ]
    cols = [c for c in cols if c in out.columns]

    return (
        out[cols]
        .sort_values("score_max", ascending=False)
        .reset_index(drop=True)
    )
