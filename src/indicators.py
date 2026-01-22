from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import pandas as pd

from src.config import BacktestConfig

try:
    from numba import njit  # type: ignore

    @njit(cache=True)
    def _hh_and_low_since_hh_numba(high, low, window):
        n = len(high)
        hh = np.empty(n, dtype=np.float64)
        low_since = np.empty(n, dtype=np.float64)

        for i in range(n):
            hh[i] = np.nan
            low_since[i] = np.nan
            if i < window - 1:
                continue

            s = i - window + 1

            max_h = high[s]
            k = 0
            for j in range(1, window):
                v = high[s + j]
                if v > max_h:
                    max_h = v
                    k = j

            min_l = low[s + k]
            for j in range(k, window):
                v = low[s + j]
                if v < min_l:
                    min_l = v

            hh[i] = max_h
            low_since[i] = min_l

        return hh, low_since

    _HAS_NUMBA = True
except Exception:
    njit = None  # type: ignore
    _HAS_NUMBA = False


def _hh_and_low_since_hh_py(high: np.ndarray, low: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray]:
    n = len(high)
    hh = np.full(n, np.nan, dtype=float)
    low_since = np.full(n, np.nan, dtype=float)

    for i in range(n):
        if i < window - 1:
            continue
        s = i - window + 1
        win_high = high[s : i + 1]
        win_low = low[s : i + 1]

        k = int(np.argmax(win_high))
        hh_i = float(win_high[k])
        low_i = float(np.min(win_low[k:]))

        hh[i] = hh_i
        low_since[i] = low_i

    return hh, low_since


def add_indicators(df: pd.DataFrame, cfg: Optional[BacktestConfig] = None) -> pd.DataFrame:
    cfg = cfg or BacktestConfig()
    df = df.sort_values(["ticker", "date"]).copy()

    if df["ticker"].dtype.name != "category":
        df["ticker"] = df["ticker"].astype("category")

    if "turnover" not in df.columns:
        df["turnover"] = df["close"] * df["volume"]

    g = df.groupby("ticker", observed=True, sort=False)

    close = df["close"].astype(float)
    df["sma20"] = g["close"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["sma50"] = g["close"].rolling(50, min_periods=50).mean().reset_index(level=0, drop=True)
    df["sma20_slope5"] = df["sma20"] - g["sma20"].shift(5)

    df["prev_close"] = g["close"].shift(1)

    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["prev_close"].astype(float)

    tr1 = (high - low).to_numpy()
    tr2 = (high - prev_close).abs().to_numpy()
    tr3 = (low - prev_close).abs().to_numpy()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))

    df["atr5"] = g["tr"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    df["atr20"] = g["tr"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    df["hh5"] = g["high"].rolling(5, min_periods=5).max().reset_index(level=0, drop=True)
    df["ll5"] = g["low"].rolling(5, min_periods=5).min().reset_index(level=0, drop=True)
    df["range5_pct"] = (df["hh5"] - df["ll5"]) / close

    df["hh20"] = g["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    df["ll20"] = g["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    df["range20_pct"] = (df["hh20"] - df["ll20"]) / close
    denom20 = (df["hh20"] - df["ll20"]).replace(0, np.nan)
    df["close_pos_20"] = (close - df["ll20"]) / denom20

    df["turnover_5d"] = g["turnover"].rolling(5, min_periods=5).sum().reset_index(level=0, drop=True)
    df["adv_turnover_20"] = g["turnover"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    df["avg_turnover_10d"] = (
        g["turnover"]
        .rolling(cfg.turnover_baseline_days, min_periods=cfg.turnover_baseline_days)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["baseline_5d_turnover"] = df["avg_turnover_10d"] * 5.0

    df["hh10_close"] = g["close"].rolling(10, min_periods=10).max().reset_index(level=0, drop=True)

    df["hh_fib"] = np.nan
    df["low_since_hh_fib"] = np.nan

    use_numba = _HAS_NUMBA

    for _, sub in df.groupby("ticker", sort=False, observed=True):
        if len(sub) < cfg.fib_lookback:
            continue

        hi = sub["high"].to_numpy(dtype=np.float64)
        lo = sub["low"].to_numpy(dtype=np.float64)

        if use_numba:
            hh, low_since = _hh_and_low_since_hh_numba(hi, lo, cfg.fib_lookback)  # type: ignore[name-defined]
        else:
            hh, low_since = _hh_and_low_since_hh_py(hi, lo, cfg.fib_lookback)

        df.loc[sub.index, "hh_fib"] = hh
        df.loc[sub.index, "low_since_hh_fib"] = low_since

    return df


def has_numba() -> bool:
    return _HAS_NUMBA
