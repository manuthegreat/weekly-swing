# engine.py
# ------------------------------------------------------------
# Multi-Setup + Multi-Entry Backtester (HK + US + SG)
# - Runs 3 SEPARATE backtests (one per country) ✅
# - Prints 3 SEPARATE "current signals" lists ✅ (consolidated by ticker)
# - Data: Yahoo (yfinance) direct download
# - Prices: auto_adjust=False (raw OHLC)
# - Ranking baseline: last 10 trading days of turnover (daily)
# ------------------------------------------------------------

from __future__ import annotations
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from io import StringIO
from datetime import datetime
import time

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- Cache directory (packaging-safe, no strategy logic change)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(_THIS_DIR, "cache_daily")
os.makedirs(CACHE_DIR, exist_ok=True)

def cache_path(country: str) -> str:
    return os.path.join(CACHE_DIR, f"{country}_daily.parquet")  # single rolling file

def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    return df

def _safe_start_for_incremental(last_date: pd.Timestamp, overlap_days: int = 180) -> str:
    start_dt = (pd.Timestamp(last_date) - pd.Timedelta(days=overlap_days)).date()
    return str(start_dt)

def build_or_update_daily_cache(
    country: str,
    universe: pd.DataFrame,
    cfg: "BacktestConfig",
    full_start: str = "2016-01-01",
    end: Optional[str] = None,
    overlap_calendar_days: int = 30,
) -> pd.DataFrame:
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    path = cache_path(country)
    tickers = universe["Ticker"].dropna().astype(str).unique().tolist()

    if not os.path.exists(path):
        print(f"[{country}] Cache missing. Building full -> {path}")
        df_raw = get_data_from_yahoo(
            tickers=tickers, country=country, start=full_start, end=end,
            sleep_s=0.15, max_retries=2,
            chunk_size=250 if country == "US" else 40, threads=8,
        )
        df_ind = add_indicators(df_raw, cfg)
        df_ind.to_parquet(path, index=False)
        return _ensure_datetime(df_ind)

    print(f"[{country}] Loading cache -> {path}")
    df_cached = _ensure_datetime(pd.read_parquet(path))

    if df_cached.empty:
        os.remove(path)
        return build_or_update_daily_cache(country, universe, cfg, full_start=full_start, end=end)

    last_date = pd.to_datetime(df_cached["date"]).max()
    inc_start = _safe_start_for_incremental(last_date, overlap_days=overlap_calendar_days)
    print(f"[{country}] Incremental refresh: {inc_start} -> {end} (last cached: {last_date.date()})")

    df_new = get_data_from_yahoo(
        tickers=tickers, country=country, start=inc_start, end=end,
        sleep_s=0.15, max_retries=2,
        chunk_size=250 if country == "US" else 40, threads=8,
    )
    df_new = _ensure_datetime(df_new)

    base_cols = ["date","ticker","open","high","low","close","volume","turnover","market_cap","country"]

    df_merged = pd.concat([df_cached[base_cols], df_new[base_cols]], ignore_index=True)
    df_merged = (
        df_merged.drop_duplicates(subset=["date","ticker"], keep="last")
        .sort_values(["ticker","date"])
        .reset_index(drop=True)
    )

    cutoff = pd.to_datetime(inc_start)
    df_old_part = df_cached[df_cached["date"] < cutoff].copy()

    df_tail_raw = df_merged[df_merged["date"] >= cutoff].copy()
    df_tail_ind = add_indicators(df_tail_raw, cfg)

    df_final = pd.concat([df_old_part, df_tail_ind], ignore_index=True)
    df_final = (
        df_final.drop_duplicates(subset=["date","ticker"], keep="last")
        .sort_values(["date","ticker"])
        .reset_index(drop=True)
    )

    df_final.to_parquet(path, index=False)
    return df_final

def load_or_build_cache(country: str, universe: pd.DataFrame, cfg: "BacktestConfig", start="2016-01-01", end=None):
    return build_or_update_daily_cache(country, universe, cfg, full_start=start, end=end)


# =========================
# Optional speedups (Numba)
# =========================
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


# =========================
# Config
# =========================
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
        mcap_max=80e9,
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


# =========================
# Robust HTTP helper
# =========================
def http_get_with_retries(
    url: str,
    headers: Optional[dict] = None,
    timeout: int = 30,
    max_retries: int = 5,
    backoff_s: float = 1.25,
) -> requests.Response:
    last_err: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=timeout)
            r.raise_for_status()
            return r
        except Exception as e:
            last_err = e
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"GET failed after {max_retries} retries: {url} | last_err={last_err}")


# =========================
# Universes
# =========================
def get_sp500_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = http_get_with_retries(url, headers=headers, timeout=30)
    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        if "Symbol" in t.columns:
            df = t.copy()
            break
    if df is None:
        raise ValueError("Could not find S&P 500 table on Wikipedia.")

    df["Ticker"] = df["Symbol"].astype(str).str.replace(".", "-", regex=False)
    df["Name"] = df["Security"]
    df["Sector"] = df["GICS Sector"]
    out = df[["Ticker", "Name", "Sector"]].copy()
    out["Country"] = "US"
    return out


def get_hsi_universe() -> pd.DataFrame:
    url = "https://en.wikipedia.org/wiki/Hang_Seng_Index"
    headers = {"User-Agent": "Mozilla/5.0"}
    r = http_get_with_retries(url, headers=headers, timeout=30)
    tables = pd.read_html(StringIO(r.text))

    df = None
    for t in tables:
        cols = [str(c).lower() for c in t.columns]
        if any(x in cols for x in ["ticker", "constituent", "sub-index", "sub index", "sehk", "code"]):
            df = t.copy()
            break
    if df is None:
        raise ValueError("No HSI table found on Wikipedia page.")

    df.columns = [str(c).lower() for c in df.columns]

    ticker_col = None
    for c in df.columns:
        if "sehk" in c or "ticker" in c or "code" in c:
            ticker_col = c
            break
    if ticker_col is None:
        raise ValueError("Could not find SEHK ticker column")

    df["Ticker"] = (
        df[ticker_col]
        .astype(str)
        .str.extract(r"(\d+)")
        .iloc[:, 0]
        .astype(int)
        .astype(str)
        .str.zfill(4)
        + ".HK"
    )

    if "name" in df.columns:
        name_col = "name"
    else:
        possible = [c for c in df.columns if c not in [ticker_col, "sub-index", "sub index"]]
        name_col = possible[0] if possible else ticker_col

    df["Name"] = df[name_col]

    if "sub-index" in df.columns:
        df["Sector"] = df["sub-index"]
    elif "sub index" in df.columns:
        df["Sector"] = df["sub index"]
    elif "industry" in df.columns:
        df["Sector"] = df["industry"]
    else:
        df["Sector"] = None

    out = df[["Ticker", "Name", "Sector"]].copy()
    out["Country"] = "HK"
    return out


def get_sti_universe() -> pd.DataFrame:
    data = [
        ("D05.SI", "DBS Group Holdings", "Financials"),
        ("U11.SI", "United Overseas Bank", "Financials"),
        ("O39.SI", "Oversea-Chinese Banking Corporation", "Financials"),
        ("C07.SI", "Jardine Matheson", "Conglomerate"),
        ("C09.SI", "City Developments", "Real Estate"),
        ("C38U.SI", "CapitaLand Integrated Commercial Trust", "Real Estate"),
        ("C52.SI", "ComfortDelGro", "Transportation"),
        ("F34.SI", "Frasers Logistics & Commercial Trust", "Real Estate"),
        ("G13.SI", "Genting Singapore", "Entertainment"),
        ("H78.SI", "Hongkong Land", "Real Estate"),
        ("J36.SI", "Jardine Cycle & Carriage", "Industrial"),
        ("M44U.SI", "Mapletree Logistics Trust", "Real Estate"),
        ("ME8U.SI", "Mapletree Industrial Trust", "Real Estate"),
        ("N2IU.SI", "NetLink NBN Trust", "Utilities"),
        ("S63.SI", "Singapore Airlines", "Transportation"),
        ("S68.SI", "Singapore Exchange", "Financials"),
        ("S58.SI", "Sembcorp Industries", "Utilities"),
        ("U96.SI", "SATS Ltd", "Services"),
        ("S07.SI", "Singapore Technologies Engineering", "Industrial"),
        ("Z74.SI", "Singtel", "Telecom"),
        ("BN4.SI", "Keppel Corporation", "Industrial"),
        ("M01.SI", "Micro-Mechanics", "Industrial"),
        ("A17U.SI", "CapitaLand Ascendas REIT", "Real Estate"),
        ("BS6.SI", "Yangzijiang Shipbuilding", "Industrial"),
        ("C31.SI", "CapitaLand Investment", "Real Estate"),
        ("E5H.SI", "Emperador Inc", "Consumer Staples"),
        ("5DP.SI", "Delfi Limited", "Consumer Goods"),
        ("D01.SI", "Dairy Farm International", "Consumer Staples"),
        ("K71U.SI", "Keppel DC REIT", "Real Estate"),
    ]
    out = pd.DataFrame(data, columns=["Ticker", "Name", "Sector"])
    out["Country"] = "SG"
    return out


# =========================
# Yahoo data loader (generic)
# =========================
def get_data_from_yahoo(
    tickers: List[str],
    country: str,
    start: str = "2016-01-01",
    end: Optional[str] = None,
    sleep_s: float = 0.15,
    max_retries: int = 2,
    chunk_size: int = 80,
    threads: int = 8,
) -> pd.DataFrame:
    """
    Downloads OHLCV from Yahoo and returns:
      date, ticker, open, high, low, close, volume, turnover, market_cap, country

    - auto_adjust=False (raw close)
    - market_cap is best-effort using sharesOutstanding
    """
    if end is None:
        end = datetime.today().strftime("%Y-%m-%d")

    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    # 1) sharesOutstanding (best-effort)
    def _shares_one(tkr: str) -> Tuple[str, Optional[float]]:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                yt = yf.Ticker(tkr)
                so = None

                fi = getattr(yt, "fast_info", None)
                if fi is not None and hasattr(fi, "get"):
                    try:
                        so = fi.get("shares", None)
                    except Exception:
                        so = None

                if so is None:
                    try:
                        info = yt.get_info()
                        so = info.get("sharesOutstanding", None)
                    except Exception:
                        so = None

                so = float(so) if so is not None and np.isfinite(float(so)) and float(so) > 0 else None
                return tkr, so
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, sleep_s * (attempt + 1)))
        print(f"[WARN] sharesOutstanding failed for {tkr}: {last_err}")
        return tkr, None

    shares_map: Dict[str, Optional[float]] = {}
    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futs = {ex.submit(_shares_one, t): t for t in tickers}
        for fut in as_completed(futs):
            tkr, so = fut.result()
            shares_map[tkr] = so

    # 2) OHLCV (chunked)
    frames: List[pd.DataFrame] = []
    failed: List[str] = []

    for i in range(0, len(tickers), chunk_size):
        batch = tickers[i : i + chunk_size]

        data = None
        last_err = None
        ok = False
        for attempt in range(max_retries + 1):
            try:
                data = yf.download(
                    batch,
                    start=start,
                    end=end,
                    group_by="column",
                    auto_adjust=False,
                    actions=False,
                    threads=True,
                    progress=False,
                )
                ok = True
                break
            except Exception as e:
                last_err = e
                time.sleep(min(2.0, sleep_s * (attempt + 1)))

        if not ok or data is None or len(data) == 0:
            print(f"[WARN] price download failed for batch {batch[:3]}...: {last_err}")
            failed.extend(batch)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            # Fast path: stack once instead of per-ticker extraction loops
            try:
                stacked = data.stack(level=1, future_stack=True).reset_index()
            except TypeError:
                stacked = data.stack(level=1).reset_index()

            if stacked.shape[1] < 3:
                failed.extend(batch)
                continue

            date_col = stacked.columns[0]
            tkr_col  = stacked.columns[1]
            stacked = stacked.rename(columns={date_col: "date", tkr_col: "ticker"})

            col_map = {"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
            stacked = stacked.rename(columns=col_map)

            need = ["date", "ticker", "open", "high", "low", "close"]
            missing = [c for c in need if c not in stacked.columns]
            if missing:
                failed.extend(batch)
                continue

            stacked["ticker"] = stacked["ticker"].astype(str)
            stacked = stacked.dropna(subset=need)
            stacked = stacked[stacked["ticker"].isin(batch)]

            if stacked.empty:
                failed.extend(batch)
            else:
                if "volume" not in stacked.columns:
                    stacked["volume"] = np.nan
                frames.append(stacked[["date", "ticker", "open", "high", "low", "close", "volume"]].copy())
        else:
            t = batch[0]
            try:
                sub = data.reset_index().rename(
                    columns={"Date": "date", "Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"}
                ).dropna(subset=["open", "high", "low", "close"])
                if sub.empty:
                    failed.append(t)
                else:
                    sub["ticker"] = t
                    frames.append(sub)
            except Exception:
                failed.append(t)

        time.sleep(sleep_s)

    if not frames:
        raise RuntimeError("No price data downloaded from Yahoo (all batches failed/empty).")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    df["volume"] = pd.to_numeric(df["volume"], errors="coerce").fillna(0.0)
    df = df[df["volume"] >= 0]

    df["turnover"] = df["close"].astype(float) * df["volume"].astype(float)
    df["shares_outstanding"] = df["ticker"].map(shares_map).astype(float)
    df["market_cap"] = df["shares_outstanding"].astype(float) * df["close"].astype(float)
    df["country"] = country

    df = (
        df.drop_duplicates(subset=["date", "ticker"])
        .sort_values(["date", "ticker"])
        .reset_index(drop=True)
    )

    if failed:
        print(f"[WARN] Failed tickers: {len(set(failed))} -> {sorted(set(failed))[:20]}{' ...' if len(set(failed))>20 else ''}")

    return df[["date", "ticker", "open", "high", "low", "close", "volume", "turnover", "market_cap", "country"]]


# =========================
# Helpers: Indicators
# =========================
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

    # --- SMAs
    close = df["close"].astype(float)
    df["sma20"] = g["close"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)
    df["sma50"] = g["close"].rolling(50, min_periods=50).mean().reset_index(level=0, drop=True)
    df["sma20_slope5"] = df["sma20"] - g["sma20"].shift(5)

    # --- Prev close
    df["prev_close"] = g["close"].shift(1)

    # --- True Range (vectorized)
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    prev_close = df["prev_close"].astype(float)

    tr1 = (high - low).to_numpy()
    tr2 = (high - prev_close).abs().to_numpy()
    tr3 = (low - prev_close).abs().to_numpy()
    df["tr"] = np.maximum(tr1, np.maximum(tr2, tr3))

    # --- ATRs
    df["atr5"] = g["tr"].rolling(5, min_periods=5).mean().reset_index(level=0, drop=True)
    df["atr20"] = g["tr"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    # --- Structure
    df["hh5"] = g["high"].rolling(5, min_periods=5).max().reset_index(level=0, drop=True)
    df["ll5"] = g["low"].rolling(5, min_periods=5).min().reset_index(level=0, drop=True)
    df["range5_pct"] = (df["hh5"] - df["ll5"]) / close

    df["hh20"] = g["high"].rolling(20, min_periods=20).max().reset_index(level=0, drop=True)
    df["ll20"] = g["low"].rolling(20, min_periods=20).min().reset_index(level=0, drop=True)
    df["range20_pct"] = (df["hh20"] - df["ll20"]) / close
    denom20 = (df["hh20"] - df["ll20"]).replace(0, np.nan)
    df["close_pos_20"] = (close - df["ll20"]) / denom20

    # --- Turnover
    df["turnover_5d"] = g["turnover"].rolling(5, min_periods=5).sum().reset_index(level=0, drop=True)
    df["adv_turnover_20"] = g["turnover"].rolling(20, min_periods=20).mean().reset_index(level=0, drop=True)

    # --- Ranking baseline
    df["avg_turnover_10d"] = (
        g["turnover"]
        .rolling(cfg.turnover_baseline_days, min_periods=cfg.turnover_baseline_days)
        .mean()
        .reset_index(level=0, drop=True)
    )
    df["baseline_5d_turnover"] = df["avg_turnover_10d"] * 5.0

    # --- Trend pause reference
    df["hh10_close"] = g["close"].rolling(10, min_periods=10).max().reset_index(level=0, drop=True)

    # --- Fib: HH(high) + LOW-SINCE-HH
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


# =========================
# Setup detection
# =========================
def _universe_filter(df: pd.DataFrame, cfg: BacktestConfig) -> pd.DataFrame:
    """
    Country-aware universe filter (vectorized).
    """
    needed = [
        "sma20", "sma50", "sma20_slope5",
        "atr5", "atr20",
        "hh5", "ll5",
        "range5_pct", "range20_pct", "close_pos_20",
        "turnover_5d", "baseline_5d_turnover",
        "adv_turnover_20",
        "hh10_close", "hh_fib", "low_since_hh_fib",
        "market_cap", "country",
    ]

    d = df.dropna(subset=needed).copy()
    if d.empty:
        return d

    c = d["country"].astype(str)

    mcap_min = c.map(lambda x: UNIVERSE_RULES.get(x, {}).get("mcap_min", np.nan)).astype(float)
    mcap_max = c.map(lambda x: UNIVERSE_RULES.get(x, {}).get("mcap_max", np.nan)).astype(float)
    adv_min  = c.map(lambda x: UNIVERSE_RULES.get(x, {}).get("adv_turnover_20_min", np.nan)).astype(float)

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

    # --- A) VOL_COMPRESSION_BREAKOUT
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

    # --- C base
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

    # C1 breakout
    C1 = PB.copy()
    if not C1.empty:
        vol_confirm_C = C1["turnover_expansion"] >= cfg.turnover_expansion_min_C
        C1 = C1[vol_confirm_C].copy()
        C1["setup_tag"] = "TREND_PAUSE_BREAKOUT"
        C1["expected_entry_type"] = "BREAKOUT"
        C1["breakout_level"] = C1["hh5"]
        C1["pullback_level"] = np.nan
        C1["stop_level"] = C1["ll5"]

    # C2 pullback
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
        "signal_date", "ticker", "country", "setup_tag", "expected_entry_type",
        "score", "breakout_level", "pullback_level", "stop_level",
        "close", "sma20", "sma50", "dist_20dma",
        "range5_pct", "range20_pct", "close_pos_20",
        "atr5", "atr20",
        "turnover_expansion", "adv_turnover_20", "market_cap",
        "turnover_5d", "baseline_5d_turnover", "avg_turnover_10d",
    ]
    keep_cols = [c for c in keep_cols if c in out.columns]
    return (
        out[keep_cols]
        .sort_values(["signal_date", "score"], ascending=[True, False])
        .reset_index(drop=True)
    )


# =========================
# Trade simulation
# =========================
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


def run_backtest(df_raw: pd.DataFrame, cfg: Optional[BacktestConfig] = None, indicators_precomputed: bool = False):
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

        # 1) Exits
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

        # 2) Entries: execute prior day signals
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

    return trades_df, entries_df, summary_overall, summary_by, exit_breakdown, summary_by_year, signals


# =========================
# Reporting helpers
# =========================
def _streak_stats(pnl: pd.Series) -> Dict[str, float]:
    pnl = pnl.dropna().astype(float)
    if pnl.empty:
        return {
            "max_winning_streak": np.nan,
            "max_losing_streak": np.nan,
            "avg_winning_streak": np.nan,
            "avg_losing_streak": np.nan,
        }

    is_win = (pnl > 0).to_numpy(dtype=bool)
    runs: List[Tuple[bool, int]] = []
    cur = is_win[0]
    length = 1
    for v in is_win[1:]:
        if v == cur:
            length += 1
        else:
            runs.append((cur, length))
            cur = v
            length = 1
    runs.append((cur, length))

    win_runs = [l for w, l in runs if w]
    loss_runs = [l for w, l in runs if not w]

    return {
        "max_winning_streak": float(max(win_runs)) if win_runs else 0.0,
        "max_losing_streak": float(max(loss_runs)) if loss_runs else 0.0,
        "avg_winning_streak": float(np.mean(win_runs)) if win_runs else 0.0,
        "avg_losing_streak": float(np.mean(loss_runs)) if loss_runs else 0.0,
    }


def _profit_factor(pnl: pd.Series) -> float:
    pnl = pnl.dropna().astype(float)
    if pnl.empty:
        return np.nan
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    if losses == 0:
        return np.nan
    return float(gains / losses)


def summarize_trades(trades_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "trades",
        "win_rate_%",
        "avg_win_%",
        "avg_loss_%",
        "payoff_ratio",
        "expectancy_%",
        "profit_factor",
        "median_pnl_%",
        "p25_pnl_%",
        "p75_pnl_%",
        "worst_trade_%",
        "best_trade_%",
        "max_winning_streak",
        "max_losing_streak",
        "avg_winning_streak",
        "avg_losing_streak",
        "target_hit_rate_%",
    ]

    if trades_df is None or trades_df.empty:
        return pd.DataFrame([{c: (0 if c == "trades" else np.nan) for c in cols}])

    pnl = trades_df["pnl_pct_net"].astype(float)
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]

    win_rate = (pnl > 0).mean() * 100.0
    avg_win = wins.mean() if len(wins) else np.nan
    avg_loss = losses.mean() if len(losses) else np.nan
    expectancy = pnl.mean()
    median = pnl.median()
    p25 = pnl.quantile(0.25)
    p75 = pnl.quantile(0.75)
    worst_trade = pnl.min()
    best_trade = pnl.max()

    payoff_ratio = (avg_win / abs(avg_loss)) if (np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0) else np.nan

    profit_factor = _profit_factor(pnl)

    streaks = _streak_stats(pnl)

    target_hit_rate = (trades_df["exit_reason"] == "TARGET_10PCT").mean() * 100.0

    row = {
        "trades": len(trades_df),
        "win_rate_%": win_rate,
        "avg_win_%": avg_win,
        "avg_loss_%": avg_loss,
        "payoff_ratio": payoff_ratio,
        "expectancy_%": expectancy,
        "profit_factor": profit_factor,
        "median_pnl_%": median,
        "p25_pnl_%": p25,
        "p75_pnl_%": p75,
        "worst_trade_%": worst_trade,
        "best_trade_%": best_trade,
        **streaks,
        "target_hit_rate_%": target_hit_rate,
    }
    return pd.DataFrame([row])[cols]


def summarize_trades_by_setup_entry(trades_df: pd.DataFrame) -> pd.DataFrame:
    cols = [
        "country", "setup_tag", "entry_type",
        "trades",
        "win_rate_%",
        "avg_win_%",
        "avg_loss_%",
        "payoff_ratio",
        "expectancy_%",
        "profit_factor",
        "p25_pnl_%",
        "median_pnl_%",
        "p75_pnl_%",
        "worst_trade_%",
        "best_trade_%",
        "target_hit_rate_%",
    ]

    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=cols)

    group_cols = ["country", "setup_tag", "entry_type"]
    out_rows = []

    for keys, g in trades_df.groupby(group_cols, dropna=False):
        country, setup_tag, entry_type = keys

        pnl = g["pnl_pct_net"].astype(float)
        wins = pnl[pnl > 0]
        losses = pnl[pnl <= 0]

        avg_win = wins.mean() if len(wins) else np.nan
        avg_loss = losses.mean() if len(losses) else np.nan
        payoff_ratio = (avg_win / abs(avg_loss)) if (np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0) else np.nan

        profit_factor = _profit_factor(pnl)

        out_rows.append(
            {
                "country": country,
                "setup_tag": setup_tag,
                "entry_type": entry_type,
                "trades": len(g),
                "win_rate_%": (pnl > 0).mean() * 100.0,
                "avg_win_%": avg_win,
                "avg_loss_%": avg_loss,
                "payoff_ratio": payoff_ratio,
                "expectancy_%": pnl.mean(),
                "profit_factor": profit_factor,
                "p25_pnl_%": pnl.quantile(0.25),
                "median_pnl_%": pnl.median(),
                "p75_pnl_%": pnl.quantile(0.75),
                "worst_trade_%": pnl.min(),
                "best_trade_%": pnl.max(),
                "target_hit_rate_%": (g["exit_reason"] == "TARGET_10PCT").mean() * 100.0,
            }
        )

    return (
        pd.DataFrame(out_rows)[cols]
        .sort_values(["country", "expectancy_%", "trades"], ascending=[True, False, False])
        .reset_index(drop=True)
    )


def summarize_exit_reasons(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["exit_reason", "pct_%", "count"])

    counts = trades_df["exit_reason"].astype(str).value_counts(dropna=False)
    total = float(counts.sum()) if counts.sum() else 1.0
    out = pd.DataFrame({
        "exit_reason": counts.index.astype(str),
        "pct_%": (counts.values / total) * 100.0,
        "count": counts.values.astype(int),
    })
    out = out.sort_values(["pct_%", "exit_reason"], ascending=[False, True]).reset_index(drop=True)
    return out


def summarize_trades_by_year(trades_df: pd.DataFrame) -> pd.DataFrame:
    if trades_df is None or trades_df.empty:
        return pd.DataFrame(columns=["year", "trades", "expectancy_%", "profit_factor", "win_rate_%"])

    d = trades_df.copy()
    d["exit_date"] = pd.to_datetime(d["exit_date"])
    d["year"] = d["exit_date"].dt.year.astype(int)

    rows = []
    for year, g in d.groupby("year"):
        pnl = g["pnl_pct_net"].astype(float)
        rows.append({
            "year": int(year),
            "trades": int(len(g)),
            "expectancy_%": float(pnl.mean()) if len(pnl) else np.nan,
            "profit_factor": _profit_factor(pnl),
            "win_rate_%": float((pnl > 0).mean() * 100.0) if len(pnl) else np.nan,
        })

    out = pd.DataFrame(rows).sort_values("year").reset_index(drop=True)
    return out


def summarize_trades_by_score_bins(trades_enriched: pd.DataFrame) -> pd.DataFrame:
    cols = ["score_bin", "trades", "expectancy_%", "profit_factor", "win_rate_%"]
    if trades_enriched is None or trades_enriched.empty or "score" not in trades_enriched.columns:
        return pd.DataFrame(columns=cols)

    t = trades_enriched.copy()
    t = t.dropna(subset=["score"]).copy()
    if t.empty:
        return pd.DataFrame(columns=cols)

    bins = [-1e-9, 0.2, 0.4, 0.6, 0.8, 1.000000001]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    t["score_bin"] = pd.cut(t["score"].astype(float), bins=bins, labels=labels, include_lowest=True)

    rows = []
    for b, g in t.groupby("score_bin", dropna=False):
        pnl = g["pnl_pct_net"].astype(float)
        rows.append({
            "score_bin": str(b),
            "trades": int(len(g)),
            "expectancy_%": float(pnl.mean()) if len(pnl) else np.nan,
            "profit_factor": _profit_factor(pnl),
            "win_rate_%": float((pnl > 0).mean() * 100.0) if len(pnl) else np.nan,
        })

    return pd.DataFrame(rows)[cols].sort_values("score_bin").reset_index(drop=True)


def summarize_trades_by_mcap_bins(trades_enriched: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    cols = ["mcap_bucket", "trades", "expectancy_%", "profit_factor", "win_rate_%"]
    if trades_enriched is None or trades_enriched.empty or "market_cap" not in trades_enriched.columns:
        return pd.DataFrame(columns=cols)

    t = trades_enriched.copy()
    t = t.dropna(subset=["market_cap"]).copy()
    if t.empty:
        return pd.DataFrame(columns=cols)

    try:
        t["mcap_bucket"] = pd.qcut(t["market_cap"].astype(float), q=q, duplicates="drop")
    except Exception:
        # fallback: if qcut fails, just return empty (keeps run safe)
        return pd.DataFrame(columns=cols)

    rows = []
    for b, g in t.groupby("mcap_bucket", dropna=False):
        pnl = g["pnl_pct_net"].astype(float)
        rows.append({
            "mcap_bucket": str(b),
            "trades": int(len(g)),
            "expectancy_%": float(pnl.mean()) if len(pnl) else np.nan,
            "profit_factor": _profit_factor(pnl),
            "win_rate_%": float((pnl > 0).mean() * 100.0) if len(pnl) else np.nan,
        })

    return pd.DataFrame(rows)[cols].sort_values("mcap_bucket").reset_index(drop=True)


def show_grouped_signals_last_n_days(
    signals: pd.DataFrame,
    n_days: int = 5,
    title: str = "CURRENT SIGNALS",
) -> pd.DataFrame:
    """
    More concise "current signals" output:
      ticker, country, latest_signal_date, setups, score_max, breakout, stop, close, mcap
    """
    if signals is None or signals.empty:
        print(f"\n[{title}] No signals.")
        return pd.DataFrame()

    s = signals.copy()
    s["signal_date"] = pd.to_datetime(s["signal_date"]).dt.normalize()

    dates = sorted(s["signal_date"].unique())
    last_dates = dates[-n_days:] if len(dates) >= n_days else dates

    window = s[s["signal_date"].isin(last_dates)].copy()
    if window.empty:
        print(f"\n[{title}] No signals in window.")
        return pd.DataFrame()

    def _join_unique(vals) -> str:
        vals = [str(v) for v in pd.Series(list(vals)).dropna().astype(str).unique().tolist()]
        return "; ".join(sorted(vals))

    window = window.sort_values(["ticker", "signal_date", "score"], ascending=[True, True, False])
    latest = window.groupby("ticker", as_index=False, observed=True).tail(1).copy()

    score_max = window.groupby("ticker", observed=True)["score"].max().rename("score_max")
    setups = window.groupby("ticker", observed=True)["setup_tag"].agg(_join_unique).rename("setups")

    out = (
        latest.set_index("ticker")
        .join([score_max, setups])
        .reset_index()
    )

    out = out.rename(columns={
        "signal_date": "latest_signal_date",
        "breakout_level": "breakout_level_latest",
        "pullback_level": "pullback_level_latest",
        "stop_level": "stop_level_latest",
        "close": "close_latest",
        "market_cap": "market_cap_latest",
    })

    cols = [
        "ticker",
        "country",
        "latest_signal_date",
        "setups",
        "score_max",
        "breakout_level_latest",
        "stop_level_latest",
        "close_latest",
        "market_cap_latest",
    ]
    cols = [c for c in cols if c in out.columns]

    out = out.sort_values(["score_max", "ticker"], ascending=[False, True]).reset_index(drop=True)

    print(
        f"\n=== {title} (last {len(last_dates)} trading days: "
        f"{last_dates[0].date()} -> {last_dates[-1].date()}) ==="
    )
    print(out[cols].to_string(index=False, float_format=lambda x: f"{x:,.4f}"))
    return out[cols].reset_index(drop=True)


# =========================
# 3 separate runs
# =========================
def run_country_backtest(
    country: str,
    universe: pd.DataFrame,
    cfg: BacktestConfig,
    start: str = "2016-01-01",
    end: Optional[str] = None,
    n_signal_days: int = 5,
):
    tickers = universe["Ticker"].dropna().astype(str).unique().tolist()
    print(f"\n[{country}] Universe tickers: {len(tickers)}")

    df = load_or_build_cache(country, universe, cfg, start=start, end=end)
    trades, entries, summary_overall, summary_by, exit_breakdown, summary_by_year, signals = run_backtest(
        df, cfg, indicators_precomputed=True
    )

    # --- Enrich for score/mcap bins (exactly as golden source)
    trades_enriched = trades
    if (trades is not None) and (not trades.empty) and (signals is not None) and (not signals.empty):
        sig_small = signals[["signal_date", "ticker", "setup_tag", "score", "market_cap"]].copy()
        sig_small["signal_date"] = pd.to_datetime(sig_small["signal_date"]).dt.normalize()
        trades_enriched = trades.merge(sig_small, on=["signal_date", "ticker", "setup_tag"], how="left")

    score_stats = summarize_trades_by_score_bins(trades_enriched)
    mcap_stats = summarize_trades_by_mcap_bins(trades_enriched, q=5)

    current_signals = show_grouped_signals_last_n_days(
        signals,
        n_days=n_signal_days,
        title=f"{country} CURRENT SIGNALS",
    )

    return {
        "country": country,
        "universe": universe,
        "df": df,
        "trades": trades,
        "entries": entries,
        "summary_overall": summary_overall,
        "summary_by": summary_by,
        "exit_breakdown": exit_breakdown,
        "summary_by_year": summary_by_year,
        "signals": signals,
        "score_stats": score_stats,
        "mcap_stats": mcap_stats,
        "current_signals": current_signals,
    }


def default_country_configs() -> Dict[str, BacktestConfig]:
    # --- EXACT defaults from your __main__
    cfg_hk = BacktestConfig(
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
    )

    cfg_us = BacktestConfig(
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
    )

    cfg_sg = BacktestConfig(
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
    )

    return {"HK": cfg_hk, "US": cfg_us, "SG": cfg_sg}
