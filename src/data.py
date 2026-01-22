from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import os
import time

import numpy as np
import pandas as pd
import yfinance as yf

from src.config import BacktestConfig
from src.indicators import add_indicators
from src.reporting import expected_last_daily_bar_date

CACHE_DIR = "cache_daily"


def cache_path(country: str) -> str:
    return os.path.join(CACHE_DIR, f"{country}_daily.parquet")


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
    cfg: BacktestConfig,
    full_start: str = "2016-01-01",
    end: Optional[str] = None,
    overlap_calendar_days: int = 200,
) -> pd.DataFrame:
    if end is None:
        end = (datetime.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    os.makedirs(CACHE_DIR, exist_ok=True)
    path = cache_path(country)
    tickers = universe["Ticker"].dropna().astype(str).unique().tolist()

    if not os.path.exists(path):
        print(f"[{country}] Cache missing. Building full -> {path}")
        df_raw = get_data_from_yahoo(
            tickers=tickers,
            country=country,
            start=full_start,
            end=end,
            sleep_s=0.15,
            max_retries=2,
            chunk_size=250 if country == "US" else 20,
            threads=6 if country == "US" else 4,
        )
        df_ind = add_indicators(df_raw, cfg)
        df_ind.to_parquet(path, index=False)
        return _ensure_datetime(df_ind)

    print(f"[{country}] Loading cache -> {path}")
    try:
        df_cached = _ensure_datetime(pd.read_parquet(path))
    except Exception as exc:
        print(f"[{country}] Cache read failed; rebuilding ({exc}).")
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        return build_or_update_daily_cache(country, universe, cfg, full_start=full_start, end=end)

    if df_cached.empty:
        os.remove(path)
        return build_or_update_daily_cache(country, universe, cfg, full_start=full_start, end=end)

    last_date = pd.to_datetime(df_cached["date"]).max().normalize()
    expected_last = expected_last_daily_bar_date(country).normalize()

    if last_date >= expected_last:
        print(f"[{country}] Cache is fresh (last cached: {last_date.date()}). Skipping refresh.")
        return df_cached

    inc_start = _safe_start_for_incremental(last_date, overlap_days=overlap_calendar_days)
    print(f"[{country}] Incremental refresh: {inc_start} -> {end} (last cached: {last_date.date()})")

    try:
        df_new = get_data_from_yahoo(
            tickers=tickers,
            country=country,
            start=inc_start,
            end=end,
            sleep_s=0.2,
            max_retries=3,
            chunk_size=200 if country == "US" else 20,
            threads=4 if country != "US" else 6,
        )

        if df_new is None or df_new.empty:
            print(f"[{country}] Incremental refresh returned empty; keeping cache.")
            return df_cached
    except RuntimeError as e:
        print(f"[{country}] Incremental refresh error (keeping cache): {e}")
        return df_cached

    df_new = _ensure_datetime(df_new)

    base_cols = [
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "turnover",
        "market_cap",
        "country",
    ]

    df_merged = pd.concat([df_cached[base_cols], df_new[base_cols]], ignore_index=True)
    df_merged = df_merged.drop_duplicates(subset=["date", "ticker"], keep="last").sort_values(
        ["ticker", "date"]
    )
    df_merged = df_merged.reset_index(drop=True)

    cutoff = pd.to_datetime(inc_start)
    df_old_part = df_cached[df_cached["date"] < cutoff].copy()

    df_tail_raw = df_merged[df_merged["date"] >= cutoff].copy()
    df_tail_ind = add_indicators(df_tail_raw, cfg)

    df_final = pd.concat([df_old_part, df_tail_ind], ignore_index=True)
    df_final = df_final.drop_duplicates(subset=["date", "ticker"], keep="last")
    df_final = df_final.sort_values(["date", "ticker"]).reset_index(drop=True)

    df_final.to_parquet(path, index=False)
    return df_final


def load_or_build_cache(
    country: str,
    universe: pd.DataFrame,
    cfg: BacktestConfig,
    start: str = "2016-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    return build_or_update_daily_cache(country, universe, cfg, full_start=start, end=end)


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
    if end is None:
        end = (datetime.today() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

    tickers = [str(t).strip() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    def _fundamentals_one(tkr: str) -> Tuple[str, Optional[float], Optional[float]]:
        last_err = None
        for attempt in range(max_retries + 1):
            try:
                yt = yf.Ticker(tkr)

                shares = None
                mcap_now = None

                fi = getattr(yt, "fast_info", None)
                if fi is not None and hasattr(fi, "get"):
                    try:
                        shares = fi.get("shares", None) or fi.get("shares_outstanding", None)
                        mcap_now = fi.get("market_cap", None) or fi.get("marketCap", None)
                    except Exception:
                        pass

                if shares is None or mcap_now is None:
                    try:
                        info = yt.get_info()
                        if shares is None:
                            shares = info.get("sharesOutstanding", None)
                        if mcap_now is None:
                            mcap_now = info.get("marketCap", None)
                    except Exception:
                        pass

                def _clean(x):
                    try:
                        x = float(x)
                        return x if np.isfinite(x) and x > 0 else None
                    except Exception:
                        return None

                shares = _clean(shares)
                mcap_now = _clean(mcap_now)

                return tkr, shares, mcap_now
            except Exception as e:  # noqa: BLE001 - surfaced after retries
                last_err = e
                time.sleep(min(2.0, sleep_s * (attempt + 1)))

        print(f"[WARN] fundamentals failed for {tkr}: {last_err}")
        return tkr, None, None

    shares_map: Dict[str, Optional[float]] = {}
    mcap_now_map: Dict[str, Optional[float]] = {}

    with ThreadPoolExecutor(max_workers=max(1, int(threads))) as ex:
        futs = {ex.submit(_fundamentals_one, t): t for t in tickers}
        for fut in as_completed(futs):
            tkr, so, mc = fut.result()
            shares_map[tkr] = so
            mcap_now_map[tkr] = mc

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
            except Exception as e:  # noqa: BLE001 - surfaced after retries
                last_err = e
                time.sleep(min(2.0, sleep_s * (attempt + 1)))

        if not ok or data is None or len(data) == 0:
            print(f"[WARN] price download failed for batch {batch[:3]}...: {last_err}")
            failed.extend(batch)
            continue

        if isinstance(data.columns, pd.MultiIndex):
            try:
                stacked = data.stack(level=1, future_stack=True).reset_index()
            except TypeError:
                stacked = data.stack(level=1).reset_index()

            if stacked.shape[1] < 3:
                failed.extend(batch)
                continue

            date_col = stacked.columns[0]
            tkr_col = stacked.columns[1]
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
                sub = (
                    data.reset_index()
                    .rename(
                        columns={
                            "Date": "date",
                            "Open": "open",
                            "High": "high",
                            "Low": "low",
                            "Close": "close",
                            "Volume": "volume",
                        }
                    )
                    .dropna(subset=["open", "high", "low", "close"])
                )
                if sub.empty:
                    failed.append(t)
                else:
                    sub["ticker"] = t
                    frames.append(sub[["date", "ticker", "open", "high", "low", "close", "volume"]].copy())
            except Exception:
                failed.append(t)

        time.sleep(sleep_s)

    if not frames:
        raise RuntimeError("No price data downloaded from Yahoo (all batches failed/empty).")

    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)

    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["volume"] = df["volume"].fillna(0.0)
    df = df.dropna(subset=["open", "high", "low", "close"])
    df = df[df["volume"] >= 0]

    df["turnover"] = df["close"].astype(float) * df["volume"].astype(float)

    df["shares_outstanding"] = pd.to_numeric(df["ticker"].map(shares_map), errors="coerce")

    market_cap_hist = df["shares_outstanding"] * df["close"].astype(float)
    market_cap_now = pd.to_numeric(df["ticker"].map(mcap_now_map), errors="coerce")

    df["market_cap"] = market_cap_hist.fillna(market_cap_now)

    df["country"] = country

    df = df.drop_duplicates(subset=["date", "ticker"], keep="last")
    df = df.sort_values(["date", "ticker"]).reset_index(drop=True)

    if failed:
        print(
            f"[WARN] Failed tickers: {len(set(failed))} -> "
            f"{sorted(set(failed))[:20]}{' ...' if len(set(failed))>20 else ''}"
        )

    return df[["date", "ticker", "open", "high", "low", "close", "volume", "turnover", "market_cap", "country"]]
