from __future__ import annotations

from io import StringIO
from typing import Optional
import time

import pandas as pd
import requests


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
        except Exception as e:  # noqa: BLE001 - surfacing last error later
            last_err = e
            time.sleep(backoff_s * (attempt + 1))
    raise RuntimeError(f"GET failed after {max_retries} retries: {url} | last_err={last_err}")


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


def get_universe(country: str) -> pd.DataFrame:
    country = country.upper()
    if country == "HK":
        return get_hsi_universe()
    if country == "US":
        return get_sp500_universe()
    if country == "SG":
        return get_sti_universe()
    raise ValueError(f"Unsupported country: {country}")
