from __future__ import annotations

from datetime import datetime, time as dtime
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from src.config import MARKET_BAR_READY_TIME, MARKET_TZ


def expected_last_daily_bar_date(country: str) -> pd.Timestamp:
    tz = ZoneInfo(MARKET_TZ.get(country, "UTC"))
    now_local = datetime.now(tz)
    ready_tuple = MARKET_BAR_READY_TIME.get(country, (18, 0))
    ready_t = dtime(ready_tuple[0], ready_tuple[1])

    today_local = pd.Timestamp(now_local.date())
    wd = now_local.weekday()

    if wd == 5:
        return today_local - pd.Timedelta(days=1)
    if wd == 6:
        return today_local - pd.Timedelta(days=2)

    if now_local.time() < ready_t:
        if wd == 0:
            return today_local - pd.Timedelta(days=3)
        return today_local - pd.Timedelta(days=1)

    return today_local


def _fmt_num(x: float, decimals: int = 2) -> str:
    if x is None:
        return "NA"
    try:
        x = float(x)
    except Exception:
        return "NA"
    if not np.isfinite(x):
        return "NA"
    return f"{x:,.{decimals}f}"


def _fmt_interval(iv, decimals: int = 2) -> str:
    if iv is None or str(iv) == "nan":
        return "NA"
    try:
        left = _fmt_num(iv.left, decimals)
        right = _fmt_num(iv.right, decimals)
        lbr = "(" if not iv.closed_left else "["
        rbr = "]" if iv.closed_right else ")"
        return f"{lbr}{left} .. {right}{rbr}"
    except Exception:
        return str(iv)


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

    payoff_ratio = (
        (avg_win / abs(avg_loss))
        if (np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0)
        else np.nan
    )

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
        "country",
        "setup_tag",
        "entry_type",
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
        payoff_ratio = (
            (avg_win / abs(avg_loss))
            if (np.isfinite(avg_win) and np.isfinite(avg_loss) and avg_loss != 0)
            else np.nan
        )

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
    out = pd.DataFrame(
        {
            "exit_reason": counts.index.astype(str),
            "pct_%": (counts.values / total) * 100.0,
            "count": counts.values.astype(int),
        }
    )
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
        rows.append(
            {
                "year": int(year),
                "trades": int(len(g)),
                "expectancy_%": float(pnl.mean()) if len(pnl) else np.nan,
                "profit_factor": _profit_factor(pnl),
                "win_rate_%": float((pnl > 0).mean() * 100.0) if len(pnl) else np.nan,
            }
        )

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
    for b, g in t.groupby("score_bin", dropna=False, observed=True):
        pnl = g["pnl_pct_net"].astype(float)
        rows.append(
            {
                "score_bin": str(b),
                "trades": int(len(g)),
                "expectancy_%": float(pnl.mean()) if len(pnl) else np.nan,
                "profit_factor": _profit_factor(pnl),
                "win_rate_%": float((pnl > 0).mean() * 100.0) if len(pnl) else np.nan,
            }
        )

    return pd.DataFrame(rows)[cols].sort_values("score_bin").reset_index(drop=True)


def summarize_trades_by_mcap_bins(trades_enriched: pd.DataFrame, q: int = 5) -> pd.DataFrame:
    cols = ["mcap_bucket", "trades", "expectancy_%", "profit_factor", "win_rate_%"]
    if trades_enriched is None or trades_enriched.empty or "market_cap" not in trades_enriched.columns:
        return pd.DataFrame(columns=cols)

    t = trades_enriched.dropna(subset=["market_cap", "pnl_pct_net"]).copy()
    if t.empty:
        return pd.DataFrame(columns=cols)

    try:
        t["mcap_bucket"] = pd.qcut(t["market_cap"].astype(float), q=q, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=cols)

    rows = []
    for b, g in t.groupby("mcap_bucket", dropna=False, observed=True):
        pnl = g["pnl_pct_net"].astype(float)
        rows.append(
            {
                "mcap_bucket": _fmt_interval(b, 2),
                "trades": int(len(g)),
                "expectancy_%": float(pnl.mean()) if len(pnl) else np.nan,
                "profit_factor": _profit_factor(pnl),
                "win_rate_%": float((pnl > 0).mean() * 100.0) if len(pnl) else np.nan,
            }
        )

    out = pd.DataFrame(rows)[cols]
    return out.sort_values("mcap_bucket").reset_index(drop=True)


def summarize_heatmap_score_x_mcap(
    trades_enriched: pd.DataFrame,
    mcap_q: int = 5,
    value: str = "win_rate_%",
) -> pd.DataFrame:
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    cols = ["mcap_bucket"] + labels
    if trades_enriched is None or trades_enriched.empty:
        return pd.DataFrame(columns=cols)

    t = trades_enriched.dropna(subset=["score", "market_cap", "pnl_pct_net"]).copy()
    if t.empty:
        return pd.DataFrame(columns=cols)

    bins = [-1e-9, 0.2, 0.4, 0.6, 0.8, 1.000000001]
    t["score_bin"] = pd.cut(t["score"].astype(float), bins=bins, labels=labels, include_lowest=True)

    try:
        t["mcap_bucket"] = pd.qcut(t["market_cap"].astype(float), q=mcap_q, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=cols)

    t["is_win"] = (t["pnl_pct_net"].astype(float) > 0).astype(int)

    grp_cols = ["score_bin", "mcap_bucket"]

    counts = t.groupby(grp_cols, dropna=False, observed=True).size().rename("trades")

    if value == "win_rate_%":
        metric = t.groupby(grp_cols, dropna=False, observed=True)["is_win"].mean().mul(100.0).rename("metric")
        metric_fmt = lambda x: f"{x:,.2f}"
    elif value == "expectancy_%":
        metric = t.groupby(grp_cols, dropna=False, observed=True)["pnl_pct_net"].mean().rename("metric")
        metric_fmt = lambda x: f"{x:,.2f}"
    elif value == "profit_factor":
        gains = (
            t.assign(gain=t["pnl_pct_net"].where(t["pnl_pct_net"] > 0, 0.0))
            .groupby(grp_cols, dropna=False, observed=True)["gain"]
            .sum()
        )
        losses = (
            t.assign(loss=(-t["pnl_pct_net"]).where(t["pnl_pct_net"] < 0, 0.0))
            .groupby(grp_cols, dropna=False, observed=True)["loss"]
            .sum()
        )
        metric = (gains / losses.replace(0, np.nan)).rename("metric")
        metric_fmt = lambda x: f"{x:,.2f}"
    else:
        raise ValueError("value must be one of: win_rate_%, expectancy_%, profit_factor")

    heat = metric.unstack("score_bin")
    cnt = counts.unstack("score_bin")

    heat.index = [_fmt_interval(c, 2) for c in heat.index]
    cnt.index = heat.index

    heat.columns = heat.columns.astype(str)
    cnt.columns = heat.columns

    combined = heat.copy()
    for col in combined.columns:
        combined[col] = combined[col].combine(
            cnt[col],
            lambda m, n: f"{metric_fmt(m)} ({int(n)})" if pd.notna(m) and pd.notna(n) else "",
        )

    combined = combined.reset_index().rename(columns={"index": "mcap_bucket"})
    combined = combined.reindex(columns=cols, fill_value="")
    return combined


def summarize_heatmap_score_x_mcap_long(
    trades_enriched: pd.DataFrame,
    mcap_q: int = 5,
) -> pd.DataFrame:
    cols = ["mcap_bucket", "score_bin", "metric", "trades"]
    if trades_enriched is None or trades_enriched.empty:
        return pd.DataFrame(columns=cols)

    t = trades_enriched.dropna(subset=["score", "market_cap", "pnl_pct_net"]).copy()
    if t.empty:
        return pd.DataFrame(columns=cols)

    bins = [-1e-9, 0.2, 0.4, 0.6, 0.8, 1.000000001]
    labels = ["0-0.2", "0.2-0.4", "0.4-0.6", "0.6-0.8", "0.8-1.0"]
    t["score_bin"] = pd.cut(t["score"].astype(float), bins=bins, labels=labels, include_lowest=True)

    try:
        t["mcap_bucket"] = pd.qcut(t["market_cap"].astype(float), q=mcap_q, duplicates="drop")
    except Exception:
        return pd.DataFrame(columns=cols)

    t["is_win"] = (t["pnl_pct_net"].astype(float) > 0).astype(int)

    grp_cols = ["score_bin", "mcap_bucket"]
    grouped = t.groupby(grp_cols, dropna=False, observed=True)
    metric = grouped["is_win"].mean().mul(100.0).rename("metric")
    trades = grouped.size().rename("trades")

    out = pd.DataFrame({"metric": metric, "trades": trades}).reset_index()
    out["mcap_bucket"] = out["mcap_bucket"].map(lambda x: _fmt_interval(x, 2))
    out["score_bin"] = out["score_bin"].astype(str)
    return out[cols].sort_values(["mcap_bucket", "score_bin"]).reset_index(drop=True)
