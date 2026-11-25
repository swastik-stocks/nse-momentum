"""
Live track record of the picks this scanner has actually published.

WHY (2026-11-25)
    The evening email showed a per-pick "Conf: 69-72%". That number was a
    hard-coded seed win rate per pattern (agents/conviction_agent.py
    SEED_WIN_RATES) minus a small-cap discount - never measured, and its
    score adjustment never fired because it assumed Tier 1 gates of 78-82
    while the real gates are 62-66. Every large-cap Swing High pick said 72%.

    This module replaces it with what really happened to published picks:
    every row of Turso's scanner_signals is replayed against real daily
    prices using the SAME rules as the backtests (validation/trade_outcome.py:
    buy-stop fill within 5 sessions, the email's own stop and T1, 20-session
    time exit, stop wins a same-bar tie), one trade per ticker at a time.

    Measured 3 Aug - 24 Nov 2026 with these rules: Tier 1 52% wins, +0.5%
    avg, +1.6% vs Nifty (88 closed); Tier 2 34% wins, -2.7% avg, -1.2% vs
    Nifty (97 closed).

Never raises into the scan: any failure returns None and the email simply
omits the track-record line.
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd

from validation.trade_outcome import simulate_trade

log = logging.getLogger(__name__)

FORWARD_BARS = 20
# Same value as validation/backtest.py's ROUND_TRIP_COST_PCT (0.363%: STT,
# brokerage, exchange/SEBI fees, stamp, GST). Not imported: that module calls
# logging.basicConfig() at import time, which would hijack the scan's logging.
ROUND_TRIP_COST_PCT = 0.00363
MIN_CLOSED_FOR_STATS = 20   # below this, say "too few" instead of a rate


def score_signals(signals: list[dict], price_data: dict,
                  nifty_close: pd.Series | None = None) -> pd.DataFrame:
    """Replay published signals against prices.

    signals    : dicts with ticker, scan_date, tier, entry_price, stop_loss,
                 target1 (scanner_signals rows).
    price_data : {ticker: DataFrame with DatetimeIndex and Open/High/Low/Close}.
    Returns one row per distinct trade (filled or not); repeat signals for a
    ticker whose previous trade is still running are skipped, because a
    reader already holding it would not buy it again.
    """
    rows, busy_until = [], {}
    for s in sorted(signals, key=lambda x: (x["scan_date"], x["ticker"])):
        df = price_data.get(s["ticker"])
        if df is None or df.empty or not s.get("entry_price"):
            continue
        df = df.dropna(subset=["Open", "High", "Low", "Close"])
        sig_day = pd.Timestamp(s["scan_date"])
        idx = int(df.index.searchsorted(sig_day, side="right")) - 1
        if idx < 0 or idx >= len(df) - 1:
            continue                      # no bar after the signal yet
        next_day = df.index[idx + 1]
        if s["ticker"] in busy_until and next_day <= busy_until[s["ticker"]]:
            continue
        t = simulate_trade(df, idx, s["entry_price"], stop=s.get("stop_loss"),
                           target=s.get("target1"), forward_bars=FORWARD_BARS,
                           cost_pct=ROUND_TRIP_COST_PCT)
        rec = {"ticker": s["ticker"], "scan_date": s["scan_date"],
               "tier": str(s.get("tier")), "filled": t is not None}
        if t is not None:
            exit_i = idx + 1 + t["fill_offset"] + t["bars_held"] - 1
            closed = t["exit_reason"] != "TIME" or t["bars_held"] >= FORWARD_BARS
            busy_until[s["ticker"]] = df.index[exit_i]
            rec.update(closed=closed, ret=t["net_return"], r=t["net_r"],
                       exit_reason=t["exit_reason"])
            if nifty_close is not None and not nifty_close.empty:
                try:
                    n0 = nifty_close.asof(df.index[idx])
                    n1 = nifty_close.asof(df.index[exit_i])
                    rec["excess"] = t["net_return"] - (n1 / n0 - 1)
                except Exception:
                    pass
        rows.append(rec)
    return pd.DataFrame(rows)


def summarize(trades: pd.DataFrame) -> dict:
    """{"since", "tiers": {"1": {...}, "2": {...}}} from score_signals output."""
    out = {"since": None, "tiers": {}}
    if trades.empty:
        return out
    out["since"] = str(trades["scan_date"].min())
    filled = trades[trades["filled"]] if "filled" in trades else trades.iloc[0:0]
    for tier in ("1", "2"):
        tt = trades[trades["tier"] == tier]
        f = filled[filled["tier"] == tier]
        closed = f[f["closed"]] if "closed" in f else f.iloc[0:0]
        stats = {
            "signals": int(len(tt)),
            "filled": int(len(f)),
            "closed": int(len(closed)),
            "open": int(len(f) - len(closed)),
            "enough_data": bool(len(closed) >= MIN_CLOSED_FOR_STATS),
        }
        if len(closed):
            stats.update(
                win_rate=round(float((closed["ret"] > 0).mean()), 3),
                avg_return_pct=round(float(closed["ret"].mean() * 100), 2),
                median_return_pct=round(float(closed["ret"].median() * 100), 2),
                avg_r=round(float(closed["r"].mean()), 2),
                # [2026-11-25] How often T1 is actually reached. The cards
                # show R:R to T1, but in the 2016-2026 replay only ~10% of
                # trades reached T1 within 20 sessions; most ended at the
                # stop or the time exit. Shown next to the R:R as a reality check.
                t1_hit_rate=round(float((closed["exit_reason"] == "TARGET").mean()), 3),
            )
            if "excess" in closed and closed["excess"].notna().any():
                stats["avg_vs_nifty_pct"] = round(float(closed["excess"].mean() * 100), 2)
        out["tiers"][tier] = stats
    return out


def build_track_record(price_data: dict, nifty_df: pd.DataFrame | None = None,
                       signals: list[dict] | None = None) -> dict | None:
    """Fetch published signals (unless given) and summarise. None on failure."""
    try:
        if signals is None:
            from turso_sync import get_scanner_signals
            signals = get_scanner_signals()
        if not signals:
            return None
        nifty_close = None
        if nifty_df is not None and not nifty_df.empty:
            nifty_close = nifty_df["Close"].squeeze()
            nifty_close.index = pd.to_datetime(nifty_close.index).tz_localize(None)
        trades = score_signals(signals, price_data, nifty_close)
        summary = summarize(trades)
        summary["as_of"] = date.today().isoformat()
        summary["rules"] = ("fill if price reaches entry within 5 sessions; exit at the "
                            "email's stop or T1, else after 20 sessions; 0.36% costs")
        return summary
    except Exception as e:   # an input to the email, never a reason to fail the scan
        log.warning(f"Live track record unavailable ({type(e).__name__}: {e})")
        return None


def describe_tier(track: dict | None, tier: str) -> str | None:
    """One plain sentence for the email, or None when there's nothing to say."""
    if not track:
        return None
    s = (track.get("tiers") or {}).get(tier)
    if not s or not s.get("closed"):
        return None
    since = track.get("since") or "?"
    if not s.get("enough_data"):
        return (f"Tier {tier} live record since {since}: only {s['closed']} closed "
                f"trades so far — too few to judge.")
    vs = s.get("avg_vs_nifty_pct")
    vs_txt = f", {vs:+.1f}% vs Nifty" if vs is not None else ""
    t1 = s.get("t1_hit_rate")
    t1_txt = f" T1 was reached in {t1:.0%} of them." if t1 is not None else ""
    return (f"Tier {tier} live record since {since}: {s['closed']} closed trades, "
            f"{s['win_rate']:.0%} winners, average {s['avg_return_pct']:+.1f}% per trade"
            f"{vs_txt}.{t1_txt}")
