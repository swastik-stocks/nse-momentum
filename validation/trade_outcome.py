"""
Shared, path-dependent trade simulator for every validation/backtest script.

WHY THIS EXISTS (2026-11-25)
    Sixteen validation scripts each carried a copy of the same
    `_forward_test_net`, which scored a signal as:
        win  -> if the HIGHEST high in the next 20 bars was >= +5% net,
                booked AT that highest high;
        loss -> otherwise, booked AT the LOWEST low in those 20 bars.
    That is maximum-favourable vs maximum-adverse excursion, not a trade:
    it ignores whether the stop was hit before the high, books winners at
    the exact top and losers at the exact bottom, never checks that the
    breakout entry was ever reached, and ignores the stop/target the live
    email actually prints. Re-scoring the deep replay's 2,280 Swing High
    Breakout / Cup & Handle signals with tradeable exits moved them from
    +0.67R / +0.37R to +0.03R / -0.01R per trade - i.e. the "SIGNIFICANT"
    verdicts came from the scoring method, not the patterns.

RULES (what a reader following the email could actually get)
    Fill   : buy-stop at `entry`. The first of the next `fill_window` bars
             whose High reaches `entry` fills it - at the Open if that bar
             gapped above `entry`, else at `entry`. Never reached -> None
             (no trade; callers must EXCLUDE it, not score it as zero).
    Exit   : from the fill bar, bar by bar for up to `forward_bars` bars:
               - Open at/below stop (gap down, after the fill bar) -> Open
               - Low at/below stop                                -> stop
               - High at/above target                             -> target
             A bar that touches both counts as the STOP (daily bars can't
             tell which came first; assuming the worse one keeps results
             honest). Neither hit -> the last available bar's Close.
    R unit : fill - stop (the risk actually taken), so +1R / -1R mean the
             same thing for every trade. Costs subtracted once.
    Defaults when the caller has no stop/target: stop = entry - 5%
    (the old fixed 1R unit), target = fill + 2R.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_RISK_PCT = 0.05
DEFAULT_TARGET_R = 2.0
FILL_WINDOW = 5


def simulate_trade(
    df: pd.DataFrame,
    signal_idx: int,
    entry: float,
    stop: float | None = None,
    target: float | None = None,
    forward_bars: int = 20,
    cost_pct: float = 0.0,
    fill_window: int = FILL_WINDOW,
    require_full_window: bool = False,
) -> dict | None:
    """Simulate one long trade signalled on bar `signal_idx` (decision made
    after that bar's close; orders work from bar signal_idx+1).

    Returns None when there is no trade (entry never reached, bad inputs,
    or - with require_full_window - not enough future bars to judge it).
    Otherwise a dict:
        is_win, net_r, gross_r, net_return, gross_return, fill_price,
        exit_price, exit_reason (STOP | STOP_GAP | TARGET | TIME),
        bars_held, risk_pct, filled (always True)
    """
    if entry is None or entry <= 0:
        return None
    future = df.iloc[signal_idx + 1:]
    if future.empty:
        return None
    if require_full_window and len(future) < forward_bars:
        return None

    opens = future["Open"].to_numpy(dtype=float)
    highs = future["High"].to_numpy(dtype=float)
    lows = future["Low"].to_numpy(dtype=float)
    closes = future["Close"].to_numpy(dtype=float)

    # ── Fill ────────────────────────────────────────────────────────────
    k = None
    for j in range(min(fill_window, len(future))):
        if highs[j] >= entry:
            k = j
            break
    if k is None:
        return None
    fill = max(opens[k], entry)

    # ── Stop / target relative to the actual fill ───────────────────────
    if stop is None or not (0 < stop < fill):
        stop = min(entry, fill) * (1 - DEFAULT_RISK_PCT)
        if stop >= fill:
            stop = fill * (1 - DEFAULT_RISK_PCT)
    risk = fill - stop
    if target is None or target <= fill:
        # A target at/below the fill (e.g. the open gapped over it) means
        # the limit sell fills immediately; DEFAULT keeps the old 2R unit
        # only when the caller gave no target at all.
        target = fill + DEFAULT_TARGET_R * risk if target is None else fill

    # ── Walk forward ────────────────────────────────────────────────────
    last = min(k + forward_bars, len(future))
    exit_px, reason, held = None, None, 0
    for j in range(k, last):
        held = j - k + 1
        if j > k and opens[j] <= stop:
            exit_px, reason = opens[j], "STOP_GAP"
            break
        if lows[j] <= stop:
            exit_px, reason = stop, "STOP"
            break
        if highs[j] >= target:
            exit_px, reason = (opens[j] if j > k and opens[j] > target else target), "TARGET"
            break
    if exit_px is None:
        exit_px, reason = closes[last - 1], "TIME"

    gross_ret = exit_px / fill - 1
    net_ret = gross_ret - cost_pct
    risk_pct = risk / fill
    return {
        "is_win": bool(net_ret > 0),   # plain Python types: callers json.dumps this
        "net_r": round(float(net_ret / risk_pct), 3),
        "gross_r": round(float(gross_ret / risk_pct), 3),
        "net_return": round(float(net_ret), 5),
        "gross_return": round(float(gross_ret), 5),
        "fill_price": round(float(fill), 4),
        "exit_price": round(float(exit_px), 4),
        "exit_reason": str(reason),
        "bars_held": int(held),
        "fill_offset": int(k),        # bars after the signal bar's next bar (0 = next bar)
        "risk_pct": round(float(risk_pct), 5),
        "filled": True,
    }
