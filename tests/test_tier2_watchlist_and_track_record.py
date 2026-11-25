"""
[2026-11-25] Fixes 2 and 3 from the pick-quality investigation:

  - Tier 2 is a watchlist, not a buy signal: no trade cards in the evening
    email, no "CONFIRMED — ENTER" / Telegram / BTST in the morning flow.
    (Replaying every published Tier 2 pick since 3 Aug: 34% winners, -2.7%.)
  - The per-pick "Conf: x%" (a hard-coded seed rate) is gone; each tier shows
    its measured live record from live_track_record.py instead.
"""
import pandas as pd

from emailer import _build_html
from orchestrator import StockResult
from live_track_record import score_signals, summarize, describe_tier
import confirm_picks


def _pick(ticker, tier, score):
    return StockResult(ticker=ticker, name=f"{ticker} Ltd", sector="IT", price=100.0,
                       tier=tier, total_score=score, pattern="Swing High Breakout",
                       entry=100.0, stop_loss=95.0, target1=110.0, target2=120.0,
                       rrr=2.0, rs_percentile=80.0, rvol=1.0, del_pct=40.0,
                       universe="MID", breakout_quality="MINOR", confidence_pct=72.0)


TRACK = {"since": "2026-08-03", "rules": "test rules",
         "tiers": {"1": {"closed": 88, "enough_data": True, "win_rate": 0.52,
                         "avg_return_pct": 0.5, "avg_vs_nifty_pct": 1.6},
                   "2": {"closed": 97, "enough_data": True, "win_rate": 0.34,
                         "avg_return_pct": -2.7, "avg_vs_nifty_pct": -1.2}}}


def _html(t1, t2, track=TRACK):
    return _build_html(
        t1=t1, t2=t2, t3=[], all_r=t1 + t2, near_bo=[], defensive=[],
        regime="C", rlbl="RANGE", rcol="#000", rbg="#fff", rborder="#000",
        rnote="Choppy.", breadth=6, bd={}, date_str="23 Nov 2026", penalty=-5,
        macro_state="MIXED", event_risk="NORMAL", t1_cap=8, track_record=track)


# ── Evening email ────────────────────────────────────────────────────────────

def test_tier2_has_no_trade_card_but_stays_in_watchlist_table():
    html = _html([_pick("AAA.NS", 1, 70)], [_pick("BBB.NS", 2, 60)])
    assert "TIER 2 - AGGRESSIVE" not in html
    assert "not a buy signal" in html
    assert "BBB" in html                       # still visible, in the table


def test_conf_percentage_is_gone_and_track_records_are_shown():
    html = _html([_pick("AAA.NS", 1, 70)], [_pick("BBB.NS", 2, 60)])
    assert "Conf:" not in html and "CONF." not in html
    assert "Tier 1 live record since 2026-08-03: 88 closed trades, 52% winners" in html
    assert "Tier 2 live record since 2026-08-03: 97 closed trades, 34% winners" in html


def test_email_renders_without_a_track_record():
    html = _html([_pick("AAA.NS", 1, 70)], [], track=None)
    assert "live record" not in html and "AAA" in html


# ── Morning confirmation ─────────────────────────────────────────────────────

def _result(tier, status="CONFIRMED"):
    return {"pick": {"ticker": "X", "tier": tier, "entry": 100.0, "sl": 95.0, "t1": 110.0},
            "classification": {"status": status, "label": "CONFIRMED — ENTER",
                               "action": "Enter", "color": "#fff", "bg": "#16a34a",
                               "ltp": 101.0, "gap_pct": 0.5, "rvol": 1.8, "rvol_src": "test",
                               "drift_pct": 1.0, "is_chasing": False}}


def test_tier2_breakout_is_shown_as_watchlist_and_original_is_untouched():
    r = _result(2)
    shown = confirm_picks._as_watchlist(r, {"live_track_record": TRACK})
    assert shown["classification"]["status"] == "WATCHLIST"
    assert "NOT A BUY" in shown["classification"]["label"]
    assert "34% winners" in shown["classification"]["action"]
    assert r["classification"]["status"] == "CONFIRMED"   # recorded outcome unchanged


def test_tier1_and_non_entry_statuses_are_left_alone():
    assert confirm_picks._as_watchlist(_result(1), {})["classification"]["status"] == "CONFIRMED"
    assert confirm_picks._as_watchlist(_result(2, "MISSED"), {})["classification"]["status"] == "MISSED"


def test_tier2_confirmation_does_not_trigger_action_box():
    html = confirm_picks.build_html([_result(2)], "2026-11-24", "09:35", meta={})
    assert "ACTION REQUIRED" not in html
    assert "T2 WATCHLIST" in html


# ── live_track_record ────────────────────────────────────────────────────────

def _prices(rows):
    idx = pd.bdate_range("2026-08-03", periods=len(rows))
    return pd.DataFrame(rows, index=idx, columns=["Open", "High", "Low", "Close"])


def test_track_record_scores_stop_target_and_skips_repeat_signals():
    win = _prices([(99, 100, 98, 99), (100, 101, 99, 100), (101, 111, 100, 110)] + [(110, 111, 109, 110)] * 25)
    loss = _prices([(99, 100, 98, 99), (100, 101, 99, 100), (99, 99, 94, 95)] + [(95, 96, 94, 95)] * 25)
    sigs = [
        {"ticker": "W", "scan_date": "2026-08-03", "tier": 1, "entry_price": 100, "stop_loss": 95, "target1": 110},
        {"ticker": "W", "scan_date": "2026-08-04", "tier": 1, "entry_price": 100, "stop_loss": 95, "target1": 110},
        {"ticker": "L", "scan_date": "2026-08-03", "tier": 2, "entry_price": 100, "stop_loss": 95, "target1": 110},
    ]
    t = score_signals(sigs, {"W": win, "L": loss})
    assert len(t) == 2                          # W's repeat while in the trade is skipped
    s = summarize(t)
    assert s["tiers"]["1"]["win_rate"] == 1.0 and s["tiers"]["1"]["closed"] == 1
    assert s["tiers"]["2"]["win_rate"] == 0.0
    assert not s["tiers"]["1"]["enough_data"]   # 1 trade is too few to quote a rate
    assert "too few to judge" in describe_tier(s, "1")
