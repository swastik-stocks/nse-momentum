"""17 Nov 2026 RCA: completed-bar RVOL window, median shadow, re-check crons, labels."""
from datetime import datetime, date, time, timedelta
from unittest.mock import patch

import pytest

from rvol_window import completed_cutoff, rvol_mean_median, window_minutes

D = date(2026, 11, 17)


def at(h, m, s=0, d=D):
    return datetime(d.year, d.month, d.day, h, m, s)


# ── completed_cutoff ─────────────────────────────────────────────────────────

def test_cutoff_excludes_the_still_forming_bar():
    # 10:05:46 -> the 10:05 bar is ~46s old; last COMPLETE bar ends at 10:05.
    assert completed_cutoff(at(10, 5, 46), 5) == time(10, 5)


def test_first_checkpoint_measures_one_full_bar_not_one_and_a_bit():
    # 09:20:48 (the real 09:20 checkpoint): only the 09:15 bar is complete.
    assert completed_cutoff(at(9, 20, 48), 5) == time(9, 20)
    assert window_minutes(time(9, 20)) == 5


def test_no_complete_bar_yet_is_none():
    assert completed_cutoff(at(9, 19, 59), 5) is None


def test_feed_lag_pulls_window_back_to_last_published_bar():
    # Dhan has published bars only up to the 09:55 bar at 10:05:46 -> both
    # today and history must stop at 10:00, not 10:05.
    starts = [at(9, 15) + timedelta(minutes=5 * i) for i in range(9)]  # 09:15..09:55
    assert completed_cutoff(at(10, 5, 46), 5, starts) == time(10, 0)


def test_forming_bar_in_feed_does_not_extend_window():
    starts = [at(9, 15) + timedelta(minutes=5 * i) for i in range(11)]  # ..10:05 (forming)
    assert completed_cutoff(at(10, 5, 46), 5, starts) == time(10, 5)


# ── mean / median ────────────────────────────────────────────────────────────

def test_median_is_robust_to_one_event_day():
    hist = [100, 100, 100, 100, 700]          # one results-day spike
    mean, med = rvol_mean_median(120, hist)
    assert mean == pytest.approx(120 / 220, abs=0.01)   # dragged down by the spike
    assert med == pytest.approx(1.2)


def test_mean_median_ignores_empty_days():
    assert rvol_mean_median(100, [0, 0]) == (None, None)
    assert rvol_mean_median(100, [0, 100]) == (1.0, 1.0)


# ── dhan_rvol.compute_rvol regression (the 17 Nov bias) ─────────────────────

def _bar(dt, vol):
    return {"datetime": dt, "volume": vol, "open": 100, "high": 101, "low": 99, "close": 100}


def test_compute_rvol_no_longer_penalises_forming_bar():
    """Identical volume to history must read ~1.0x, not ~0.5x, at 09:20:48."""
    import dhan_rvol
    hist_days = [date(2026, 11, 13), date(2026, 11, 16)]
    hist = [_bar(datetime.combine(d, t), 1000) for d in hist_days
            for t in (time(9, 15), time(9, 20))]
    today = [_bar(at(9, 15), 1000), _bar(at(9, 20), 80)]   # 09:20 bar 48s old

    with patch.object(dhan_rvol, "_load_instrument_map", return_value={"X": "1"}), \
         patch.object(dhan_rvol, "_fetch_intraday_candles", return_value=hist + today) as f, \
         patch.object(dhan_rvol, "_get_trading_days_back", return_value=hist_days), \
         patch.object(dhan_rvol.time, "sleep"):
        r = dhan_rvol.compute_rvol("X", as_of=at(9, 20, 48))

    # Today and history must come from the SAME request: Dhan builds the
    # 09:15 bar differently for single-day vs multi-day requests (17 Nov
    # KOTAKBANK: 285,771 vs 451,044 shares).
    assert f.call_count == 1
    assert f.call_args.kwargs["from_date"] == "2026-11-13"
    assert f.call_args.kwargs["to_date"] == D.isoformat()

    assert r["rvol"] == pytest.approx(1.0)        # old code: 1080/2000 = 0.54
    assert r["rvol_median"] == pytest.approx(1.0)
    assert r["elapsed_minutes"] == 5 and r["window_end"] == "09:20"


# ── confirm_picks plumbing ───────────────────────────────────────────────────

def test_get_rvol_returns_median_shadow_alongside():
    import confirm_picks as cp
    with patch.object(cp, "_rvol_dhan", return_value=(0.9, 50.0, 1.4)):
        assert cp.get_rvol("YATHARTH", as_of=at(10, 5, 46)) == (0.9, "Dhan", 50.0, 1.4)


def test_median_never_gates_classification():
    import confirm_picks as cp
    pick = {"entry": 100.0, "sl": 95.0, "pivot": 100.0, "t1": 110.0}
    c = cp.classify(pick, cmp=100.5, rvol=0.9, rvol_src="Dhan")
    assert c["status"] == "BREAKOUT_NO_VOLUME"   # 0.9x mean still blocks


def test_drift_reference_price_is_exposed_for_the_email():
    import confirm_picks as cp
    # 17 Nov KOTAKBANK: entry 488, opening-range high 497.1 > pivot -> anchor = OR high.
    pick = {"entry": 488.0, "sl": 472.6, "pivot": 481.9, "t1": 576.8}
    c = cp.classify(pick, cmp=492.7, rvol=0.3, rvol_src="Dhan",
                    opening_range=(497.1, 492.9, "dhan"))
    assert c["drift_anchor_is_or"] and c["drift_anchor"] == 497.1
    assert c["drift_pct"] == -0.9
    html = cp._row({"pick": {**pick, "ticker": "KOTAKBANK"}, "classification": c})
    assert "vs OR-high ₹497.1" in html


def test_recheck_crons_are_labelled_and_not_final():
    # Plain-text checks: the CI test job doesn't install PyYAML.
    import re
    from pathlib import Path
    wf_text = (Path(__file__).resolve().parents[1] / ".github/workflows/confirm.yml").read_text(encoding="utf-8")
    crons = set(re.findall(r"- cron: '([^']+)'", wf_text))
    assert crons == {"5 5 * * 1-5", "35 5 * * 1-5"}
    for cron, label in (("5 5 * * 1-5", "10:35"), ("35 5 * * 1-5", "11:05")):
        line = next(l for l in wf_text.splitlines() if f'"{cron}")' in l)
        assert f"label={label}" in line and 'final=" >>' in line
    assert re.search(r"^concurrency:\n\s+group: confirm-checkpoints\n\s+cancel-in-progress: false",
                     wf_text, re.M)
