"""
[2026-09-07] Regression tests for emailer.py's "NEW SECTOR" badge --
renders whatever StockResult.sector_is_new / near_breakout["is_new_sector"]
it's handed (computed upstream by sector_novelty.py). These tests only
cover the render layer: given the flag is already set, does the right
HTML come out in every place it's supposed to, and -- just as important --
does everything else stay unchanged when the flag is False/missing.

Fixture data is the real 04-Sep-2026 email content (see the Gmail export
that prompted this feature): JSWSTEEL (Tier 3) and JINDALSTEL
(Near-Breakout) are this scenario's "first appearance of Metals."
"""
from emailer import _build_html, _new_sector_badge
from orchestrator import StockResult


def _t1_pick(sector_is_new=False):
    return StockResult(
        ticker="IPCALAB.NS", name="Ipca Laboratories Ltd.", sector="Pharma",
        price=1957.0, tier=1, total_score=68, confidence_pct=58.0,
        pattern="Cup & Handle", entry=1957.0, stop_loss=1878.7,
        target1=2375.8, target2=2634.6, rrr=5.3, rs_percentile=92.0,
        rvol=0.1, del_pct=57.0, universe="MID", breakout_quality="MINOR",
        sector_is_new=sector_is_new,
    )


def _t3_pick_jswsteel(sector_is_new=True):
    return StockResult(
        ticker="JSWSTEEL.NS", name="JSW Steel Ltd.", sector="Metals",
        price=1323.9, tier=3, total_score=52, confidence_pct=61.0,
        pattern="Cup & Handle", entry=1323.9, stop_loss=1284.2,
        target1=1436.1, target2=1579.7, rrr=2.8, rs_percentile=56.0,
        rvol=0.7, del_pct=53.0, universe="LARGE", breakout_quality="MINOR",
        sector_is_new=sector_is_new,
    )


def _near_breakout_jindalstel(is_new_sector=True):
    return {
        "ticker": "JINDALSTEL.NS", "name": "Jindal Steel Ltd.",
        "pattern": "Cup & Handle", "price": 1165.0, "breakout": 1170.7,
        "gap_pct": 0.5, "rsi": 63, "universe": "LARGE", "sector": "Metals",
        "is_new_sector": is_new_sector,
    }


def _render(t1=None, t3=None, near_bo=None):
    t1 = t1 or []
    t3 = t3 or []
    near_bo = near_bo or []
    return _build_html(
        t1=t1, t2=[], t3=t3, all_r=t1 + t3,
        near_bo=near_bo, defensive=[],
        regime="C", rlbl="RANGE BOUND", rcol="#96690A", rbg="#F7F9FC",
        rborder="#96690A", rnote="Choppy market.",
        breadth=6, bd={}, date_str="04 Sep 2026", penalty=-5,
        macro_state="MIXED", event_risk="NORMAL", t1_cap=12,
    )


# ---------------------------------------------------------------------
# _new_sector_badge() unit behavior
# ---------------------------------------------------------------------

def test_badge_helper_renders_nothing_when_not_new():
    assert _new_sector_badge(False) == ""
    assert _new_sector_badge(False, compact=True) == ""


def test_badge_helper_renders_full_and_compact_variants():
    full = _new_sector_badge(True)
    compact = _new_sector_badge(True, compact=True)
    assert "NEW SECTOR" in full
    assert "NEW" in compact
    assert full != compact


# ---------------------------------------------------------------------
# Trade card (Section 1)
# ---------------------------------------------------------------------

def test_new_sector_badge_appears_on_flagged_trade_card():
    html = _render(t3=[_t3_pick_jswsteel(sector_is_new=True)])
    assert "NEW SECTOR" in html


def test_no_badge_on_unflagged_trade_card():
    html = _render(t1=[_t1_pick(sector_is_new=False)])
    assert "NEW SECTOR" not in html
    assert ">NEW<" not in html


def test_badge_does_not_leak_onto_other_cards_same_email():
    """The critical case: one flagged pick (JSWSTEEL) alongside one
    unflagged pick (IPCALAB) in the same email -- only JSWSTEEL's card
    gets the badge."""
    html = _render(t1=[_t1_pick(sector_is_new=False)],
                    t3=[_t3_pick_jswsteel(sector_is_new=True)])
    assert html.count("NEW SECTOR") == 1


# ---------------------------------------------------------------------
# Near-Breakout table (Section 4) -- the exact gap the user flagged
# ---------------------------------------------------------------------

def test_near_breakout_row_gets_compact_badge_when_flagged():
    html = _render(near_bo=[_near_breakout_jindalstel(is_new_sector=True)])
    assert "JINDALSTEL" in html
    # compact chip, not the full card-style badge
    assert ">NEW<" in html


def test_near_breakout_row_has_no_badge_when_not_flagged():
    html = _render(near_bo=[_near_breakout_jindalstel(is_new_sector=False)])
    assert "JINDALSTEL" in html
    assert ">NEW<" not in html


def test_near_breakout_missing_is_new_sector_key_defaults_to_no_badge():
    """Defensive: near_breakout dicts that predate this feature (or a
    future caller that forgets the key) must render cleanly, not KeyError."""
    nb = _near_breakout_jindalstel()
    del nb["is_new_sector"]
    html = _render(near_bo=[nb])   # must not raise
    assert "JINDALSTEL" in html
    assert ">NEW<" not in html


# ---------------------------------------------------------------------
# Top-20 watchlist table (Section 2) and At-A-Glance heatmap (Section 1a)
# ---------------------------------------------------------------------

def test_top20_table_shows_badge_for_flagged_tier3_pick():
    # JSWSTEEL is T3 -- Top-20 pulls from all_r (T1 excluded), so it
    # must appear there even when it's not one of the t3[:2] full cards.
    html = _render(t3=[_t3_pick_jswsteel(sector_is_new=True)])
    assert html.count(">NEW<") >= 1   # at least the Top-20 row chip


def test_at_a_glance_heatmap_shows_badge_for_flagged_t1_pick():
    html = _render(t1=[_t1_pick(sector_is_new=True)])
    assert "At A Glance" in html
    assert ">NEW<" in html


# ---------------------------------------------------------------------
# Top-of-email summary pill
# ---------------------------------------------------------------------

def test_summary_pill_lists_new_sector_once():
    html = _render(t3=[_t3_pick_jswsteel(sector_is_new=True)],
                    near_bo=[_near_breakout_jindalstel(is_new_sector=True)])
    assert "NEW 1 sector: Metals" in html


def test_summary_pill_absent_when_nothing_new():
    html = _render(t1=[_t1_pick(sector_is_new=False)])
    assert "NEW 1 sector" not in html
    assert "NEW 2 sector" not in html


def test_summary_pill_lists_multiple_new_sectors_alphabetically():
    metals = _t3_pick_jswsteel(sector_is_new=True)
    it_pick = _t1_pick(sector_is_new=True)
    it_pick.sector = "Information Technology"
    html = _render(t1=[it_pick], t3=[metals])
    assert "Information Technology, Metals" in html


# ---------------------------------------------------------------------
# Downstream-safety: StockResult without the field explicitly set
# ---------------------------------------------------------------------

def test_stock_result_defaults_sector_is_new_to_false():
    """A StockResult built by code that predates this feature (or any
    call site that never learns about sector_is_new) must default safely
    -- this is what makes the field additive/backward-compatible."""
    r = StockResult(ticker="FOO.NS", name="Foo Ltd", sector="Auto", price=100.0)
    assert r.sector_is_new is False
    html = _render(t1=[r])
    assert "NEW SECTOR" not in html
