"""
[2026-08-21] Regression test for emailer.py's Section 1 trade-card/heatmap
render caps. Confirmed live 2026-08-20: orchestrator.py legitimately
approved 11 Tier 1 picks (t1_cap=15, sector gate + sanity gate both passed
clean), but _build_html's hardcoded t1[:8] silently rendered only 8 --
AUBANK, APLAPOLLO, JINDALSAW never appeared anywhere in the email despite
being fully qualified, yet still sat in picks_latest.json and got
auto-confirmed by the next morning's checkpoint with the user never having
seen them. t1/t2 render caps are removed (both already bounded by real
upstream caps in orchestrator.py); this pins that all of t1 -- not just
the first 8 -- makes it into the rendered HTML.
"""
from emailer import _build_html
from orchestrator import StockResult


def _fake_t1(n: int) -> list:
    return [
        StockResult(ticker=f"TICK{i}.NS", name=f"Ticker {i} Ltd", sector="IT",
                    price=100.0 + i, tier=1, total_score=70 + i,
                    pattern="Cup & Handle", entry=100.0 + i, stop_loss=95.0,
                    target1=120.0, target2=130.0, rrr=3.0, rs_percentile=80.0,
                    rvol=1.0, del_pct=40.0, universe="MID",
                    breakout_quality="MAJOR")
        for i in range(n)
    ]


def test_all_eleven_t1_picks_appear_in_html_not_just_first_eight():
    t1 = _fake_t1(11)
    html = _build_html(
        t1=t1, t2=[], t3=[], all_r=t1,
        near_bo=[], defensive=[],
        regime="B", rlbl="BULL", rcol="#0E8F63", rbg="#F7F9FC", rborder="#0E8F63",
        rnote="Good conditions.",
        breadth=6, bd={}, date_str="20 Aug 2026", penalty=0,
        macro_state="MIXED", event_risk="NORMAL", t1_cap=15,
    )
    for i in range(11):
        assert f"TICK{i}.NS" in html or f"Ticker {i} Ltd" in html, \
            f"TICK{i} missing from the email -- t1 render cap silently dropped a qualified pick"
