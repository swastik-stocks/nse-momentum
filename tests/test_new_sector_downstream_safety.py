"""
[2026-09-07] Downstream-impact guardrails for the "NEW SECTOR" badge
feature. Two systems outside the email consume orchestrator.py's output
and must be completely unaffected by the new StockResult.sector_is_new
field and the new sector_pick_history table:

  1. picks_latest.json -- read by confirm_picks.py and Portfolio Dashboard.
  2. Turso's scanner_signals table (turso_sync.publish_signals) -- the
     P1-03 bridge Portfolio Dashboard reads live.

Both are schema contracts owned by another consumer; silently adding a
column/key to either would be exactly the kind of "downstream impact"
this feature must not cause.
"""
import re

import turso_sync
from orchestrator import StockResult


def test_picks_json_dict_literal_does_not_reference_sector_is_new():
    """Static guard on orchestrator.py's picks_json construction (an
    inline dict literal inside run_universe(), not a standalone testable
    function -- see the comment above it re: picks_latest.json's schema
    history). Protects against a future edit accidentally adding
    sector_is_new to the Portfolio Dashboard-facing JSON payload, which
    was never part of this feature's scope."""
    src = open("orchestrator.py", encoding="utf-8").read()
    m = re.search(r"picks_json = \[\](.*?)with open\(\"picks_latest\.json\"", src, re.S)
    assert m, "picks_json construction block not found -- orchestrator.py structure changed"
    assert "sector_is_new" not in m.group(1)


def test_stock_result_with_sector_is_new_still_publishes_expected_columns(monkeypatch):
    """publish_signals() (P1-03, scanner_signals) reads explicit
    attributes off StockResult, not asdict()/fields() -- confirms adding
    sector_is_new didn't change what gets sent to Turso."""
    sent = {}

    class _FakeClient:
        def execute(self, sql, params=None):
            sent["sql"] = sql
            sent["params"] = params
        def close(self):
            pass

    monkeypatch.setattr(turso_sync, "get_client", lambda: _FakeClient())

    r = StockResult(
        ticker="JSWSTEEL.NS", name="JSW Steel Ltd.", sector="Metals", price=1323.9,
        universe="LARGE", tier=3, pattern="Cup & Handle", entry=1323.9,
        stop_loss=1284.2, target1=1436.1, target2=1579.7, rrr=2.8,
        rs_percentile=56.0, total_score=52, confidence_pct=61.0,
        breadth_score=6, sector_is_new=True,   # the new field, set
    )
    turso_sync.publish_signals([r], regime="C", scan_date="2026-09-04")

    assert "sector_is_new" not in sent["sql"]
    assert "sector_pick_history" not in sent["sql"]
    assert "scanner_signals" in sent["sql"]
    # exactly the 16 columns this table has always had (see SCHEMA_STATEMENTS)
    assert sent["params"] is not None and len(sent["params"]) == 16
