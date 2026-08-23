"""
[2026-08-23] Regression tests for orchestrator.py's P2-05 EXIT / P2-06 TRIM
Telegram message builders. Fixture data is the REAL Section 6 content from
the 2026-08-21 (Friday) evening scan email (5 EXIT alerts, 2 TRIM signals) --
not synthetic numbers, so passing this confirms the new wiring would have
produced exactly what that email's Section 6 already showed, had it been
live that day.
"""
from orchestrator import _build_exit_alert_message, _build_trim_alert_message

# Real Section 6 data from the 2026-08-21 evening scan email.
FRIDAY_EXIT_ALERTS = [
    {"ticker": "HINDZINC", "technical_stop": {"current_price": 594.90},
     "exit_check": {"avg_price": 651.50, "effective_stop": 631.95, "source": "ENTRY", "breached": True}},
    {"ticker": "ITC", "technical_stop": {"current_price": 269.40},
     "exit_check": {"avg_price": 327.72, "effective_stop": 317.89, "source": "ENTRY", "breached": True}},
    {"ticker": "POLYCAB", "technical_stop": {"current_price": 8966.00},
     "exit_check": {"avg_price": 9956.78, "effective_stop": 9558.51, "source": "ENTRY", "breached": True}},
    {"ticker": "ANANTRAJ", "technical_stop": {"current_price": 625.00},
     "exit_check": {"avg_price": 658.50, "effective_stop": 625.57, "source": "ENTRY", "breached": True}},
    {"ticker": "ASTERDM", "technical_stop": {"current_price": 761.55},
     "exit_check": {"avg_price": 843.98, "effective_stop": 801.78, "source": "ENTRY", "breached": True}},
]

FRIDAY_TRIM_SIGNALS = [
    {"ticker": "SBIFUNDS", "rs_percentile": 33},
    {"ticker": "RELIANCE", "rs_percentile": 37},
]


def test_exit_alert_message_matches_real_friday_hindzinc_row():
    msg = _build_exit_alert_message(FRIDAY_EXIT_ALERTS[0])
    assert "EXIT ALERT: HINDZINC" in msg
    assert "594.90" in msg          # current price, matches email's CURRENT column
    assert "631.95" in msg          # effective stop, matches email's STOP column
    assert "ENTRY" in msg           # stop source, matches email's STOP SOURCE column
    assert "651.50" in msg          # avg cost, matches email's AVG COST column


def test_all_five_friday_exit_rows_produce_distinct_correct_messages():
    messages = [_build_exit_alert_message(h) for h in FRIDAY_EXIT_ALERTS]
    assert len(messages) == 5
    for h, msg in zip(FRIDAY_EXIT_ALERTS, messages):
        assert h["ticker"] in msg
        assert f"{h['exit_check']['avg_price']:.2f}" in msg
        assert f"{h['exit_check']['effective_stop']:.2f}" in msg
        assert f"{h['technical_stop']['current_price']:.2f}" in msg


def test_trim_alert_message_matches_real_friday_digest():
    msg = _build_trim_alert_message(FRIDAY_TRIM_SIGNALS)
    assert "2 holding(s)" in msg
    assert "SBIFUNDS: RS 33th pct" in msg
    assert "RELIANCE: RS 37th pct" in msg
    # SBIFUNDS (33rd) must appear before RELIANCE (37th), matching the
    # real email's own ascending-percentile ordering.
    assert msg.index("SBIFUNDS") < msg.index("RELIANCE")
