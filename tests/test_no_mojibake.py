"""
Regression guard: email-facing source files must not contain mojibake.

WHY: on 2026-11-07 emailer.py was found with ~60 corrupted literals
(e.g. a warning sign stored as 'Î“ÃœÃ¡', rupee as 'â‚¹') -- UTF-8 text that
had been round-tripped through cp437/cp1252 by a Windows tool, once or
twice. Python happily ran it, so nothing failed; it only showed up as
garbage in every recipient's inbox. This test fails at the source instead.

Detection: a run of non-ASCII chars is mojibake if re-encoding it as
cp1252 or cp437 yields valid UTF-8 that differs from the input. Correctly
encoded text (a real em dash, rupee, emoji) never passes that round-trip.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
EMAIL_FACING = ["emailer.py", "data_fetcher.py", "orchestrator.py", "confirm_picks.py"]


def _looks_mojibaked(run: str) -> bool:
    for enc in ("cp1252", "cp437"):
        try:
            if run.encode(enc).decode("utf-8") != run:
                return True
        except (UnicodeEncodeError, UnicodeDecodeError):
            continue
    return False


@pytest.mark.parametrize("name", EMAIL_FACING)
def test_no_mojibake(name):
    path = ROOT / name
    if not path.exists():
        pytest.skip(f"{name} not present")
    text = path.read_text(encoding="utf-8-sig")
    bad = [(i + 1, m.group()) for i, line in enumerate(text.splitlines())
           for m in re.finditer(r"[^\x00-\x7f]+", line) if _looks_mojibaked(m.group())]
    assert not bad, f"{name}: mojibake at lines {[ln for ln, _ in bad[:10]]}"
