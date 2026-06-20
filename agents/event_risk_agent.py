"""
NSE Momentum v5.0 - EventRiskAgent
Tags each trading day as NORMAL / WATCH / HIGH_RISK
from calendar/event_calendar.json (maintained manually).
"""

import json
import logging
from datetime import date, datetime
from pathlib import Path

log = logging.getLogger(__name__)

CALENDAR_PATH = Path(__file__).parent.parent / "calendar" / "event_calendar.json"


class EventRiskAgent:
    def __init__(self, scan_date: date = None):
        self.scan_date = scan_date or date.today()
        self._state    = "NORMAL"
        self._note     = ""
        self._score_penalty = 0
        self._load()

    def _load(self):
        try:
            if not CALENDAR_PATH.exists():
                return
            with open(CALENDAR_PATH, encoding="utf-8") as f:
                calendar = json.load(f)
            date_str = self.scan_date.isoformat()
            if date_str in calendar:
                event = calendar[date_str]
                self._state = event.get("risk", "NORMAL")
                self._note  = event.get("note", "")
                if self._state == "WATCH":
                    self._score_penalty = 2
                elif self._state == "HIGH_RISK":
                    self._score_penalty = 5
        except Exception as e:
            log.debug("EventRiskAgent load error: %s", e)

    def get_state(self) -> str:
        return self._state

    def get_note(self) -> str:
        return self._note

    def get_score_penalty(self) -> int:
        """Extra points required for T1 promotion on event days."""
        return self._score_penalty

    def blocks_small_cap_t1(self) -> bool:
        return self._state == "HIGH_RISK"
