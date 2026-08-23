"""
NSE Momentum -- Telegram push alerts.

[2026-08-21] Instant, terse companion to the HTML email reports -- fires at
the SAME trigger points confirm_picks.py's checkpoints already decide are
worth an email (a ticker newly CONFIRMED, a BTST candidate), just as a
one-line push instead of a full report. Email stays the detailed record;
this exists because email has to be actively checked, Telegram pushes to
the phone the instant it's sent -- see conversation 2026-08-21, "email has
a headache to babysit every time".

Setup: message @BotFather -> /newbot for TELEGRAM_BOT_TOKEN, then message
the new bot once and read https://api.telegram.org/bot<token>/getUpdates
for the chat's "id" field -> TELEGRAM_CHAT_ID.
"""
import os
import logging

import requests

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

log = logging.getLogger(__name__)

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
TELEGRAM_CHAT_ID   = os.getenv("TELEGRAM_CHAT_ID")


def send_telegram_alert(message: str, parse_mode: str | None = None) -> bool:
    """
    Best-effort push -- never raises. A Telegram outage or missing config
    must never be able to break a real checkpoint/scan run the way an
    unhandled exception here could; this is a bonus notification channel,
    not a dependency the rest of the pipeline can be allowed to inherit.
    Returns False (silently) if not configured or the send failed, so
    callers can log it themselves if they care.

    parse_mode: [2026-08-23] optional "HTML" to let a caller bold/emphasize
    part of the message (e.g. confirm_picks.py's volume-cross alert, made
    bold per explicit user feedback that it's the most time-sensitive
    signal and needs to stand out from EXIT/TRIM at a glance). HTML chosen
    over Telegram's Markdown/MarkdownV2 modes because those require
    escaping characters like '.', '-', '(', ')' that appear constantly in
    prices and tickers -- HTML only needs '<'/'>'/'&' escaped, none of
    which this codebase's alert content ever contains.
    """
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        log.debug("Telegram not configured (TELEGRAM_BOT_TOKEN/TELEGRAM_CHAT_ID unset) -- skipping")
        return False
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": message}
        if parse_mode:
            payload["parse_mode"] = parse_mode
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            data=payload,
            timeout=10,
        )
        r.raise_for_status()
        return True
    except Exception as e:
        log.warning(f"Telegram alert failed (non-fatal): {e}")
        return False
