"""
P0 Step 4: dead-man's switch for the evening scan.

Runs from a separate scheduled workflow after both the cron-job.org
primary trigger (19:30 IST) and the GitHub Actions backstop schedule
(20:30 IST) have had time to fire and finish. If scan_metadata has no row
for today by then, both triggers failed silently (or Turso itself is
down) and this sends an alert email — the point being that a missed scan
should never look identical to a quiet market night.
"""
import datetime as dt
import os
import smtplib
import sys
from email.mime.text import MIMEText

import libsql_client

from market_calendar.staleness_check import is_trading_day, previous_trading_day

IST = dt.timezone(dt.timedelta(hours=5, minutes=30))
# Before this IST hour, tonight's scan isn't due yet, so check the previous
# trading day's scan instead. GitHub cron can start this workflow many hours
# late (25 Nov: the 21:15 IST run started 05:38 IST the next morning), and
# runners use UTC, so date.today() would point at a scan that isn't due yet.
EVENING_SCAN_DUE_HOUR_IST = 21


def expected_scan_date(now_ist: dt.datetime) -> dt.date | None:
    """The trading date whose evening scan should already exist at now_ist,
    or None if there is nothing to check (today isn't a trading day and the
    check is running in its normal evening slot)."""
    today = now_ist.date()
    if now_ist.hour >= EVENING_SCAN_DUE_HOUR_IST:
        return today if is_trading_day(today) else None
    return previous_trading_day(today)

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")
GMAIL_ADDRESS = os.environ.get("GMAIL_ADDRESS")
GMAIL_APP_PASSWORD = os.environ.get("GMAIL_APP_PASSWORD")


def _alert(subject: str, body: str):
    if not GMAIL_ADDRESS or not GMAIL_APP_PASSWORD:
        print(f"ALERT (email unavailable, no Gmail creds): {subject}\n{body}")
        return
    msg = MIMEText(body)
    msg["Subject"] = subject
    msg["From"] = GMAIL_ADDRESS
    msg["To"] = GMAIL_ADDRESS
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as smtp:
        smtp.login(GMAIL_ADDRESS, GMAIL_APP_PASSWORD)
        smtp.sendmail(GMAIL_ADDRESS, [GMAIL_ADDRESS], msg.as_string())
    print(f"Alert email sent: {subject}")


def main() -> int:
    now_ist = dt.datetime.now(IST)
    today = expected_scan_date(now_ist)
    if today is None:
        print(f"{now_ist.date()} is not a trading day — nothing to check.")
        return 0
    print(f"Running at {now_ist:%Y-%m-%d %H:%M} IST; checking scan for {today}.")

    if not TURSO_URL or not TURSO_TOKEN:
        _alert(
            "NSE Momentum: dead-man's switch could not run",
            "TURSO_DATABASE_URL / TURSO_AUTH_TOKEN missing in the "
            "dead-mans-switch workflow — cannot verify today's scan ran.",
        )
        return 1

    url = TURSO_URL.replace("libsql://", "https://")
    client = libsql_client.create_client_sync(url, auth_token=TURSO_TOKEN)
    try:
        rs = client.execute(
            "SELECT scan_date FROM scan_metadata WHERE scan_date = ?",
            [today.isoformat()],
        )
        if rs.rows:
            print(f"OK — scan_metadata has a row for {today}.")
            return 0
        _alert(
            "NSE Momentum: evening scan did not run today",
            f"No scan_metadata row found for {today} as of this check. "
            "Both the cron-job.org primary trigger and the GitHub Actions "
            "backstop schedule appear to have missed today's run — check "
            "the Actions tab and the cron-job.org job history.",
        )
        return 1
    except Exception as e:
        _alert(
            "NSE Momentum: dead-man's switch query failed",
            f"Could not query scan_metadata to verify today's scan: {e}",
        )
        return 1
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
