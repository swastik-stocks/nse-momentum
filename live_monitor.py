"""
NSE Momentum — Live Breakout Monitor (Dhan WebSocket)

Real-time alternative to confirm.yml's periodic 09:16/20/35/55
checkpoints. Subscribes to live Dhan quote ticks for today's picks and
emails the INSTANT a pick's status transitions to CONFIRMED or
CONFIRMED_LOW_VOL -- reusing the EXACT same classify() decision logic
as confirm_picks.py (imported directly, not reimplemented), so this
system and the periodic-checkpoint system can never silently disagree.

SCOPE, per the 2026-07-30 hourly evidence chain (531 tickers, 10,010
trades, bootstrap + walk-forward split-period tested): only Cup & Handle
(rho=+0.32/+0.33 both halves) and Swing High Breakout (rho=+0.09 both
halves) hold up at hourly resolution. VCP does NOT (avg R -0.89, p=0.98,
negative in the pre-2021 half) -- excluded from live triggering entirely,
regardless of what's in picks_latest.json.

REQUIRES: Dhan Data API subscription (~Rs.499+tax/month). Live Market
Feed via WebSocket is a SEPARATE paid tier from the historical/free-tier
access already used by daily_scan.yml/confirm.yml. If you see disconnect
code 806 ("Subscribe to Data APIs to continue"), that's this.

VERIFIED SOURCE: this was built by reading the actual dhanhq==2.2.0
package source directly (marketfeed.py), not blog examples -- those
showed real API drift across versions (some use marketfeed.DhanFeed,
current is DhanContext+MarketFeed). Packet parsing, subscribe format,
and callback signatures below match the real, current package.

UNTESTED against a live connection -- no network path to
wss://api-feed.dhan.co from the environment this was built in. The
on_message/classify wiring is tested against synthetic tick data (see
test comments), but the actual WebSocket handshake, binary parsing at
scale, and reconnect behavior have never run against Dhan's real feed.
Run --dry-run first, during real market hours, before trusting this.

DESIGN: persistent, long-running process. NOT meant for GitHub Actions
(short-lived runners don't fit a live WebSocket connection) -- run
locally or on a small always-on host during market hours (09:15-15:30
IST).

Usage:
    python live_monitor.py              # live, sends real alert emails
    python live_monitor.py --dry-run    # subscribe + classify + log only
"""

import os
import sys
import json
import time
import logging
import argparse
import smtplib
from datetime import datetime, time as dtime
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from dotenv import load_dotenv
load_dotenv()

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dhanhq import DhanContext, MarketFeed

from confirm_picks import classify, REGIME_TOLERANCE, MARKET_OPEN_TIME, IST
from data_fetcher import _dhan_symbol_map, _symbol_from_ticker, fetch_dhan, _check_dhan_auth

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)

MARKET_CLOSE_TIME = dtime(15, 30)

# [SCOPE] See module docstring -- only these two patterns are validated
# at hourly/intraday resolution. A VCP pick sitting in picks_latest.json
# is still tracked in logs but will never trigger an alert email.
LIVE_TRIGGER_PATTERNS = {"Cup & Handle", "Swing High Breakout"}

ALERTED_STATE_FILE = "logs/live_monitor_alerted.json"


def _load_picks(path: str = "picks_latest.json") -> list:
    with open(path) as f:
        return json.load(f)


def _load_alerted_state() -> set:
    """Persisted across restarts -- a mid-day restart shouldn't re-alert
    something already emailed earlier today."""
    try:
        with open(ALERTED_STATE_FILE) as f:
            data = json.load(f)
        if data.get("date") == datetime.now(IST).strftime("%Y-%m-%d"):
            return set(data.get("alerted", []))
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return set()


def _save_alerted_state(alerted: set) -> None:
    os.makedirs("logs", exist_ok=True)
    with open(ALERTED_STATE_FILE, "w") as f:
        json.dump({"date": datetime.now(IST).strftime("%Y-%m-%d"), "alerted": list(alerted)}, f)


def send_alert_email(ticker: str, pick: dict, classification: dict, ltp: float) -> None:
    gmail_address = os.environ.get("GMAIL_ADDRESS")
    gmail_password = os.environ.get("GMAIL_APP_PASSWORD")
    if not gmail_address or not gmail_password:
        log.warning(f"  GMAIL credentials not set — would have alerted {ticker} but cannot send email")
        return

    status = classification["status"]
    action = "FULL SIZE" if status == "CONFIRMED" else "HALF SIZE"
    ticker_disp = ticker.replace(".NS", "")
    subject = f"\U0001F514 BREAKOUT CONFIRMED: {ticker_disp} — {action} — Enter Now"

    t2_val = pick.get("t2", 0) or 0
    body = f"""
    <h2 style="color:#16a34a">{ticker_disp} — {pick.get('pattern', '')}</h2>
    <p><b>Status:</b> {status} ({action})</p>
    <p><b>Live Price:</b> Rs.{ltp:,.2f}</p>
    <p><b>Entry:</b> Rs.{pick['entry']:,.2f} &nbsp; <b>SL:</b> Rs.{pick['sl']:,.2f}
       &nbsp; <b>T1:</b> Rs.{pick['t1']:,.2f} &nbsp; <b>T2:</b> Rs.{t2_val:,.2f}</p>
    <p><b>Score:</b> {pick.get('score', '-')} &nbsp; <b>Sector:</b> {pick.get('sector', '-')}</p>
    <p style="color:#6b7280;font-size:12px">Real-time alert via Dhan Live Market Feed —
       reuses the exact same classify() logic as the 09:16/20/35/55 checkpoint emails.</p>
    """

    msg = MIMEMultipart("alternative")
    msg["Subject"] = subject
    msg["From"] = gmail_address
    msg["To"] = gmail_address
    msg.attach(MIMEText(body, "html"))

    try:
        with smtplib.SMTP("smtp.gmail.com", 587) as server:
            server.starttls()
            server.login(gmail_address, gmail_password)
            server.send_message(msg)
        log.info(f"  \u2713 Alert email sent for {ticker} ({status})")
    except Exception as e:
        log.error(f"  Email send failed for {ticker}: {type(e).__name__}: {e}")


class LiveMonitor:
    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run

        all_picks = _load_picks()
        self.picks = [p for p in all_picks if p.get("pattern") in LIVE_TRIGGER_PATTERNS]
        skipped_patterns = [p.get("pattern") for p in all_picks if p.get("pattern") not in LIVE_TRIGGER_PATTERNS]
        if skipped_patterns:
            log.info(f"  {len(skipped_patterns)} picks excluded from live triggering "
                     f"(not Cup & Handle / Swing High Breakout): {skipped_patterns}")
        if not self.picks:
            log.warning("No Cup & Handle / Swing High Breakout picks today — nothing to monitor.")

        regime_code = self.picks[0].get("regime") if self.picks else None
        self.tolerance = REGIME_TOLERANCE.get(regime_code, REGIME_TOLERANCE["C"])
        log.info(f"Regime: '{regime_code}' -> tolerance entry_drift={self.tolerance['max_entry_drift_pct']}% "
                 f"pivot_ext={self.tolerance['max_pivot_extension_pct']}%")

        self.alerted = _load_alerted_state()
        if self.alerted:
            log.info(f"  Resuming: {len(self.alerted)} ticker(s) already alerted today, won't re-alert")

        # One auth check up front (same pattern as data_fetcher.fetch_batch_tv)
        # so fetch_dhan() below actually works instead of silently returning
        # empty because _dhan_status['available'] was never set.
        _check_dhan_auth()

        symbol_map = _dhan_symbol_map()
        self.security_to_ticker = {}   # int security_id -> ticker, for matching incoming ticks
        self.state = {}                # ticker -> live tick state
        instruments = []

        for pick in self.picks:
            ticker = pick["ticker"]
            symbol = _symbol_from_ticker(ticker)
            sec_id = symbol_map.get(symbol)
            if not sec_id:
                log.warning(f"  {ticker}: no Dhan security_id found — skipping (won't be monitored live)")
                continue

            self.security_to_ticker[int(sec_id)] = ticker
            self.state[ticker] = {
                "pick": pick, "open": None, "high": None, "low": None,
                "volume": 0, "avg20v": None,
            }
            instruments.append((MarketFeed.NSE, str(sec_id), MarketFeed.Quote))

        log.info(f"Monitoring {len(instruments)} / {len(self.picks)} eligible picks "
                 f"({len(self.picks) - len(instruments)} skipped — no security_id mapping)")

        # 20-day avg volume baseline for elapsed-time RVOL. Reuses the
        # already-tested Dhan historical fetch (data_fetcher.fetch_dhan),
        # not a new mechanism.
        for ticker, st in self.state.items():
            try:
                df = fetch_dhan(ticker, n_bars=25)
                if not df.empty and len(df) >= 20:
                    st["avg20v"] = float(df["Volume"].iloc[-21:-1].mean())
            except Exception as e:
                log.debug(f"  {ticker}: avg20v fetch failed ({e}), RVOL will show as N/A")

        self.client_id = os.environ.get("DHAN_CLIENT_ID")
        self.access_token = os.environ.get("DHAN_ACCESS_TOKEN")
        if not self.client_id or not self.access_token:
            raise RuntimeError("DHAN_CLIENT_ID/DHAN_ACCESS_TOKEN not set — cannot start live feed.")

        self.dhan_context = DhanContext(self.client_id, self.access_token)
        self.feed = MarketFeed(
            self.dhan_context, instruments, version="v2",
            on_connect=self._on_connect,
            on_message=self._on_message,
            on_close=self._on_close,
            on_error=self._on_error,
        )

    # ── callbacks ──────────────────────────────────────────────────────────

    def _on_connect(self, instance):
        log.info("  Connected to Dhan Live Market Feed.")

    def _on_close(self, instance):
        log.warning("  Dhan feed connection closed.")

    def _on_error(self, instance, error):
        log.error(f"  Dhan feed error: {type(error).__name__}: {error}")

    def process_tick(self, data: dict) -> None:
        """
        Core tick-processing logic, factored out from _on_message so it
        can be exercised directly with synthetic packet dicts in tests
        without needing a real WebSocket connection.
        """
        if not data or data.get("type") != "Quote Data":
            return

        sec_id = data.get("security_id")
        ticker = self.security_to_ticker.get(int(sec_id)) if sec_id is not None else None
        if not ticker:
            return

        st = self.state[ticker]
        try:
            ltp = float(data["LTP"])
            st["volume"] = int(data.get("volume", st["volume"]))
            new_open  = float(data.get("open", 0) or 0)
            new_high  = float(data.get("high", 0) or 0)
            new_low   = float(data.get("low", 0) or 0)
            if new_open:  st["open"] = new_open
            if new_high:  st["high"] = new_high
            if new_low:   st["low"]  = new_low
        except (TypeError, ValueError):
            return

        if ticker in self.alerted:
            return   # already alerted today — don't re-check or re-email

        # Elapsed-time-matched RVOL from live cumulative volume — same
        # philosophy as confirm_picks.py's get_rvol(): compares today's
        # volume-so-far against the SAME elapsed fraction of a normal
        # session, not a fixed calendar window.
        now = datetime.now(IST)
        elapsed_min = (now.hour * 60 + now.minute) - (MARKET_OPEN_TIME.hour * 60 + MARKET_OPEN_TIME.minute)
        if elapsed_min < 5 or not st["avg20v"]:
            rvol = -1.0
        else:
            total_session_min = (MARKET_CLOSE_TIME.hour * 60 + MARKET_CLOSE_TIME.minute) - \
                                 (MARKET_OPEN_TIME.hour * 60 + MARKET_OPEN_TIME.minute)
            expected_frac = min(max(elapsed_min, 0) / total_session_min, 1.0)
            rvol = st["volume"] / (st["avg20v"] * expected_frac) if expected_frac > 0 else -1.0

        opening_range = (st["high"], st["low"], "dhan_live") if st["high"] and st["low"] else (None, None, None)

        classification = classify(
            st["pick"], ltp, rvol, "dhan_live",
            max_entry_drift_pct=self.tolerance["max_entry_drift_pct"],
            max_pivot_extension_pct=self.tolerance["max_pivot_extension_pct"],
            opening_range=opening_range,
        )

        status = classification["status"]
        if status in ("CONFIRMED", "CONFIRMED_LOW_VOL"):
            log.info(f"  \U0001F514 {ticker}: {status} at Rs.{ltp:.2f} (RVOL {rvol:.1f}x)")
            if not self.dry_run:
                send_alert_email(ticker, st["pick"], classification, ltp)
            else:
                log.info(f"  [DRY RUN] Would have sent alert email for {ticker}")
            self.alerted.add(ticker)
            _save_alerted_state(self.alerted)

    def _on_message(self, instance, data):
        self.process_tick(data)

    # ── lifecycle ────────────────────────────────────────────────────────────

    def run(self):
        if not self.state:
            log.warning("Nothing to monitor — exiting.")
            return

        log.info(f"Starting live feed for {len(self.state)} tickers "
                 f"({'DRY RUN — no emails will be sent' if self.dry_run else 'LIVE — alerts will be emailed'})...")
        thread = self.feed.start()

        try:
            while True:
                now = datetime.now(IST).time()
                if now >= MARKET_CLOSE_TIME:
                    log.info("Market close reached — shutting down live monitor.")
                    break
                if not thread.is_alive():
                    log.error("Feed thread died unexpectedly — exiting. Check logs above for the cause.")
                    break
                time.sleep(10)
        except KeyboardInterrupt:
            log.info("Interrupted by user — shutting down.")
        finally:
            self.feed.close_connection()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                         help="Subscribe and classify normally, but never send email — just log what WOULD be sent.")
    args = parser.parse_args()

    monitor = LiveMonitor(dry_run=args.dry_run)
    monitor.run()
