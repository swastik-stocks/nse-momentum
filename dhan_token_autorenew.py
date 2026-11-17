r"""
Unattended Dhan token renewal -- run by .github/workflows/renew_dhan_token.yml,
triggered by cron-job.org (07:00 / 17:00 IST, every day) with GitHub's own
scheduler as a backup (08:00 / 18:00 IST). No laptop involved.

FLOW (each step exists for a specific failure mode)
  1. Pick the NEWEST valid token from {GitHub secret, Turso token store}.
     Normally identical; if a previous run renewed but one save failed,
     the other copy is the live one -- using the stale copy would fail.
  2. Self-heal: if the two copies differ, write the newest to both.
  3. Skip if > SKIP_IF_HOURS_LEFT_ABOVE hours remain. Makes the backup
     trigger a no-op after a successful primary run, and stops back-to-back
     renewals (each one kills the previous token).
  4. Wait while a Dhan-consuming workflow (daily_scan / confirm) is running:
     RenewToken expires the token those runs were started with, mid-run.
     If still busy after BUSY_WAIT_MAX_MIN, skip -- unless the token is
     about to die anyway (< RENEW_ANYWAY_BELOW_HOURS).
  5. Renew (retries). Dhan expires the old token the moment it answers.
  6. Persist the new token: Turso store FIRST (laptop + self-heal source),
     then the GitHub secret. If BOTH fail, the new token would be lost while
     the old one is already dead -- so it is written, encrypted, to
     token_rescue.enc for the workflow to upload as a 1-day artifact.
  7. Any failure -> Telegram alert stating hours left and what to do.

The token is never printed: every token value is registered with GitHub's
log masker (::add-mask::) before anything else happens -- the repo and its
Actions logs are public.

Exit codes: 0 = renewed or correctly skipped; 1 = needs attention (alerted).
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Callable, Optional

import requests

from dhan_token_check import dhan_token_status

log = logging.getLogger("dhan_token_autorenew")
logging.basicConfig(level=logging.INFO, format="%(message)s")

SKIP_IF_HOURS_LEFT_ABOVE = 20.0
RENEW_ANYWAY_BELOW_HOURS = 3.0
BUSY_WAIT_MAX_MIN = 40
BUSY_POLL_SECONDS = 60
RENEW_RETRY_DELAYS = (0, 20, 60)      # seconds before attempt 1, 2, 3
SAVE_RETRY_DELAYS = (0, 5, 20)
DHAN_WORKFLOWS = ("daily_scan.yml", "confirm.yml")
RESCUE_FILE = Path("token_rescue.enc")
SECRET_NAME = "DHAN_ACCESS_TOKEN"


def _mask(token: Optional[str]) -> None:
    """Register with GitHub's log masker. Harmless no-op output locally."""
    if token and os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::add-mask::{token}", flush=True)


def _alert(msg: str) -> None:
    log.error(msg)
    try:
        from telegram_alert import send_telegram_alert
        send_telegram_alert(f"DHAN TOKEN: {msg}")
    except Exception as e:   # alerting must never mask the real outcome
        log.warning(f"(Telegram alert failed: {e})")


def _retry(fn: Callable[[], bool], delays, sleep=time.sleep) -> bool:
    for d in delays:
        if d:
            sleep(d)
        try:
            if fn():
                return True
        except Exception as e:
            log.warning(f"  attempt failed: {type(e).__name__}: {e}")
    return False


# ── default (real) dependencies; tests inject fakes ──────────────────────────

def _gh_busy_runs(repo: str, token: str) -> list:
    busy = []
    for wf in DHAN_WORKFLOWS:
        for status in ("in_progress", "queued"):
            r = requests.get(
                f"https://api.github.com/repos/{repo}/actions/workflows/{wf}/runs",
                params={"status": status, "per_page": 5},
                headers={"Authorization": f"Bearer {token}",
                         "Accept": "application/vnd.github+json"}, timeout=15)
            r.raise_for_status()
            busy += [f"{wf} #{x['run_number']} ({status})" for x in r.json().get("workflow_runs", [])]
    return busy


class Deps:
    """Everything with a side effect, bundled so tests can replace it."""
    def __init__(self):
        import dhan_token_store
        from github_secrets import push_github_secret
        from renew_dhan_token import renew_dhan_token
        self.store = dhan_token_store
        self._push = push_github_secret
        self._renew = renew_dhan_token
        self.repo = os.environ.get("GITHUB_REPOSITORY", "swastik-stocks/nse-momentum")
        self.admin_pat = os.environ.get("REPO_ADMIN_PAT", "")
        self.actions_token = os.environ.get("ACTIONS_READ_TOKEN", "")
        self.sleep = time.sleep

    def fetch_store(self):
        return self.store.fetch_token()

    def publish_store(self, token: str) -> bool:
        return self.store.publish_token(token, source="github-actions-autorenew")

    def push_secret(self, token: str) -> bool:
        if not self.admin_pat:
            log.warning("  REPO_ADMIN_PAT not set -- cannot update the GitHub secret")
            return False
        return self._push(self.repo, SECRET_NAME, token, self.admin_pat)

    def renew(self, client_id: str, token: str) -> Optional[str]:
        return self._renew(client_id, token)

    def busy_runs(self) -> list:
        if not self.actions_token:
            return []
        return _gh_busy_runs(self.repo, self.actions_token)

    def write_rescue(self, token: str) -> bool:
        RESCUE_FILE.write_text(self.store.encrypt(token), encoding="utf-8")
        return True


# ── main flow ────────────────────────────────────────────────────────────────

def _valid(token: Optional[str]) -> bool:
    st = dhan_token_status(token or "")
    return bool(token) and st["expired"] is False


def run(client_id: str, secret_token: str, deps: Deps, force: bool = False,
        dry_run: bool = False) -> int:
    _mask(secret_token)

    stored = None
    try:
        stored = deps.fetch_store()
    except Exception as e:
        log.warning(f"Token store unreadable ({type(e).__name__}: {e}) -- using the GitHub secret only")
    store_token = stored["token"] if stored else None
    _mask(store_token)

    # 1. newest valid copy wins
    candidates = [t for t in (secret_token, store_token) if _valid(t)]
    if not candidates:
        _alert("No valid token in the GitHub secret or the token store -- it has EXPIRED. "
               "Log in at web.dhan.co (Profile > API > Generate Access Token), then run "
               "`python propagate_dhan_token.py --paste` on the laptop. Automation resumes after that.")
        return 1
    current = max(candidates, key=lambda t: dhan_token_status(t)["expiry"])
    hours = dhan_token_status(current)["hours_remaining"]
    log.info(f"Current token: {hours:.1f}h remaining "
             f"(source: {'GitHub secret' if current == secret_token else 'token store'}).")

    # 2. self-heal a previous partial save
    if not dry_run:
        if store_token != current:
            log.info("Token store is behind -- updating it.")
            _retry(lambda: deps.publish_store(current), SAVE_RETRY_DELAYS, deps.sleep)
        if secret_token != current:
            log.info("GitHub secret is behind -- updating it.")
            _retry(lambda: deps.push_secret(current), SAVE_RETRY_DELAYS, deps.sleep)

    # 3. fresh enough -> nothing to do
    if hours > SKIP_IF_HOURS_LEFT_ABOVE and not force:
        log.info(f"More than {SKIP_IF_HOURS_LEFT_ABOVE:.0f}h left -- no renewal needed this run.")
        return 0

    # 4. never renew under a running Dhan consumer
    waited = 0
    while True:
        try:
            busy = deps.busy_runs()
        except Exception as e:
            log.warning(f"Could not check running workflows ({e}) -- proceeding")
            busy = []
        if not busy:
            break
        if waited >= BUSY_WAIT_MAX_MIN:
            if hours > RENEW_ANYWAY_BELOW_HOURS:
                log.warning(f"Still busy after {waited} min ({', '.join(busy)}); token has "
                            f"{hours:.1f}h left -- skipping, the next trigger will renew.")
                return 0
            log.warning(f"Still busy but token has only {hours:.1f}h left -- renewing anyway.")
            break
        log.info(f"Waiting: Dhan-consuming run(s) in progress: {', '.join(busy)}")
        deps.sleep(BUSY_POLL_SECONDS)
        waited += BUSY_POLL_SECONDS // 60

    if dry_run:
        log.info("[dry-run] would call Dhan RenewToken now and save the result to the store + secret.")
        return 0

    # 5. renew
    new_token = None
    def _attempt():
        nonlocal new_token
        new_token = deps.renew(client_id, current)
        return bool(new_token)
    if not _retry(_attempt, RENEW_RETRY_DELAYS, deps.sleep):
        _alert(f"RenewToken FAILED after {len(RENEW_RETRY_DELAYS)} attempts. Current token still has "
               f"{hours:.1f}h left; the next scheduled run will retry. If this repeats, run "
               f"`python propagate_dhan_token.py --renew` on the laptop before it expires.")
        return 1
    _mask(new_token)
    new_status = dhan_token_status(new_token)
    log.info(f"Renewed. New token valid until {new_status['expiry']} ({new_status['hours_remaining']:.1f}h).")

    # 6. persist -- store first, then secret
    store_ok = _retry(lambda: deps.publish_store(new_token), SAVE_RETRY_DELAYS, deps.sleep)
    secret_ok = _retry(lambda: deps.push_secret(new_token), SAVE_RETRY_DELAYS, deps.sleep)
    log.info(f"  token store: {'updated' if store_ok else 'FAILED'}; "
             f"GitHub secret: {'updated' if secret_ok else 'FAILED'}")

    if store_ok and secret_ok:
        return 0
    if not store_ok and not secret_ok:
        try:
            deps.write_rescue(new_token)
            where = "saved ENCRYPTED as the run's `token_rescue` artifact (1 day)"
        except Exception as e:
            where = f"could not even be saved encrypted ({e})"
        _alert("CRITICAL: renewed, but BOTH saves failed -- the old token is already dead. "
               f"The new token was {where}. Fastest fix: web.dhan.co login + "
               "`python propagate_dhan_token.py --paste`.")
        return 1
    _alert(f"Renewed OK but the {'token store' if not store_ok else 'GitHub secret'} save failed. "
           "The next run self-heals it; laptop: `python propagate_dhan_token.py --pull` "
           "works only once the store is updated.")
    return 1


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--force", action="store_true", help="Renew even if the token is still fresh")
    ap.add_argument("--dry-run", action="store_true", help="Decide and report, but never call Dhan or write")
    args = ap.parse_args(argv)
    client_id = os.environ.get("DHAN_CLIENT_ID", "")
    if not client_id:
        _alert("DHAN_CLIENT_ID secret is missing -- cannot renew.")
        return 1
    return run(client_id, os.environ.get("DHAN_ACCESS_TOKEN", ""), Deps(),
               force=args.force, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
