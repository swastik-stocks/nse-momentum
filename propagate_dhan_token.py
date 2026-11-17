r"""
NSE Momentum -- single entry point that keeps DHAN_ACCESS_TOKEN in sync
across all 3 places it's currently duplicated by hand: F:\nse_momentum\.env
(prod), E:\nse_momentum_dev\.env (dev worktree), and the GitHub Actions
DHAN_ACCESS_TOKEN secret (daily_scan.yml's live evening-scan trigger).

WHY THIS EXISTS
    renew_dhan_token.py and update_dhan_token.py both already know how to
    write ONE .env file, and renew_dhan_token.py already knows how to push
    to the GitHub secret via github_secrets.py -- but nothing calls either
    against all 3 destinations in a single run, so the token has been
    hand-copied into 3 places every trading day. This is that missing
    wrapper, not new token-handling logic.

TWO MODES
    1. --renew (default, weekday case): the current PROD_ENV token is
       still ACTIVE. Calls Dhan's /v2/RenewToken exactly once -- never
       call it twice for one token, that would mint two different
       replacement tokens and defeat the point of a single shared value --
       then propagates the ONE resulting token to all 3 destinations.
    2. --paste (post-lapse case, e.g. after a weekend): prompts once for a
       manually-generated token from web.dhan.co (unavoidable -- Dhan
       requires an interactive login to mint a token from scratch, no API
       shortcut exists), then propagates that ONE value to all 3
       destinations instead of pasting it 3 times by hand.

GitHub secret push needs GITHUB_TOKEN (a PAT with repo-secrets scope --
the same REPO_ADMIN_PAT referenced in renew_dhan_token.yml) and
GITHUB_REPO ("owner/name") as environment variables when this script runs
locally. Skipped with a warning (not a hard failure) if either is absent,
since the other 2 destinations are still worth updating on their own.

[2026-11-17] CLOUD AUTO-RENEWAL
    Renewal now runs unattended in GitHub Actions (renew_dhan_token.yml /
    dhan_token_autorenew.py, see docs/DHAN_TOKEN_AUTORENEW.md). Because
    RenewToken expires the previous token, the laptop's .env copies go stale
    on every cloud renewal -- the canonical copy is the encrypted Turso
    token store (dhan_token_store.py), a 4th destination of propagate().
    --pull copies it down to both .env files (an hourly Windows task does
    this automatically: install_token_pull_task.ps1).

Usage (run locally, from either E: or F: -- paths below are absolute so it
doesn't matter which checkout invokes it):
    python propagate_dhan_token.py --pull      # fetch the cloud-renewed token into both .env files
    python propagate_dhan_token.py --paste     # after a lapse: one web.dhan.co login restarts the chain
    python propagate_dhan_token.py --renew     # manual renewal (normally the cloud does this)
    python propagate_dhan_token.py --sync      # re-push the current prod token if a destination failed
    python propagate_dhan_token.py --rescue token_rescue.enc   # recover from the CRITICAL alert's artifact
    python propagate_dhan_token.py --renew --dry-run   # shows what would happen, no writes/calls
"""
import argparse
import logging
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from dhan_token_check import dhan_token_status
from renew_dhan_token import renew_dhan_token

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")

PROD_ENV = Path(r"F:\nse_momentum\.env")
DEV_ENV = Path(r"E:\nse_momentum_dev\.env")
GITHUB_REPO_DEFAULT = "swastik-stocks/nse-momentum"


def _write_token(env_path: Path, new_token: str, dry_run: bool = False) -> bool:
    if not env_path.exists():
        log.warning(f"  {env_path} does not exist -- skipped.")
        return False
    content = env_path.read_text(encoding="utf-8")
    if "DHAN_ACCESS_TOKEN" not in content:
        log.warning(f"  {env_path} has no DHAN_ACCESS_TOKEN line -- skipped.")
        return False
    new_content = re.sub(r"DHAN_ACCESS_TOKEN=.*", f"DHAN_ACCESS_TOKEN={new_token}", content)
    if dry_run:
        log.info(f"  [dry-run] would update {env_path}")
        return True
    env_path.write_text(new_content, encoding="utf-8")
    log.info(f"  Updated {env_path}")
    return True


def _push_github(new_token: str, dry_run: bool = False) -> bool:
    gh_token = os.environ.get("GITHUB_TOKEN", "")
    gh_repo = os.environ.get("GITHUB_REPO", GITHUB_REPO_DEFAULT)
    if not gh_token:
        log.warning("  GITHUB_TOKEN not set in environment -- GitHub secret push skipped "
                     "(prod/dev .env were still updated above).")
        return False
    if dry_run:
        log.info(f"  [dry-run] would push DHAN_ACCESS_TOKEN to GitHub secret on {gh_repo}")
        return True
    from github_secrets import push_github_secret
    ok = push_github_secret(gh_repo, "DHAN_ACCESS_TOKEN", new_token, gh_token)
    log.info(f"  GitHub secret updated on {gh_repo}." if ok else "  GitHub secret push FAILED.")
    return ok


def _publish_store(new_token: str, dry_run: bool = False) -> bool:
    """4th destination [2026-11-17]: the encrypted Turso token store the
    cloud renewal and --pull read from. Newer-wins inside publish_token(),
    so propagating an older token can never clobber a fresher cloud one."""
    if dry_run:
        log.info("  [dry-run] would publish to the encrypted Turso token store")
        return True
    try:
        from dhan_token_store import publish_token
        ok = publish_token(new_token, source="laptop-propagate")
    except Exception as e:
        log.warning(f"  Token store publish FAILED ({type(e).__name__}: {e}) -- the cloud "
                    f"renewal self-heals it from the GitHub secret on its next run.")
        return False
    log.info("  Token store updated." if ok else "  Token store kept its NEWER token (see warning above).")
    return ok


def propagate(new_token: str, dry_run: bool = False) -> None:
    log.info("Propagating token to all 4 destinations:")
    _write_token(PROD_ENV, new_token, dry_run)
    _write_token(DEV_ENV, new_token, dry_run)
    _publish_store(new_token, dry_run)
    _push_github(new_token, dry_run)
    log.info("Done." if not dry_run else "Dry run complete -- nothing was actually written.")


def _env_token(env_path: Path) -> str:
    if not env_path.exists():
        return ""
    m = re.search(r"^DHAN_ACCESS_TOKEN=(.*)$", env_path.read_text(encoding="utf-8"), re.M)
    return m.group(1).strip() if m else ""


def mode_pull(dry_run: bool = False) -> int:
    """Copy the cloud-renewed token from the Turso store into both .env
    files -- only if it expires LATER than what each file already has.
    Safe to run as often as you like (the hourly Windows task does)."""
    load_dotenv(PROD_ENV)   # TURSO_* creds; must precede the turso_sync import
    try:
        from dhan_token_store import fetch_token
        stored = fetch_token()
    except Exception as e:
        log.error(f"Could not read the token store: {type(e).__name__}: {e}")
        return 1
    if not stored:
        log.error("Token store is empty -- run --renew or --paste once to seed it.")
        return 1
    st = dhan_token_status(stored["token"])
    if st["expired"]:
        log.error(f"Stored token EXPIRED at {stored['expiry']} -- the cloud chain is broken. "
                  f"Run --paste after a web.dhan.co login.")
        return 1
    changed = 0
    for env_path in (PROD_ENV, DEV_ENV):
        local_exp = dhan_token_status(_env_token(env_path))["expiry"]
        if local_exp is not None and local_exp >= stored["expiry"]:
            continue
        if _write_token(env_path, stored["token"], dry_run):
            changed += 1
    log.info(f"Pulled: token valid until {stored['expiry']} ({st['hours_remaining']:.1f}h left, "
             f"renewed by {stored['source']}). {changed} .env file(s) updated.")
    return 0


def mode_rescue(path: str, dry_run: bool = False) -> int:
    """Recover the token from the encrypted `token_rescue` artifact the
    cloud run uploads when BOTH of its saves failed (CRITICAL alert)."""
    load_dotenv(PROD_ENV)
    from dhan_token_store import decrypt
    token = decrypt(Path(path).read_text(encoding="utf-8").strip())
    st = dhan_token_status(token)
    if st["expired"] is not False:
        log.error("Rescued token is expired or unreadable -- use --paste instead.")
        return 1
    log.info(f"Rescued token valid until {st['expiry']} ({st['hours_remaining']:.1f}h).")
    propagate(token, dry_run)
    return 0


def _pull_quietly(dry_run: bool) -> None:
    if dry_run:
        return
    try:
        mode_pull()
    except Exception as e:
        log.warning(f"  (token store not checked: {e})")


def mode_renew(dry_run: bool = False) -> int:
    # [2026-11-17] The cloud renewal kills the token in the local .env; start
    # from the newest copy in the token store, not a possibly-dead local one.
    _pull_quietly(dry_run)
    load_dotenv(PROD_ENV, override=True)
    client_id = os.environ.get("DHAN_CLIENT_ID", "")
    access_token = os.environ.get("DHAN_ACCESS_TOKEN", "")
    if not client_id or not access_token:
        log.error(f"DHAN_CLIENT_ID / DHAN_ACCESS_TOKEN not found in {PROD_ENV}.")
        return 1

    status = dhan_token_status(access_token)
    if status["expired"]:
        log.error("Current prod token is already EXPIRED -- RenewToken cannot resurrect it. "
                   "Run with --paste instead after generating a fresh token at web.dhan.co "
                   "(Profile > API > Generate Access Token).")
        return 1
    log.info(f"Current token has {status['hours_remaining']:.1f}h remaining.")

    if dry_run:
        log.info("[dry-run] would call Dhan's /v2/RenewToken now.")
        propagate("<renewed-token-placeholder>", dry_run=True)
        return 0

    new_token = renew_dhan_token(client_id, access_token)
    if not new_token:
        return 1
    new_status = dhan_token_status(new_token)
    log.info(f"Renewed. New token valid until {new_status['expiry']} "
             f"({new_status['hours_remaining']:.1f}h).")
    propagate(new_token)
    return 0


def mode_sync(dry_run: bool = False) -> int:
    """Re-propagate the token already in PROD_ENV, without calling Dhan.

    Why: --renew writes the prod .env *before* the GitHub push, so a
    transient network failure on the push leaves the token updated locally
    but stale in CI. Re-running --renew to fix that needlessly mints another
    token and fails outright if api.dhan.co is unreachable. --sync is
    idempotent and safe to retry until every destination succeeds."""
    # [2026-11-17] The cloud renewal kills the token in the local .env; start
    # from the newest copy in the token store, not a possibly-dead local one.
    _pull_quietly(dry_run)
    load_dotenv(PROD_ENV, override=True)
    token = os.environ.get("DHAN_ACCESS_TOKEN", "")
    if not token:
        log.error(f"DHAN_ACCESS_TOKEN not found in {PROD_ENV}.")
        return 1
    status = dhan_token_status(token)
    if status["expired"]:
        log.error("Prod token is EXPIRED -- nothing worth syncing. Use --paste with a fresh token.")
        return 1
    log.info(f"Syncing prod token (valid until {status['expiry']}, {status['hours_remaining']:.1f}h left).")
    propagate(token, dry_run)
    return 0


def mode_paste(dry_run: bool = False) -> int:
    print()
    print("Step 1: Go to https://web.dhan.co")
    print("        Profile (top-right) > API > Generate Access Token")
    print("        Copy the NEW Access Token")
    print()
    if dry_run:
        log.info("[dry-run] would prompt for a pasted token here, then propagate it to all 3 destinations.")
        propagate("<pasted-token-placeholder>", dry_run=True)
        return 0
    new_token = input("Paste your new Dhan Access Token here and press Enter: ").strip()
    if not new_token:
        log.error("No token entered. Exiting.")
        return 1
    propagate(new_token)
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--renew", action="store_true",
                       help="Weekday case: current prod token still active, renew via API (no login needed)")
    mode.add_argument("--paste", action="store_true",
                       help="Post-lapse case: prompt once for a manually-generated token, propagate it everywhere")
    mode.add_argument("--sync", action="store_true",
                       help="Retry case: push the token already in the prod .env to dev + store + GitHub (no Dhan call)")
    mode.add_argument("--pull", action="store_true",
                       help="Fetch the cloud-renewed token from the Turso store into both .env files")
    mode.add_argument("--rescue", metavar="FILE",
                       help="Recover a token from the encrypted token_rescue artifact, then propagate it")
    parser.add_argument("--dry-run", action="store_true", help="Show what would happen without writing/calling anything")
    parser.add_argument("--log-file", help="Also append log lines here (used by the windowless scheduled --pull task)")
    args = parser.parse_args()
    if args.log_file:
        fh = logging.FileHandler(args.log_file, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s %(message)s"))
        logging.getLogger().addHandler(fh)

    if args.renew:
        sys.exit(mode_renew(args.dry_run))
    if args.pull:
        sys.exit(mode_pull(args.dry_run))
    if args.rescue:
        sys.exit(mode_rescue(args.rescue, args.dry_run))
    sys.exit(mode_sync(args.dry_run) if args.sync else mode_paste(args.dry_run))
