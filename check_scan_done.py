"""
P0 Step 1: backstop gate for daily_scan.yml.

The GitHub cron trigger is now a backstop that fires ~1 hour after the
primary cron-job.org trigger (repository_dispatch 'evening-scan'). If the
primary trigger already ran successfully today, running the full scan
again would double the Dhan API budget and re-send the email. This script
checks scan_metadata for a row matching today's date and writes
should_run to $GITHUB_OUTPUT accordingly.

Uses UTC date throughout, matching turso_sync.publish_scan_metadata()'s
date.today() — GitHub Actions runners are UTC, and both the 19:30 IST
primary run and the 20:30 IST backstop fall on the same UTC calendar day,
so no IST conversion is needed here.
"""
import datetime as dt
import os
import sys

import libsql_client

TURSO_URL = os.environ.get("TURSO_DATABASE_URL")
TURSO_TOKEN = os.environ.get("TURSO_AUTH_TOKEN")


def _set_output(should_run: bool, reason: str):
    print(f"should_run={should_run} ({reason})")
    gh_output = os.environ.get("GITHUB_OUTPUT")
    if gh_output:
        with open(gh_output, "a") as f:
            f.write(f"should_run={'true' if should_run else 'false'}\n")


def main() -> int:
    if not TURSO_URL or not TURSO_TOKEN:
        _set_output(True, "Turso credentials missing — fail open, run the scan")
        return 0

    today = dt.date.today().isoformat()
    url = TURSO_URL.replace("libsql://", "https://")
    client = libsql_client.create_client_sync(url, auth_token=TURSO_TOKEN)
    try:
        rs = client.execute(
            "SELECT scan_date FROM scan_metadata WHERE scan_date = ?", [today]
        )
        if rs.rows:
            _set_output(False, f"scan_metadata already has a row for {today}")
        else:
            _set_output(True, f"no scan_metadata row for {today} yet")
        return 0
    except Exception as e:
        _set_output(True, f"query failed ({e}) — fail open, run the scan")
        return 0
    finally:
        client.close()


if __name__ == "__main__":
    sys.exit(main())
