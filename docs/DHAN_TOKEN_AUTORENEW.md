# Dhan token auto-renewal: setup and runbook

Dhan access tokens last 24 hours. Since 17 Nov 2026 they are renewed in the cloud every 12 hours, with no laptop involved. This page covers the one-time cron-job.org setup, what each alert means, and how to recover.

## How it works

```
cron-job.org  07:00 & 17:00 IST, every day   ─┐
GitHub cron   08:00 & 18:00 IST (backup)      ─┴─►  renew_dhan_token.yml → dhan_token_autorenew.py
                                                       1. take the newest token (GitHub secret vs Turso store)
                                                       2. skip if > 20h left   (so the backup run is normally a no-op)
                                                       3. wait while daily_scan / confirm is running
                                                       4. Dhan RenewToken (3 tries)
                                                       5. save: Turso store (encrypted) → GitHub secret
                                                       6. Telegram alert on any failure
Laptop (only when on): hourly task "NSE Momentum - Dhan token pull"
                       → propagate_dhan_token.py --pull → F:\ and E:\ .env
```

**Why the laptop still needs a pull:** RenewToken **expires the previous token**, so each cloud renewal kills the copy in your `.env` files. The hourly task copies the new one down from the encrypted Turso store. It also runs 15 minutes after each renewal slot, at logon, and on wake.

**Why cron-job.org rather than GitHub's scheduler:** GitHub cron can start hours late (on 16 Nov a 20:30 IST cron started at 05:06 IST). cron-job.org fires on time; GitHub's schedule is the safety net.

## One-time setup: cron-job.org (about 5 minutes)

You need a GitHub token that is allowed to start workflows.

- **Option A (simplest):** reuse the token already in your cron-job.org jobs for `confirm.yml`. Copy the `Authorization` header value from one of them.
- **Option B (least privilege):** create a new token at <https://github.com/settings/personal-access-tokens/new>:
  - Token name: `cron-job.org dhan renewal`
  - Expiration: 1 year. Put a reminder in your calendar, because an expired token silently stops the trigger. The GitHub backup keeps renewal working, so nothing breaks immediately.
  - Repository access: *Only select repositories* → `swastik-stocks/nse-momentum`
  - Permissions → Repository → **Actions: Read and write**. Nothing else.
  - Generate it and copy the value (`github_pat_...`).

### Create the job

1. Log in at <https://console.cron-job.org> → **Cronjobs** → **Create cronjob**.
2. **COMMON tab**
   - **Title:** `NSE Momentum - Dhan token renewal`
   - **URL:**
     ```
     https://api.github.com/repos/swastik-stocks/nse-momentum/actions/workflows/renew_dhan_token.yml/dispatches
     ```
   - **Enable job:** on
   - **Save responses in job history:** on (useful for debugging)
   - **Execution schedule:** choose **Custom**, then set:
     - Days of month: *Every day of the month*
     - Days of week: *Every day of the week* (**include Saturday and Sunday**)
     - Months: *Every month*
     - Hours: tick **7** and **17** only
     - Minutes: tick **0** only
3. **ADVANCED tab**
   - **Time zone:** `Asia/Kolkata`. This is per job; don't rely on the account default.
   - **Request method:** `POST`
   - **Headers** (add each as a key/value row):

     | Key | Value |
     |---|---|
     | `Accept` | `application/vnd.github+json` |
     | `Authorization` | `Bearer github_pat_...` (your token from above) |
     | `X-GitHub-Api-Version` | `2022-11-28` |
     | `Content-Type` | `application/json` |
     | `User-Agent` | `cron-job.org` |

   - **Request body:**
     ```json
     {"ref":"main"}
     ```
   - **Timeout:** 30 seconds
4. **NOTIFICATIONS tab:** tick **Notify me when execution of the cronjob fails**, and **when the cronjob will be disabled because of too many failures**. cron-job.org disables a job automatically after repeated failures, and you want to know if that happens.
5. Click **TEST RUN**. Expect HTTP status **204 No Content**, which is GitHub's success response. Then **Save** (or **Create**).
6. Check it at <https://github.com/swastik-stocks/nse-momentum/actions/workflows/renew_dhan_token.yml>. A new run should appear within a few seconds.
   - If the token has more than 20 hours left, the run logs `no renewal needed this run`. That's correct.
   - To force a real renewal for testing, use **Run workflow** on that page with *force = true*. Avoid 19:30 to 21:00 IST; the run would wait for the evening scan anyway.

| Test-run result | Meaning |
|---|---|
| 204 | Working |
| 401 | Wrong or expired token, or missing `Bearer ` prefix |
| 403 or 404 | The token lacks *Actions: write* on this repo, or the URL has a typo |
| 422 | The body isn't exactly `{"ref":"main"}`, or the workflow file isn't on `main` |

## One-time setup: already done on this laptop (17 Nov 2026)

- GitHub secret `DHAN_TOKEN_STORE_KEY` holds the store encryption key, and the same key is in the Windows user environment variable of the same name.
- The Turso token store is seeded.
- The scheduled task is installed by `install_token_pull_task.ps1`. Re-run that script to reinstall it, for example on a new laptop.

On a **new machine**, copy the `DHAN_TOKEN_STORE_KEY` user environment variable from the old one and run the install script. If the key is lost, generate a new one with `python -c "import dhan_token_store as s; print(s.generate_key())"`, set it in both places (the GitHub secret and the user environment variable), then run `python propagate_dhan_token.py --paste` once.

## What the Telegram alerts mean

| Alert starts with | What happened | What to do |
|---|---|---|
| `No valid token ... EXPIRED` | The chain broke: two or more renewals were missed. | Log in at web.dhan.co → Profile → API → Generate Access Token, then run `python propagate_dhan_token.py --paste`. Automation resumes from there. |
| `RenewToken FAILED` | Dhan refused or timed out. The old token still works for the stated hours. | Nothing, if the next slot is before expiry. If the alert repeats, run `python propagate_dhan_token.py --renew` before expiry. |
| `Renewed OK but the ... save failed` | The new token is in one of the two stores. | Nothing. The next run copies it into the other store. |
| `CRITICAL: ... BOTH saves failed` | The new token exists only as an encrypted artifact, and the old one is already dead. | Open the failed run on GitHub → *Artifacts* → download `token_rescue`, unzip it, then run `python propagate_dhan_token.py --rescue token_rescue.enc`. If more than a day has passed, use `--paste` instead. |

## Laptop commands

```
python propagate_dhan_token.py --pull     # get the current token now (the hourly task does this)
python propagate_dhan_token.py --paste    # after a lapse: restart the chain with a fresh web.dhan.co token
python propagate_dhan_token.py --renew    # manual renewal (pulls first; normally unnecessary)
python propagate_dhan_token.py --sync     # re-push the local token everywhere (no Dhan call)
```

Every mode also writes the encrypted Turso store. A newer token is never overwritten by an older one.

Task log: `F:\nse_momentum\logs\token_pull.log`.
To remove the task: `Unregister-ScheduledTask -TaskName "NSE Momentum - Dhan token pull" -Confirm:$false`

## Known limits

- **Dhan cannot revive an expired token.** If renewals are missed for more than 24 hours, one manual login is unavoidable.
- **Renewing expires the old token immediately.** A run started with the old token fails if a renewal lands mid-run. The schedule avoids this, and the script waits while `daily_scan` or `confirm` runs. Local scripts started in the up-to-15-minute gap before the pull task runs can hit it; run `--pull` first if you start something at about 07:00 or 17:00.
