# Registers the Windows scheduled task that keeps this laptop's .env copies of
# DHAN_ACCESS_TOKEN in step with the cloud auto-renewal (renew_dhan_token.yml).
#
# Why it's needed: Dhan's RenewToken expires the previous token, so every
# cloud renewal (07:00 / 17:00 IST) kills the token in F:\ and E:\ .env.
# This task runs `propagate_dhan_token.py --pull` (copies the newer token from
# the encrypted Turso store; no Dhan call, never mints a token):
#   - 15 minutes after each cloud renewal slot (07:15, 08:15, 17:15, 18:15)
#   - every hour, and at logon
#   - StartWhenAvailable: if the laptop was off/asleep, it runs on wake.
# It only matters while the laptop is on -- which is exactly when local
# scripts need the token. Re-running this script just replaces the task.
#
# Usage (normal PowerShell, no admin needed):
#   powershell -ExecutionPolicy Bypass -File F:\nse_momentum\install_token_pull_task.ps1
# Remove:
#   Unregister-ScheduledTask -TaskName "NSE Momentum - Dhan token pull" -Confirm:$false

$ErrorActionPreference = "Stop"
$TaskName = "NSE Momentum - Dhan token pull"
$Repo     = "F:\nse_momentum"
$Python   = Join-Path $Repo "venv\Scripts\pythonw.exe"   # pythonw: no console window
$LogFile  = Join-Path $Repo "logs\token_pull.log"

if (-not (Test-Path $Python)) { throw "Project venv not found at $Python" }
New-Item -ItemType Directory -Force (Join-Path $Repo "logs") | Out-Null

$action = New-ScheduledTaskAction -Execute $Python `
    -Argument "propagate_dhan_token.py --pull --log-file `"$LogFile`"" `
    -WorkingDirectory $Repo

$triggers = @(
    (New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME),
    (New-ScheduledTaskTrigger -Once -At (Get-Date).Date -RepetitionInterval (New-TimeSpan -Hours 1)),
    (New-ScheduledTaskTrigger -Daily -At "07:15"),
    (New-ScheduledTaskTrigger -Daily -At "08:15"),
    (New-ScheduledTaskTrigger -Daily -At "17:15"),
    (New-ScheduledTaskTrigger -Daily -At "18:15")
)

$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -AllowStartIfOnBatteries `
    -DontStopIfGoingOnBatteries -ExecutionTimeLimit (New-TimeSpan -Minutes 5) `
    -MultipleInstances IgnoreNew

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $triggers `
    -Settings $settings -Description "Pulls the cloud-renewed Dhan token into F:\ and E:\ .env (see docs\DHAN_TOKEN_AUTORENEW.md)" `
    -Force | Out-Null

Write-Host "Registered '$TaskName'. Log: $LogFile"
Write-Host "Test it now:  Start-ScheduledTask -TaskName '$TaskName'; Start-Sleep 15; Get-Content '$LogFile' -Tail 3"
