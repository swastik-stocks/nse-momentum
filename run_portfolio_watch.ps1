# NSE Momentum v5.4 — Portfolio Watch daily runner
# ===================================================
# Meant to be called by Windows Task Scheduler each morning. Finds the most
# recently downloaded Axis Direct / Yes Securities holdings files in
# Downloads and runs portfolio_watch.py against them, emailing the result.
#
# LIMITATION (unavoidable, not a bug): this cannot download the broker files
# for you — there's no public API for individual account holdings. It only
# automates the ANALYSIS step. If you haven't downloaded a fresh export
# before this task fires, it will silently analyze a stale file. The
# MaxFileAgeHours check below guards against that — it warns instead of
# running on an old file, rather than silently producing a report from
# yesterday's (or last week's) numbers.

$ErrorActionPreference = "Stop"

# ── Config — adjust these patterns/paths for your setup ─────────────────────
$RepoDir            = "C:\Users\User\Desktop\nse_momentum"
$DownloadsDir        = "C:\Users\hp\Downloads"
$HufFilePattern      = "portfolio_holding_report_*.xlsx"
$MayaFilePath        = Join-Path $RepoDir "maya_holdings.csv"   # manually maintained — Omni has no holdings export
$MaxFileAgeHours     = 20                  # warn if the newest matching HUF file is older than this
$LogFile             = Join-Path $RepoDir "logs\portfolio_watch_$(Get-Date -Format 'yyyyMMdd_HHmmss').log"

function Get-LatestFile($dir, $pattern) {
    Get-ChildItem -Path $dir -Filter $pattern -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending |
        Select-Object -First 1
}

Set-Location $RepoDir
& ".\venv\Scripts\Activate.ps1"

$hufFile  = Get-LatestFile $DownloadsDir $HufFilePattern

$args = @()
$staleWarnings = @()

if ($null -eq $hufFile) {
    $staleWarnings += "No file found matching pattern for HUF — skipped."
} else {
    $ageHours = (New-TimeSpan -Start $hufFile.LastWriteTime -End (Get-Date)).TotalHours
    if ($ageHours -gt $MaxFileAgeHours) {
        $staleWarnings += "HUF file '$($hufFile.Name)' is $([math]::Round($ageHours,1))h old (>$MaxFileAgeHours h threshold) — likely stale, running anyway but VERIFY."
    }
    $args += "--file"
    $args += $hufFile.FullName
    $args += "--account"
    $args += "HUF"
}

if (Test-Path $MayaFilePath) {
    $args += "--file"
    $args += $MayaFilePath
    $args += "--account"
    $args += "Maya"
} else {
    $staleWarnings += "maya_holdings.csv not found at $MayaFilePath — create it manually (see repo notes)."
}

if ($args.Count -eq 0) {
    "No holdings files found in $DownloadsDir matching configured patterns. Nothing to run." |
        Tee-Object -FilePath $LogFile
    exit 1
}

if ($staleWarnings.Count -gt 0) {
    $staleWarnings | ForEach-Object { Write-Output "WARNING: $_" } | Tee-Object -FilePath $LogFile -Append
}

$args += "--email"

Write-Output "Running: python portfolio_watch.py $($args -join ' ')" | Tee-Object -FilePath $LogFile -Append
python portfolio_watch.py @args *>> $LogFile

Write-Output "Done. Log: $LogFile"
