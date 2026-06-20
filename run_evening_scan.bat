@echo off
REM ============================================================
REM  NSE Momentum v4.3 — Evening Scan (4:00 PM)
REM  Place this file in: C:\Users\User\Desktop\nse_momentum\
REM ============================================================

cd /d "C:\Users\User\Desktop\nse_momentum"

REM Activate virtual environment (if using venv)
REM Uncomment the line below if you use a venv:
REM call venv\Scripts\activate.bat

echo [%DATE% %TIME%] Starting NSE Evening Scan... >> logs\evening_scan.log

REM Run the scanner
python scanner.py >> logs\evening_scan.log 2>&1

REM Log exit status
if %ERRORLEVEL% NEQ 0 (
    echo [%DATE% %TIME%] ERROR: Scanner exited with code %ERRORLEVEL% >> logs\evening_scan.log
) else (
    echo [%DATE% %TIME%] Scan completed successfully. >> logs\evening_scan.log
)

exit /b %ERRORLEVEL%
