@echo off
title NSE Momentum Scanner v4.3
cd /d C:\Users\User\Desktop\nse_momentum
call venv\Scripts\activate.bat
echo.
echo ============================================================
echo   NSE MOMENTUM SCANNER v4.3
echo   Starting scan... (takes 60-90 seconds)
echo ============================================================
echo.
venv\Scripts\python.exe scanner.py
echo.
echo ============================================================
echo   Scan complete. Check your email inbox.
echo ============================================================
echo.
pause
