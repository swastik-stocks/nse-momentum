@echo off
REM ============================================================
REM  NSE Momentum v4.3 — Dhan Token Refresh Helper
REM  Run this AFTER you generate a new token at web.dhan.co
REM  It updates the .env file and validates the connection.
REM ============================================================

cd /d "C:\Users\User\Desktop\nse_momentum"

echo.
echo ============================================
echo  DHAN TOKEN REFRESH — NSE Momentum v4.3
echo ============================================
echo.
echo Step 1: Go to https://web.dhan.co
echo         Profile ^(top-right^) ^> API ^> Generate Access Token
echo         Copy the NEW Access Token
echo.
set /p NEW_TOKEN=Paste your new Dhan Access Token here and press Enter: 

if "%NEW_TOKEN%"=="" (
    echo ERROR: No token entered. Exiting.
    pause
    exit /b 1
)

REM Use Python to safely update only the DHAN_ACCESS_TOKEN line in .env
python -c "
import re, sys

token = r'%NEW_TOKEN%'
env_path = '.env'

with open(env_path, 'r', encoding='utf-8') as f:
    content = f.read()

if 'DHAN_ACCESS_TOKEN' in content:
    content = re.sub(r'DHAN_ACCESS_TOKEN=.*', f'DHAN_ACCESS_TOKEN={token}', content)
else:
    content += f'\nDHAN_ACCESS_TOKEN={token}'

with open(env_path, 'w', encoding='utf-8') as f:
    f.write(content)

print('Token updated in .env successfully.')
"

if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Failed to update .env
    pause
    exit /b 1
)

echo.
echo Validating Dhan connection...
echo.

python -c "
import os
from dotenv import load_dotenv
load_dotenv(override=True)
import urllib.request, json

client_id    = os.getenv('DHAN_CLIENT_ID', '')
access_token = os.getenv('DHAN_ACCESS_TOKEN', '')

if not client_id or not access_token:
    print('ERROR: Credentials missing from .env')
    exit(1)

payload = json.dumps({'NSE_EQ': ['1333']}).encode()
req = urllib.request.Request(
    'https://api.dhan.co/v2/marketfeed/quote',
    data=payload,
    headers={
        'Content-Type': 'application/json',
        'access-token': access_token,
        'client-id': client_id,
    },
    method='POST'
)
try:
    with urllib.request.urlopen(req, timeout=10) as r:
        data = json.loads(r.read())
    nse = data.get('data', {}).get('NSE_EQ', {})
    if nse:
        ltp = list(nse.values())[0].get('last_price', '?')
        print(f'Dhan connection: OK  |  HDFC Bank LTP: Rs.{ltp}')
    else:
        print(f'Connected but no data returned: {data}')
except Exception as e:
    print(f'Connection FAILED: {e}')
    exit(1)
"

echo.
if %ERRORLEVEL% EQU 0 (
    echo Token refresh complete. Evening scan will use the new token.
) else (
    echo WARNING: Token updated but validation failed. Check the token and retry.
)

echo.
pause
