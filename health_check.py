from dotenv import load_dotenv
import os, sqlite3
from pathlib import Path

load_dotenv()

print('=== .env ===')
print('Email:', os.getenv('EMAIL_ADDRESS', 'MISSING'))
print('AppPW:', 'SET' if os.getenv('EMAIL_APP_PASSWORD') else 'MISSING')

db = Path('data/momentum_v4.db')

print()
print('=== Database ===')

if db.exists():
    print(f'Found: {db} ({db.stat().st_size/1024/1024:.1f} MB)')
    conn = sqlite3.connect(db)

    rows = conn.execute(
        'SELECT COUNT(*) FROM price_history'
    ).fetchone()[0]

    tickers = conn.execute(
        'SELECT COUNT(DISTINCT ticker) FROM price_history'
    ).fetchone()[0]

    conn.close()

    print(f'Rows: {rows:,} | Tickers: {tickers}')
else:
    print('NOT FOUND')

print()
print('=== Agents ===')

for f in [
    'agents/pattern_agent.py',
    'agents/market_agent.py',
    'agents/risk_agent.py',
    'agents/rs_agent.py',
    'orchestrator.py',
    'scanner.py'
]:
    status = 'OK' if Path(f).exists() else 'MISSING'
    print(f'  {f}: {status}')

print()
print('=== Recipients ===')

r = Path('recipients.txt')
print(r.read_text().strip() if r.exists() else 'MISSING')