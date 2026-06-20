# NSE Momentum Scanner v4.3
**504 stocks · 12 agents · 19 patterns · All free data · Evidence-based**

Institutional-grade momentum stock discovery for NSE. Scans 504 stocks daily,
emails a 3-section HTML report with trade cards, watchlist table, and market
intelligence. No paid data required.

---

## Quick Start (Windows)

```powershell
# 1. Navigate to Desktop
cd C:\Users\User\Desktop

# 2. Create folder and unzip here
mkdir nse_momentum
# (copy all files into nse_momentum\)

# 3. Set up Python environment
cd nse_momentum
python -m venv venv
venv\Scripts\activate

# 4. Install dependencies
pip install -r requirements.txt

# 5. First-time setup (creates DB, seeds weights, optional price download)
python setup_v40.py --fast          # tables + weights only (fast)
# OR
python setup_v40.py                  # full setup (downloads 2yr prices, ~40 min)

# 6. Add your Gmail credentials
# Edit .env (copy from .env.template):
#   GMAIL_ADDRESS=you@gmail.com
#   GMAIL_APP_PASSWORD=your_16_char_app_password

# 7. Run the scanner
python scanner.py
# OR double-click run_scanner.bat
```

---

## Daily Workflow

| Time | Action |
|------|--------|
| 3:30 PM | Market closes |
| ~3:45 PM | Double-click `run_scanner.bat` on Desktop |
| ~3:47 PM | Email arrives with today's setups |

**Log every trade entry:**
```python
from trade_logger import log_entry
log_entry('RELIANCE.NS', 'Reliance Industries', 'Energy', 'LARGE',
          'VCP', 2540.0, 2450.0, 2720.0, 2900.0, 2.0, 82, 74.0)
```

**Log every exit:**
```python
from trade_logger import log_exit
log_exit(trade_id=1, exit_price=2710.0, exit_type='Target1')
```

**See pattern leaderboard:**
```python
from trade_logger import print_stats
print_stats()
```

**Deep-dive on one stock:**
```powershell
venv\Scripts\python.exe single_stock.py RELIANCE
```

---

## File Structure

```
nse_momentum\
│
├── scanner.py                  ← Run this (entry point)
├── orchestrator.py             ← Wires all 12 agents
├── nse_universe.py             ← 504 stocks with sector + universe
├── data_fetcher.py             ← Yahoo Finance + Bhavcopy
├── emailer.py                  ← 3-section HTML email
├── trade_logger.py             ← Trade log + dynamic weights (Learning Agent)
├── portfolio_engine.py         ← v4.0 position sizing + portfolio controls
├── setup_v40.py                ← First-time setup (run once)
├── update_weights.py           ← Manual weight recompute
├── single_stock.py             ← Deep-dive analysis
├── health_check.py             ← Verify all imports
├── run_scanner.bat             ← Double-click launcher
├── requirements.txt
├── recipients.txt              ← Email distribution list (one address per line)
├── .env                        ← Gmail credentials (create from .env.template)
│
├── agents\
│   ├── pattern_agent.py        ← 19 patterns + EMA/MACD/RSI
│   ├── rs_agent.py             ← 4/12/26-week RS percentile
│   ├── volume_agent.py         ← RVOL + delivery %
│   ├── market_agent.py         ← 5-regime A-E + breadth
│   ├── market_breadth_agent.py ← A/D + 50EMA% + 52wHL
│   ├── sector_agent.py         ← 13-sector rotation
│   ├── risk_agent.py           ← Entry/SL/T1/T2/RRR gate
│   ├── liquidity_agent.py      ← ADT + participation + mcap
│   ├── conviction_agent.py     ← Score aggregator
│   ├── fundamental_proxy_agent.py  ← Agents 9A+9B+9C
│   └── institutional_proxy_agent.py ← Agents 10A-10E
│
├── collectors\
│   └── price_collector.py      ← Builds local price_history DB
│
├── database\
│   └── schema.py               ← Full v4.0 SQLite schema
│
├── validation\
│   └── backtest.py             ← Historical pattern backtester
│
├── data\
│   └── momentum_v4.db          ← Created on first run
│
├── logs\                       ← Daily scan logs
└── reports\                    ← (reserved for HTML reports)
```

---

## Scoring Formula (100 points)

| Component | Points | Agent |
|-----------|--------|-------|
| Relative Strength (4/12/26w) | 20 | RS Agent |
| Pattern Recognition (19 patterns) | 18 | Pattern Agent |
| RSI (sweet spot 55-70) | 15 | Pattern Agent |
| Volume Quality (RVOL + delivery) | 12 | Volume Agent |
| EMA Alignment (10/21/50) | 10 | Pattern Agent |
| MACD Bullishness | 8 | Pattern Agent |
| Sector Rotation Rank | 7 | Sector Agent |
| Market Regime Score | 5 | Market Agent |
| Bonus (liquidity + proxy) | 5 | Multiple |

**Regime penalties:** A/B = 0 | C = −5 | D = −12 | E = −25

**Tier gates:**
- Tier 1 (Top Picks): LARGE ≥78, MID ≥80, SMALL ≥82
- Tier 2 (Aggressive): 60 to gate−1
- Tier 3 (Watchlist): 48−59

---

## 19 Patterns

VCP · Bull Flag · Flat Base · Base Breakout · Volume Expansion ·
52W Momentum · Double Bottom · Cup & Handle · Ascending Triangle ·
Symmetrical Triangle · Descending Wedge · Falling Wedge · Rounded Base ·
High Base · 3-Weeks-Tight · Swing High Breakout · Diamond Bottom ·
**High Tight Flag (v4.0)** · **IPO Base (v4.0)**

---

## Gmail App Password Setup

1. Go to myaccount.google.com → Security → 2-Step Verification (enable)
2. Under 2-Step Verification → App passwords
3. Generate a password for "Mail" → copy the 16-character password
4. Paste into `.env` as `GMAIL_APP_PASSWORD=xxxx xxxx xxxx xxxx`

---

## Cloud Deployment (GitHub Actions)

For automatic 7:30 PM IST scans without keeping your laptop on:

1. Create a GitHub account at github.com
2. Create a private repository named `nse_momentum`
3. Upload all project files
4. Go to Settings → Secrets → Actions → add:
   - `GMAIL_ADDRESS` = your Gmail
   - `GMAIL_APP_PASSWORD` = your app password
5. The `.github/workflows/daily_scan.yml` handles the rest

---

## Quick Reference

| Task | Command |
|------|---------|
| Run scan | `python scanner.py` or `run_scanner.bat` |
| Test email | `python emailer.py` |
| Deep-dive | `python single_stock.py RELIANCE` |
| Log entry | `from trade_logger import log_entry; log_entry(...)` |
| Log exit | `from trade_logger import log_exit; log_exit(id, price, 'Target1')` |
| Stats | `from trade_logger import print_stats; print_stats()` |
| Portfolio | `python portfolio_engine.py` |
| Update weights | `python update_weights.py` |
| Backtest | `python validation/backtest.py` |
| Health check | `python health_check.py` |

---

*NSE Momentum v4.3 · Not SEBI-registered investment advice · All free data*
*504 stocks · 12 agents · 19 patterns · Evidence-based · Built Jun 2026*
