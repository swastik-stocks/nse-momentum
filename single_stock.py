"""
NSE Momentum v4.3 — Single Stock Analysis
Deep-dive a single ticker through all 12 agents.
Usage: python single_stock.py RELIANCE
       python single_stock.py PERSISTENT
"""

import sys, logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)


def analyse(ticker: str):
    # Normalise ticker
    if not ticker.endswith(".NS"):
        ticker = ticker + ".NS"

    from data_fetcher import fetch_single, get_market_context, BhavcopyFetcher
    from nse_universe import NSE_UNIVERSE, UNIVERSE_CONFIG
    from trade_logger import init_tables

    init_tables()

    log.info(f"\n{'='*60}")
    log.info(f"  SINGLE STOCK ANALYSIS: {ticker}")
    log.info(f"{'='*60}\n")

    # Fetch data
    df = fetch_single(ticker, period="2y")
    if df.empty or len(df) < 60:
        print(f"❌ Could not fetch data for {ticker}. Check the symbol.")
        return

    ctx = get_market_context()
    delivery = BhavcopyFetcher().get_delivery_pct()

    # Universe lookup
    uni_map = {item[0]: (item[1], item[2], item[3]) for item in NSE_UNIVERSE}
    if ticker in uni_map:
        name, sector, universe = uni_map[ticker]
    else:
        name = ticker.replace(".NS", "")
        sector = "Unknown"
        universe = "LARGE"

    log.info(f"Name: {name} | Sector: {sector} | Universe: {universe}")
    log.info(f"Price: Rs.{float(df['Close'].iloc[-1]):.2f} | Bars: {len(df)}\n")

    # Prepare data dict
    data_dict = {
        "stock_data":     {ticker: df},
        "nifty50_data":   ctx["nifty50"],
        "banknifty_data": ctx["banknifty"],
        "nifty500_data":  ctx["nifty500"],
        "vix":            ctx["vix"],
        "delivery_data":  delivery,
        "universe_meta":  {ticker: sector},
    }

    from orchestrator import AgentOrchestrator
    orc = AgentOrchestrator(data_dict)
    result = orc.run(ticker, name, sector, df, delivery, universe)

    # Print detailed report
    print(f"\n{'═'*60}")
    print(f"  {ticker.replace('.NS','')} — {name}")
    print(f"  {sector} · {universe} · Rs.{result.price:.2f}")
    print(f"{'═'*60}")
    print(f"\n  TOTAL SCORE:  {result.total_score} / 100")
    print(f"  Confidence:   {result.confidence_pct:.0f}%")
    print(f"  Regime:       {result.market_regime} ({orc.regime_name})")
    print(f"  Breadth:      {result.breadth_score}/10\n")

    if result.rejected:
        print(f"  ❌ REJECTED: {result.reject_reason}")
    else:
        tier_labels = {1: "TOP PICK", 2: "AGGRESSIVE", 3: "WATCHLIST"}
        print(f"  TIER {result.tier} — {tier_labels.get(result.tier,'')}")
        print(f"\n  PATTERN:      {result.pattern}")
        print(f"  Breakout:     Rs.{result.breakout_level:.2f}")
        print(f"\n  LEVELS:")
        print(f"    Entry:      Rs.{result.entry:.2f}")
        print(f"    Stop Loss:  Rs.{result.stop_loss:.2f} ({result.stop_pct:.1f}% risk)")
        print(f"    Target 1:   Rs.{result.target1:.2f} (+{result.gain_pct_t1:.1f}%)")
        print(f"    Target 2:   Rs.{result.target2:.2f}")
        print(f"    R:R Ratio:  {result.rrr:.1f}×\n")
        print(f"  SCORES:")
        print(f"    RS:         {result.rs_score:>3} (Percentile: {result.rs_percentile:.0f}th)")
        print(f"    Pattern:    {result.pattern_score:>3} ({result.pattern})")
        print(f"    RSI:        {result.rsi_score:>3} (RSI value: {result.rsi_val:.0f})")
        print(f"    Volume:     {result.volume_score:>3} (RVOL: {result.rvol:.1f}×)")
        print(f"    EMA:        {result.ema_score:>3}")
        print(f"    Market:     {result.market_score:>3}")
        print(f"    MACD:       {result.macd_score:>3}")
        print(f"    Sector:     {result.sector_score:>3}")
        print(f"    Bonus:      {result.bonus_score:>3}")
        print(f"    ─────────────────")
        print(f"    Raw:        {result.raw_score:>3}")
        print(f"    Penalty:    {result.total_score - result.raw_score:>3}")
        print(f"    TOTAL:      {result.total_score:>3}")

        if result.what_is_working:
            print(f"\n  ✅ WHAT'S WORKING:")
            for w in result.what_is_working:
                print(f"    · {w}")
        if result.what_is_missing:
            print(f"\n  ⚠  CONDITIONS MISSING:")
            for m in result.what_is_missing:
                print(f"    · {m}")
        if result.trigger_conditions:
            print(f"\n  → TRIGGERS TO ACT:")
            for t in result.trigger_conditions:
                print(f"    · {t}")
        if result.risk_factors:
            print(f"\n  ⚡ RISK FACTORS:")
            for rk in result.risk_factors:
                print(f"    · {rk}")

    print(f"\n{'═'*60}\n")


if __name__ == "__main__":
    ticker = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE"
    analyse(ticker)
