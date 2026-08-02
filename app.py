"""
app.py — Portfolio Dashboard (Streamlit)

Run: streamlit run app.py

Dynamic portfolio tracker: add/edit/delete positions (ticker, purchase
price, quantity, account), live prices via free Yahoo Finance data,
consolidated or per-account view, optional DeepSeek red-flag/SWOT
commentary per stock.

All data lives in a local SQLite file (portfolio.db) — nothing is sent
anywhere except: (1) yfinance price lookups (ticker symbols only, no
personal data), (2) DeepSeek analysis calls IF you click "Analyze" on
a stock (sends only the ticker + company name, nothing else).
"""

import streamlit as st
import pandas as pd
from datetime import datetime

from auth import check_password
from db import (init_db, add_holding, update_holding, delete_holding, get_all_lots,
                 get_consolidated, get_accounts, get_signal_cache, save_signal_cache,
                 get_scanner_signals)
from price_fetcher import get_live_prices
from deepseek_client import get_analysis, is_configured as deepseek_configured
from signal_engine import VERDICT_COLOR
from quick_signal import compute_quick_signal

st.set_page_config(page_title="Portfolio Dashboard", layout="wide")

if not check_password():
    st.stop()

init_db()

# ── Session state ────────────────────────────────────────────────────────
if "editing_id" not in st.session_state:
    st.session_state.editing_id = None


# ── Helpers ───────────────────────────────────────────────────────────────
def fmt_inr(x: float) -> str:
    return f"₹{x:,.2f}"


def pct(x: float) -> str:
    sign = "+" if x >= 0 else ""
    return f"{sign}{x:.2f}%"


VERDICT_EMOJI = {
    "ADD": "🟢 ADD",
    "HOLD": "🟡 HOLD",
    "TRIM": "🟠 TRIM",
    "STRONG TRIM": "🔴 STRONG TRIM",
    "REVIEW": "⚪ REVIEW",
}


def build_view(rows: list, prices: dict, signal_cache: dict = None) -> pd.DataFrame:
    signal_cache = signal_cache or {}
    out = []
    for r in rows:
        ltp = prices.get(r["symbol"])
        invested = r.get("total_invested", r["qty"] * r["avg_price"])
        current_value = ltp * r["qty"] if ltp is not None else None
        pnl = (current_value - invested) if current_value is not None else None
        pnl_pct = (pnl / invested * 100) if pnl is not None and invested else None

        cached = signal_cache.get(r["symbol"])
        signal_display = VERDICT_EMOJI.get(cached["verdict"], "—") if cached else "— (not run)"

        out.append({
            "id": r.get("id"),
            "Signal": signal_display,
            "Symbol": r["symbol"],
            "Name": r.get("company_name", ""),
            "Account(s)": ", ".join(r["accounts"]) if "accounts" in r else r.get("account", ""),
            "Qty": r["qty"],
            "Avg Price": r["avg_price"],
            "LTP": ltp,
            "Invested": invested,
            "Current Value": current_value,
            "P&L": pnl,
            "P&L %": pnl_pct,
        })
    return pd.DataFrame(out)


# ── Sidebar: Add / Edit holding ──────────────────────────────────────────
with st.sidebar:
    if st.button("🔓 Log out"):
        st.session_state.authenticated = False
        st.rerun()
    st.divider()
    st.header("Add / Edit Holding")

    editing = st.session_state.editing_id is not None
    lots = get_all_lots()
    edit_row = next((l for l in lots if l["id"] == st.session_state.editing_id), None) if editing else None

    with st.form("holding_form", clear_on_submit=not editing):
        symbol = st.text_input(
            "NSE Ticker (e.g. RELIANCE.NS)",
            value=edit_row["symbol"] if edit_row else "",
        ).strip().upper()
        company_name = st.text_input(
            "Company Name",
            value=edit_row["company_name"] if edit_row else "",
        )
        account = st.text_input(
            "Account tag (e.g. AXIS, YES, DHAN)",
            value=edit_row["account"] if edit_row else "",
        )
        qty = st.number_input(
            "Quantity",
            min_value=0.0, step=1.0,
            value=float(edit_row["qty"]) if edit_row else 0.0,
        )
        avg_price = st.number_input(
            "Purchase Price (per share)",
            min_value=0.0, step=0.01, format="%.2f",
            value=float(edit_row["avg_price"]) if edit_row else 0.0,
        )
        notes = st.text_input("Notes (optional)", value=edit_row["notes"] if edit_row else "")

        col1, col2 = st.columns(2)
        submit_label = "Update Holding" if editing else "Add Holding"
        submitted = col1.form_submit_button(submit_label, use_container_width=True)
        cancelled = col2.form_submit_button("Cancel", use_container_width=True) if editing else False

        if submitted and symbol and qty > 0 and avg_price > 0:
            if not symbol.endswith(".NS") and not symbol.endswith(".BO"):
                st.warning("Ticker should end in .NS (NSE) or .BO (BSE) — added .NS automatically.")
                symbol = symbol + ".NS"
            if editing:
                update_holding(st.session_state.editing_id, symbol, company_name, account, qty, avg_price, notes)
                st.session_state.editing_id = None
            else:
                add_holding(symbol, company_name, account, qty, avg_price, notes)
            st.rerun()

        if cancelled:
            st.session_state.editing_id = None
            st.rerun()

    st.divider()
    st.caption(
        "DeepSeek analysis: " + ("configured ✓" if deepseek_configured() else "not configured — add DEEPSEEK_API_KEY to .env to enable")
    )


# ── Main ──────────────────────────────────────────────────────────────────
st.title("📊 Portfolio Dashboard")

lots = get_all_lots()
if not lots:
    st.info("No holdings yet. Add your first position using the sidebar.")
    st.stop()

def render_analysis_result(result, symbol: str = None):
    """Shared rendering for the DeepSeek analysis: verdict badge first
    (the computed decision, big and colored), then the narrative
    explaining it, then the fully auditable grounded data underneath.
    Also updates signal_cache for this symbol with the fresh verdict
    just computed, so the main table's badge reflects it immediately
    without needing a full "Refresh Signals" pass."""
    sig = result["signal"]
    if symbol:
        save_signal_cache(symbol, sig.verdict, sig.fundamental_score,
                          sig.news_sentiment, sig.data_completeness,
                          datetime.now().isoformat())
    color = VERDICT_COLOR.get(sig.verdict, "#8FA3B8")
    completeness_note = "" if sig.data_completeness == "FULL" else f" &nbsp; \u26a0\ufe0f {sig.data_completeness} DATA"
    st.markdown(
        f"<div style='background:{color}22;border:2px solid {color};border-radius:8px;"
        f"padding:12px 18px;margin-bottom:12px'>"
        f"<span style='font-size:24px;font-weight:800;color:{color}'>{sig.verdict}</span>"
        f"<span style='font-size:12px;color:#888'>{completeness_note}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )
    st.markdown(result["text"])
    with st.expander("Show grounded data behind this verdict"):
        st.write(f"**Fundamental score:** {sig.fundamental_score:.2f}")
        for factor in sig.fundamental_factors:
            st.write(f"- {factor}")
        st.write(f"**News sentiment (analyst-tagged):** {sig.news_sentiment:+d}")
        for nf in sig.news_factors:
            st.write(f"- {nf}")
        st.write(f"**Position:** {sig.position_note}")
        st.write("**Raw fundamentals (Screener.in):**")
        st.json(result["fundamentals"])


view_mode = st.radio("View", ["Consolidated (across all accounts)", "By Account"], horizontal=True)

col_refresh, col_signals, col_time = st.columns([1, 1, 3])
force_refresh = col_refresh.button("🔄 Refresh Prices")
refresh_signals_clicked = col_signals.button("🚦 Refresh Signals")

all_symbols = sorted(set(l["symbol"] for l in lots))
with st.spinner("Fetching live prices..."):
    prices = get_live_prices(all_symbols, force_refresh=force_refresh)

col_time.caption(f"Prices as of {datetime.now().strftime('%H:%M:%S')} (cached ~60s)")

missing_prices = [s for s in all_symbols if prices.get(s) is None]
if missing_prices:
    st.warning(f"Couldn't fetch live price for: {', '.join(missing_prices)}. "
               f"Check the ticker is correct, or market may be closed with no cached data yet.")

# ── Refresh Signals: computes ADD/HOLD/TRIM for every holding at once ────
# Deterministic only (no DeepSeek call) -- see quick_signal.py. Rate-limited
# by fundamentals.py's polite Screener.in delay, so this takes roughly
# 2 seconds per unique symbol -- a manual action, not run on every render.
if refresh_signals_clicked:
    consolidated_for_signals = get_consolidated()
    progress = st.progress(0.0, text="Refreshing signals...")
    for i, r in enumerate(consolidated_for_signals):
        symbol = r["symbol"]
        ltp = prices.get(symbol)
        invested = r["total_invested"]
        pnl = (ltp * r["qty"] - invested) if ltp is not None else None
        pnl_pct = (pnl / invested * 100) if pnl is not None and invested else None
        position = {"qty": r["qty"], "avg_price": r["avg_price"], "ltp": ltp,
                    "pnl": pnl, "pnl_pct": pnl_pct}
        progress.progress((i) / len(consolidated_for_signals),
                          text=f"Refreshing signals... {symbol} ({i+1}/{len(consolidated_for_signals)})")
        sig = compute_quick_signal(symbol, r["company_name"], position=position)
        save_signal_cache(symbol, sig.verdict, sig.fundamental_score,
                          sig.news_sentiment, sig.data_completeness,
                          datetime.now().isoformat())
    progress.progress(1.0, text="Signals refreshed.")
    st.rerun()

signal_cache = get_signal_cache()
if signal_cache:
    oldest = min(signal_cache.values(), key=lambda r: r["computed_at"])
    st.caption(f"Signals last refreshed: {oldest['computed_at'][:16].replace('T', ' ')} "
               f"(click 🚦 Refresh Signals above to update)")
else:
    st.info("No signals computed yet — click 🚦 Refresh Signals above to see ADD/HOLD/TRIM "
            "verdicts for all your holdings at once.")

# ── Build the table depending on view mode ──────────────────────────────
if view_mode.startswith("Consolidated"):
    rows = get_consolidated()
    df = build_view(rows, prices, signal_cache)
    st.caption("To edit or delete a specific holding, switch to \"By Account\" view above.")
else:
    df = build_view(lots, prices, signal_cache)

# ── Summary metrics ──────────────────────────────────────────────────────
total_invested = df["Invested"].sum()
total_current = df["Current Value"].sum(skipna=True)
total_pnl = total_current - total_invested if pd.notna(total_current) else None
total_pnl_pct = (total_pnl / total_invested * 100) if total_pnl is not None and total_invested else None

m1, m2, m3, m4 = st.columns(4)
m1.metric("Total Invested", fmt_inr(total_invested))
m2.metric("Current Value", fmt_inr(total_current) if pd.notna(total_current) else "—")
m3.metric("Total P&L", fmt_inr(total_pnl) if total_pnl is not None else "—",
          delta=pct(total_pnl_pct) if total_pnl_pct is not None else None)
m4.metric("Positions", len(df))

st.divider()

# ── Holdings table ────────────────────────────────────────────────────────
display_df = df.copy()
for col in ["Avg Price", "LTP", "Invested", "Current Value", "P&L"]:
    display_df[col] = display_df[col].apply(lambda x: fmt_inr(x) if pd.notna(x) else "—")
display_df["P&L %"] = df["P&L %"].apply(lambda x: pct(x) if pd.notna(x) else "—")

st.dataframe(
    display_df.drop(columns=["id"]) if "id" in display_df.columns else display_df,
    use_container_width=True,
    hide_index=True,
)

# ── Per-holding actions (edit/delete/analyze) — only meaningful in raw lot view ──
if not view_mode.startswith("Consolidated"):
    st.subheader("Manage Lots")
    for lot in lots:
        with st.expander(f"{lot['symbol']} — {lot['account']} — {lot['qty']} @ ₹{lot['avg_price']:.2f}"):
            c1, c2, c3 = st.columns(3)
            if c1.button("Edit", key=f"edit_{lot['id']}"):
                st.session_state.editing_id = lot["id"]
                st.rerun()
            if c2.button("Delete", key=f"delete_{lot['id']}"):
                delete_holding(lot["id"])
                st.rerun()
            if c3.button("Analyze (DeepSeek)", key=f"analyze_{lot['id']}", disabled=not deepseek_configured()):
                ltp = prices.get(lot["symbol"])
                invested = lot["qty"] * lot["avg_price"]
                pnl = (ltp * lot["qty"] - invested) if ltp is not None else None
                pnl_pct = (pnl / invested * 100) if pnl is not None and invested else None
                position = {"qty": lot["qty"], "avg_price": lot["avg_price"], "ltp": ltp,
                            "pnl": pnl, "pnl_pct": pnl_pct}
                with st.spinner("Fetching fundamentals + news + analyzing..."):
                    result = get_analysis(lot["symbol"], lot["company_name"], position=position)
                if result:
                    render_analysis_result(result, symbol=lot["symbol"])
                else:
                    st.error("Analysis unavailable — check DEEPSEEK_API_KEY in .env, or Screener.in fetch failed")
else:
    st.subheader("Per-Stock Analysis")
    consolidated = get_consolidated()
    # Sort so TRIM/STRONG TRIM/REVIEW stocks surface first -- these are the
    # ones actually worth spending a DeepSeek call to dig into.
    verdict_priority = {"STRONG TRIM": 0, "TRIM": 1, "REVIEW": 2, "HOLD": 3, "ADD": 4}
    def _sort_key(r):
        cached = signal_cache.get(r["symbol"])
        v = cached["verdict"] if cached else "REVIEW"
        return verdict_priority.get(v, 5)
    consolidated_sorted = sorted(consolidated, key=_sort_key)

    def _label(r):
        cached = signal_cache.get(r["symbol"])
        badge = VERDICT_EMOJI.get(cached["verdict"], "") if cached else "(no signal yet)"
        return f"{badge}  {r['symbol']}"

    symbol_labels = [_label(r) for r in consolidated_sorted]
    chosen_label = st.selectbox("Select a stock (worst signals shown first)", symbol_labels)
    symbol_choice = consolidated_sorted[symbol_labels.index(chosen_label)]["symbol"]
    if st.button("Analyze (DeepSeek)", disabled=not deepseek_configured()):
        chosen = next(r for r in consolidated if r["symbol"] == symbol_choice)
        ltp = prices.get(chosen["symbol"])
        invested = chosen["total_invested"]
        pnl = (ltp * chosen["qty"] - invested) if ltp is not None else None
        pnl_pct = (pnl / invested * 100) if pnl is not None and invested else None
        position = {"qty": chosen["qty"], "avg_price": chosen["avg_price"], "ltp": ltp,
                    "pnl": pnl, "pnl_pct": pnl_pct}
        with st.spinner("Fetching fundamentals + news + analyzing..."):
            result = get_analysis(chosen["symbol"], chosen["company_name"], position=position)
        if result:
            render_analysis_result(result, symbol=chosen["symbol"])
        else:
            st.error("Analysis unavailable — check DEEPSEEK_API_KEY in .env, or Screener.in fetch failed")

# ── Scanner Signals (P1-05) — NSE Momentum's evening-scan picks, shown ───
# alongside holdings so one screen answers both "what do I own" and
# "what's worth looking at today." Read-only, best-effort: get_scanner_signals()
# already returns [] on any failure (missing table, connection issue, etc.)
# so this section simply doesn't render rather than breaking the dashboard.
st.divider()
st.subheader("🎯 Scanner Signals (NSE Momentum)")

signals = get_scanner_signals()
if not signals:
    st.info("No scanner signals published yet — the evening scan runs Mon–Fri after market close "
            "(NSE Momentum's daily_scan.yml). Check back after the next trading day.")
else:
    scan_date = signals[0].get("scan_date", "")
    st.caption(f"From the {scan_date} evening scan — {len(signals)} signal(s), sorted by score. "
               f"Not SEBI-registered investment advice.")

    held_symbols = {lot["symbol"] for lot in lots}

    signal_rows = []
    for s in signals:
        signal_rows.append({
            "Owned": "✅" if s.get("ticker") in held_symbols else "",
            "Ticker": s.get("ticker", ""),
            "Pattern": s.get("pattern") or "",
            "Score": s.get("total_score"),
            "Tier": s.get("tier"),
            "Confidence": f"{s['confidence_pct']:.0f}%" if s.get("confidence_pct") is not None else "—",
            "Entry": s.get("entry_price"),
            "Pivot": s.get("pivot"),
            "Stop": s.get("stop_loss"),
            "Target 1": s.get("target1"),
            "Target 2": s.get("target2"),
            "R:R": f"{s['rrr']:.1f}x" if s.get("rrr") is not None else "—",
            "RS %ile": f"{s['rs_percentile']:.0f}th" if s.get("rs_percentile") is not None else "—",
            "Regime": s.get("regime") or "",
        })
    signal_df = pd.DataFrame(signal_rows)
    for money_col in ["Entry", "Pivot", "Stop", "Target 1", "Target 2"]:
        signal_df[money_col] = signal_df[money_col].apply(lambda x: fmt_inr(x) if pd.notna(x) else "—")

    st.dataframe(signal_df, use_container_width=True, hide_index=True)

    owned_count = sum(1 for s in signals if s.get("ticker") in held_symbols)
    if owned_count:
        st.caption(f"✅ {owned_count} of these signals are for stocks you already own.")

st.divider()
st.caption(
    "Not SEBI-registered investment advice. Prices via Yahoo Finance (free, "
    "may lag real-time by a few minutes). DeepSeek analysis is for research "
    "reference only, not a buy/sell recommendation."
)
