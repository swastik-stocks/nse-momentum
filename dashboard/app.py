"""
NSE Momentum v5.2 — Streamlit Dashboard
dashboard/app.py

Run locally:   streamlit run dashboard/app.py
Deploy free:   https://share.streamlit.io (connect your GitHub repo)

Reads directly from data/momentum_v4.db — no extra config needed.
Auto-refreshes every 2 minutes when open in browser.
"""

import sqlite3
import json
from pathlib import Path
from datetime import date, datetime, timedelta

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# ── Config ────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent.parent
DB_PATH = ROOT / "data" / "momentum_v4.db"
VALIDATION_SUMMARY = ROOT / "data" / "pattern_validation_summary.json"

st.set_page_config(
    page_title  = "NSE Momentum v5.2",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "collapsed",
)

# Auto-refresh every 2 minutes
st.markdown(
    '<meta http-equiv="refresh" content="120">',
    unsafe_allow_html=True
)

# ── DB helpers ────────────────────────────────────────────────────────────────

def _conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


@st.cache_data(ttl=120)
def load_today_picks() -> pd.DataFrame:
    try:
        conn = _conn()
        df = pd.read_sql_query("""
            SELECT
                ticker, name, sector, universe, pattern,
                total_score  AS score,
                entry_price  AS entry,
                stop_loss    AS sl,
                target1      AS t1,
                target2      AS t2,
                rrr,
                confidence_pct,
                regime,
                status,
                entry_date   AS date
            FROM trades_v4
            WHERE entry_date = date('now','localtime')
            ORDER BY total_score DESC
        """, conn)
        conn.close()
        df["ticker_clean"] = df["ticker"].str.replace(".NS", "", regex=False)
        return df
    except Exception as e:
        st.error(f"DB read failed: {e}")
        return pd.DataFrame()


@st.cache_data(ttl=120)
def load_picks_json() -> list:
    """Fallback: read picks_latest.json if trades_v4 is empty for today."""
    p = ROOT / "picks_latest.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            return json.load(f)
    return []


@st.cache_data(ttl=120)
def load_regime() -> dict:
    try:
        conn = _conn()
        row = conn.execute("""
            SELECT regime, breadth_score, ad_ratio, above_50_pct,
                   new_highs, new_lows, nifty_close, vix
            FROM market_regime_history
            WHERE date = date('now','localtime')
            LIMIT 1
        """).fetchone()
        conn.close()
        return dict(row) if row else {}
    except Exception:
        return {}


@st.cache_data(ttl=300)
def load_trade_log() -> pd.DataFrame:
    try:
        conn = _conn()
        df = pd.read_sql_query("""
            SELECT ticker, name, sector, pattern, entry_date, exit_date,
                   entry_price, exit_price, stop_loss, target1,
                   exit_type, r_multiple, pnl_pct, total_score,
                   regime, status
            FROM trades_v4
            WHERE status IN ('CLOSED','OPEN')
            ORDER BY entry_date DESC
            LIMIT 200
        """, conn)
        conn.close()
        df["ticker_clean"] = df["ticker"].str.replace(".NS", "", regex=False)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=600)
def load_pattern_validation() -> pd.DataFrame:
    if VALIDATION_SUMMARY.exists():
        return pd.read_json(VALIDATION_SUMMARY)
    return pd.DataFrame()


@st.cache_data(ttl=120)
def load_near_breakout() -> list:
    p = ROOT / "picks_latest.json"
    if p.exists():
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        return [x for x in data if x.get("tier", 0) == 0]
    return []


# ── Header ────────────────────────────────────────────────────────────────────

st.markdown("## 📈 NSE Momentum v5.2")
st.caption(f"Live terminal · {date.today().strftime('%A %d %b %Y')} · "
           f"auto-refresh every 2 min")

regime = load_regime()
if regime:
    c1, c2, c3, c4, c5, c6 = st.columns(6)
    c1.metric("Regime",   regime.get("regime",       "—"))
    c2.metric("Breadth",  f"{regime.get('breadth_score', 0)}/10")
    c3.metric("VIX",      f"{regime.get('vix', 0):.1f}")
    c4.metric("A/D",      f"{regime.get('ad_ratio', 0):.2f}")
    c5.metric("% > 50EMA",f"{regime.get('above_50_pct', 0):.0f}%")
    c6.metric("Nifty",    f"₹{regime.get('nifty_close', 0):,.0f}")

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4 = st.tabs([
    "✅ Today's Picks",
    "📒 Trade Log",
    "🔬 Pattern Edge",
    "⚡ Near-Breakout",
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB 1 — Today's Picks
# ─────────────────────────────────────────────────────────────────────────────
with tab1:
    picks = load_today_picks()

    # Fallback to picks_latest.json if DB has no entries yet today
    if picks.empty:
        raw = load_picks_json()
        if raw:
            picks = pd.DataFrame(raw)
            picks["ticker_clean"] = picks["ticker"].str.replace(".NS", "", regex=False)
            picks["score"] = picks.get("score", 0)

    if picks.empty:
        st.info("No picks yet for today. Evening scan runs after market close.")
    else:
        t1 = picks[picks.get("tier", pd.Series()).eq(1)] if "tier" in picks.columns \
             else picks.head(5)
        t2 = picks[picks.get("tier", pd.Series()).eq(2)] if "tier" in picks.columns \
             else picks.iloc[5:]

        # Summary metrics
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("T1 Picks",   len(t1))
        m2.metric("T2 Watchlist", len(t2))
        m3.metric("Avg Score",  f"{picks['score'].mean():.0f}" if "score" in picks.columns else "—")
        m4.metric("Sectors",    picks["sector"].nunique() if "sector" in picks.columns else "—")

        st.subheader("Tier 1 — Actionable now")

        if t1.empty:
            st.warning("No T1 picks today.")
        else:
            # Trade cards — 3 per row
            cols = st.columns(3)
            for i, (_, row) in enumerate(t1.iterrows()):
                with cols[i % 3]:
                    entry = float(row.get("entry", 0) or 0)
                    sl    = float(row.get("sl", row.get("stop_loss", 0)) or 0)
                    t1p   = float(row.get("t1", row.get("target1",  0)) or 0)
                    score = int(row.get("score", row.get("total_score", 0)) or 0)

                    risk_pct   = ((entry - sl) / entry * 100) if entry > 0 and sl > 0 else 0
                    reward_pct = ((t1p - entry) / entry * 100) if entry > 0 and t1p > 0 else 0
                    rr         = (reward_pct / risk_pct) if risk_pct > 0 else 0

                    with st.container(border=True):
                        st.markdown(
                            f"**{row.get('name', row.get('ticker_clean',''))}** "
                            f"`{row.get('ticker_clean', '')}`"
                        )
                        st.caption(
                            f"{row.get('universe','?')} · {row.get('sector','?')} · "
                            f"{row.get('pattern','?')}"
                        )
                        st.progress(min(score, 100),
                                    text=f"Score {score}/100")

                        ca, cb = st.columns(2)
                        ca.metric("Entry",  f"₹{entry:,.1f}" if entry else "—")
                        cb.metric("SL",     f"₹{sl:,.1f}",
                                  delta=f"-{risk_pct:.1f}%",
                                  delta_color="inverse")

                        cc, cd = st.columns(2)
                        cc.metric("Target", f"₹{t1p:,.1f}",
                                  delta=f"+{reward_pct:.1f}%")
                        cd.metric("R:R",    f"{rr:.1f}x")

        st.subheader("Tier 2 — Watchlist")
        if not t2.empty:
            disp_cols = [c for c in
                         ["ticker_clean", "sector", "pattern", "score",
                          "entry", "sl", "t1", "rr"]
                         if c in t2.columns]
            st.dataframe(
                t2[disp_cols].rename(columns={
                    "ticker_clean": "Ticker", "score": "Score",
                    "entry": "Entry ₹", "sl": "SL ₹",
                    "t1": "T1 ₹", "rr": "R:R"
                }),
                use_container_width=True, hide_index=True
            )

        # Sector distribution
        if "sector" in picks.columns and not picks.empty:
            st.subheader("Sector distribution")
            sec_df = picks["sector"].value_counts().reset_index()
            sec_df.columns = ["Sector", "Count"]
            fig = px.bar(sec_df, x="Sector", y="Count",
                         color="Count", color_continuous_scale="teal",
                         height=280)
            fig.update_layout(showlegend=False,
                              plot_bgcolor="rgba(0,0,0,0)",
                              paper_bgcolor="rgba(0,0,0,0)",
                              margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────────────
# TAB 2 — Trade Log
# ─────────────────────────────────────────────────────────────────────────────
with tab2:
    trades = load_trade_log()

    if trades.empty:
        st.info("No trades in the log yet.")
    else:
        closed = trades[trades["status"] == "CLOSED"]
        open_t = trades[trades["status"] == "OPEN"]

        # Summary metrics
        if not closed.empty and "r_multiple" in closed.columns:
            wins   = closed[closed["r_multiple"] > 0]
            losses = closed[closed["r_multiple"] <= 0]
            wr     = len(wins) / len(closed) * 100 if len(closed) else 0
            avg_r  = closed["r_multiple"].mean()
            pf     = (abs(wins["r_multiple"].sum() / losses["r_multiple"].sum())
                      if len(losses) > 0 and losses["r_multiple"].sum() != 0 else 0)

            m1, m2, m3, m4, m5 = st.columns(5)
            m1.metric("Total Closed",   len(closed))
            m2.metric("Win Rate",       f"{wr:.1f}%")
            m3.metric("Avg R-Multiple", f"{avg_r:.2f}R")
            m4.metric("Profit Factor",  f"{pf:.2f}x")
            m5.metric("Open Positions", len(open_t))

            # Monthly P&L bar chart
            if "entry_date" in closed.columns and "pnl_pct" in closed.columns:
                closed["month"] = pd.to_datetime(
                    closed["entry_date"], errors="coerce"
                ).dt.strftime("%Y-%m")
                monthly = closed.groupby("month")["pnl_pct"].sum().reset_index()
                monthly.columns = ["Month", "P&L %"]
                if not monthly.empty:
                    monthly["color"] = monthly["P&L %"].apply(
                        lambda x: "#16a34a" if x > 0 else "#dc2626"
                    )
                    fig2 = px.bar(monthly, x="Month", y="P&L %",
                                  title="Monthly P&L (%)",
                                  color="color",
                                  color_discrete_map="identity",
                                  height=260)
                    fig2.update_layout(
                        showlegend=False,
                        plot_bgcolor="rgba(0,0,0,0)",
                        paper_bgcolor="rgba(0,0,0,0)",
                        margin=dict(l=0, r=0, t=30, b=0)
                    )
                    st.plotly_chart(fig2, use_container_width=True)

        st.subheader("All trades")
        disp = [c for c in
                ["ticker_clean", "pattern", "entry_date", "exit_date",
                 "entry_price", "exit_price", "r_multiple", "pnl_pct",
                 "exit_type", "regime", "status"]
                if c in trades.columns]
        st.dataframe(
            trades[disp].rename(columns={
                "ticker_clean": "Ticker", "entry_date": "Entry Date",
                "exit_date": "Exit Date", "entry_price": "Entry ₹",
                "exit_price": "Exit ₹", "r_multiple": "R-Multiple",
                "pnl_pct": "P&L %", "exit_type": "Exit", "regime": "Regime"
            }),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 3 — Pattern Edge (from pattern_validator.py output)
# ─────────────────────────────────────────────────────────────────────────────
with tab3:
    val_df = load_pattern_validation()

    if val_df.empty:
        st.info(
            "No validation data yet. Run the pattern validator first:\n\n"
            "```\npython testing/pattern_validator.py\n```\n\n"
            "Takes ~10–15 minutes. Results auto-load here when ready."
        )
    else:
        val_df = val_df.sort_values("expectancy", ascending=False)

        v1, v2, v3 = st.columns(3)
        keep  = val_df[val_df["verdict"].str.contains("KEEP",  na=False)]
        watch = val_df[val_df["verdict"].str.contains("WATCH", na=False)]
        prune = val_df[val_df["verdict"].str.contains("PRUNE", na=False)]

        v1.metric("✅ KEEP",  len(keep))
        v2.metric("👀 WATCH", len(watch))
        v3.metric("❌ PRUNE", len(prune))

        # Expectancy bar chart
        fig3 = px.bar(
            val_df, x="pattern", y="expectancy",
            color="expectancy",
            color_continuous_scale=["#dc2626", "#f59e0b", "#16a34a"],
            title="Expectancy per trade (%) by pattern",
            height=320,
            labels={"expectancy": "Expectancy %", "pattern": "Pattern"}
        )
        fig3.add_hline(y=0, line_dash="dash", line_color="gray")
        fig3.update_layout(
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=30, b=0),
            xaxis_tickangle=-40
        )
        st.plotly_chart(fig3, use_container_width=True)

        # Full table
        disp_cols = [c for c in
                     ["pattern", "n", "win_rate", "expectancy",
                      "avg_win", "avg_loss", "profit_factor",
                      "avg_hold", "avg_max_adverse", "verdict"]
                     if c in val_df.columns]
        st.dataframe(
            val_df[disp_cols].rename(columns={
                "n": "Signals", "win_rate": "Win %",
                "expectancy": "Exp %", "avg_win": "Avg Win %",
                "avg_loss": "Avg Loss %", "profit_factor": "PF",
                "avg_hold": "Hold Days", "avg_max_adverse": "Max Adverse %",
                "verdict": "Verdict"
            }),
            use_container_width=True, hide_index=True
        )

# ─────────────────────────────────────────────────────────────────────────────
# TAB 4 — Near-Breakout watchlist
# ─────────────────────────────────────────────────────────────────────────────
with tab4:
    nb = load_near_breakout()
    st.subheader("Near-breakout watchlist")
    st.caption("Set alerts only — do not enter until breakout confirmed on 1.5× volume")

    if not nb:
        st.info("No near-breakout stocks detected in last scan.")
    else:
        nb_df = pd.DataFrame(nb)
        if "ticker" in nb_df.columns:
            nb_df["ticker"] = nb_df["ticker"].str.replace(".NS", "", regex=False)
        st.dataframe(nb_df, use_container_width=True, hide_index=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "NSE Momentum Scanner v5.2 · 404 stocks · All free data · "
    "Not SEBI-registered investment advice. All trading involves capital risk."
)
