#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
NSE Momentum v5.4 — Regime Narrator (NVIDIA NIM integration)
==============================================================
NOTE ON SCOPE — read this before wiring it in:
  The regime_classifier.py / market_breadth_agent.py sanity checks (R1-R4,
  SANITY FAIL, corrected VIX thresholds) already catch the exact contradiction
  class that caused the 24 Jun bug (Regime D / A/D 0.27 / HOSTILE on an up day).
  That bug is fixed by hardcoded rules already in the repo — this module does
  NOT replace that fix.

  What this adds instead:
    1. Turns the raw sanity_flags list ("R2: above_50_ema=63.2%...") into a
       plain-English explanation a human can read in the email/dashboard
       without decoding rule codes.
    2. Acts as an independent qualitative second opinion — it sees the full
       numeric context (not just whichever fixed threshold a rule checks) and
       can flag borderline contradictions the hardcoded rules miss (e.g.
       above_50_ema=59%, just under R1's 60% cutoff).

  FAIL-SAFE DESIGN: if NVIDIA_API_KEY is unset, or the API call fails or times
  out, this returns a neutral no-op result and logs a warning. It NEVER raises,
  NEVER blocks the pipeline, and NEVER overrides the existing regime/confidence
  values — orchestrator.py's own rule-based output remains authoritative.
  This is an annotation layer, not a decision-maker.

Setup:
  1. Sign up free at https://build.nvidia.com (NVIDIA Developer Program, no card)
  2. Get an API key (starts with nvapi-) from any model page
  3. Add to your .env:  NVIDIA_API_KEY=nvapi-xxxxxxxxxxxx
  4. pip install openai   (NIM endpoints are OpenAI-SDK-compatible)

Cost: this makes ~1-2 calls per scan run. NVIDIA's free tier is rate-limited
(~40 requests/minute), not credit-metered, for the free hosted models — this
usage pattern should never approach that limit. Verify current terms in your
own build.nvidia.com console before relying on this long-term.
"""

import os
import logging

try:
    from dotenv import load_dotenv
    load_dotenv(override=True)
except ImportError:
    pass

try:
    from loguru import logger as log
except ImportError:
    log = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY", "")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_MODEL = os.getenv("NVIDIA_NIM_MODEL", "meta/llama-3.1-8b-instruct")

_SYSTEM_PROMPT = """You are a market-internals sanity checker for an Indian NSE momentum
trading system. You will be given the day's regime classification along with the raw
numeric inputs and any contradiction flags already raised by hardcoded rules.

Your job has two parts:
1. Explain in 2-3 plain-English sentences what today's regime reading means and why,
   for a trader who does not want to parse rule codes.
2. Independently check the numbers for internal contradictions BEYOND what the
   hardcoded flags already caught — e.g. borderline cases just under/over a threshold,
   or combinations of signals that don't individually breach a rule but look
   inconsistent together. If you find nothing beyond what's already flagged, say so
   explicitly rather than inventing a concern.

Be concise and concrete. Do not give trading advice — only comment on data consistency
and what the regime reading means. If the data looks internally consistent, say that
plainly and briefly; do not manufacture doubt to seem thorough."""


def _build_user_prompt(context: dict) -> str:
    flags = context.get("sanity_flags") or []
    flags_str = "\n".join(f"  - {f}" for f in flags) if flags else "  (none — hardcoded rules found no contradiction)"
    return f"""Today's regime reading:

  Regime: {context.get('regime')} ({context.get('regime_name', '')})
  Confidence: {context.get('regime_confidence')}
  Regime penalty applied: {context.get('regime_penalty')}

  Raw inputs:
  - Nifty above 50-EMA: {context.get('above_50_ema_pct')}%
  - A/D ratio (advances/declines): {context.get('ad_ratio')}
  - Breadth score (0-10): {context.get('breadth_score')}
  - VIX: {context.get('vix')}
  - Macro state: {context.get('macro_state')}
  - FII flow (crore): {context.get('fii_flow_crore')}

  Hardcoded contradiction flags already raised:
{flags_str}
"""


def narrate_regime(context: dict, timeout: float = 25.0) -> dict:
    """
    Calls the NIM API for a plain-English explanation + independent second
    opinion on today's regime reading. Never raises — always returns a dict
    with at minimum {"narrative": str, "llm_flags": list, "source": str}.

    context should include: regime, regime_name, regime_confidence,
    regime_penalty, above_50_ema_pct, ad_ratio, breadth_score, vix,
    macro_state, fii_flow_crore, sanity_flags (list of existing rule flags).
    """
    if not NVIDIA_API_KEY:
        log.info("  [RegimeNarrator] NVIDIA_API_KEY not set — skipping narration (no-op).")
        return {"narrative": "", "llm_flags": [], "source": "SKIPPED_NO_KEY"}

    try:
        from openai import OpenAI
    except ImportError:
        log.warning("  [RegimeNarrator] 'openai' package not installed — skipping. pip install openai")
        return {"narrative": "", "llm_flags": [], "source": "SKIPPED_NO_SDK"}

    try:
        client = OpenAI(base_url=NVIDIA_BASE_URL, api_key=NVIDIA_API_KEY, timeout=timeout)
        resp = client.chat.completions.create(
            model=NVIDIA_MODEL,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": _build_user_prompt(context)},
            ],
            temperature=0.2,
            max_tokens=350,
        )
        narrative = resp.choices[0].message.content.strip()
        log.info(f"  [RegimeNarrator] Narration received ({len(narrative)} chars).")
        return {"narrative": narrative, "llm_flags": [], "source": NVIDIA_MODEL}

    except Exception as e:
        # Deliberately broad: ANY failure here (auth, rate limit, timeout,
        # network, malformed response) must degrade to no-op, never crash
        # the evening scan over an optional annotation feature.
        log.warning(f"  [RegimeNarrator] NIM call failed ({type(e).__name__}: {e}) — skipping, non-fatal.")
        return {"narrative": "", "llm_flags": [], "source": "SKIPPED_ERROR"}


def print_narration(result: dict) -> None:
    if not result.get("narrative"):
        return
    print("\n  " + "─" * 70)
    print(f"  REGIME NARRATIVE  [{result.get('source', '')}]")
    print("  " + "─" * 70)
    for line in result["narrative"].splitlines():
        print(f"  {line}")
    print("  " + "─" * 70 + "\n")


if __name__ == "__main__":
    # Quick standalone smoke test — run this directly to verify your API key
    # and connectivity before wiring into orchestrator.py.
    test_context = {
        "regime": "D",
        "regime_name": "Correction",
        "regime_confidence": "LOW",
        "regime_penalty": -5,
        "above_50_ema_pct": 64.9,
        "ad_ratio": 0.27,
        "breadth_score": 6.2,
        "vix": 13.33,
        "macro_state": "MIXED",
        "fii_flow_crore": -450,
        "sanity_flags": [
            "R2: above_50_ema=64.9% AND ad_ratio=0.270 (both bullish) but regime=D. "
            "Both structural and daily signals contradict the index regime."
        ],
    }
    result = narrate_regime(test_context)
    if result["narrative"]:
        print_narration(result)
    else:
        print(f"No narration produced. source={result['source']}")
        print("Check: is NVIDIA_API_KEY set in your .env? Is 'openai' pip-installed?")
