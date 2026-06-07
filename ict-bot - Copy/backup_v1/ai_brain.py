"""
=============================================================================
ai_brain.py - The Gemini LLM Context & Planner
=============================================================================

Purpose:
    - Fetch HTF data (Monthly, Weekly, Daily, H4, H1) from MetaTrader5
    - Fetch high-impact news from ForexFactory
    - Build a prompt with this context and send to Gemini
    - Save Gemini's structured JSON output to `daily_plan.json`

Run this ONCE per day (e.g., via cron/Task Scheduler at 00:00 UTC).
The execution engine (mt5_sniper.py) reads `daily_plan.json` continuously.

=============================================================================
⚠️ CRITICAL WARNINGS - READ BEFORE RUNNING IN PRODUCTION
=============================================================================

1. LLM HALLUCINATION RISK (HIGH):
   Gemini cannot do precise arithmetic on price candles. When you feed it
   500 OHLC bars and ask "where is the unmitigated FVG?", it will often:
     - Invent prices that don't exist in the data
     - Miss FVGs that ARE there
     - Confuse mitigated vs unmitigated zones
   MITIGATION: We pre-filter and pre-tag candidate FVGs in Python so Gemini
   only chooses BETWEEN real candidates, never invents new ones.

2. RATE LIMITS (Gemini Free Tier):
   - gemini-2.5-flash: 10 requests/min, 500/day (free tier)
   - gemini-1.5-pro:   2 requests/min,  50/day
   Since this script runs ONCE per day, you're safe. But if you re-run on
   errors, watch the daily quota.

3. JSON PARSING FAILURES:
   Gemini sometimes wraps JSON in markdown ```json fences, adds prose, or
   returns invalid JSON. We strip and validate. If parsing fails twice,
   we exit WITHOUT writing `daily_plan.json` so the sniper doesn't trade
   on stale/invalid data.

4. NEWS SCRAPING FRAGILITY:
   ForexFactory has no public API. We scrape the calendar HTML, which
   breaks when they change their layout. RECOMMENDATION: Use Finnhub
   (free 60 calls/min) as primary, ForexFactory as fallback.

5. MT5 PYTHON LIBRARY = WINDOWS ONLY (officially):
   The MetaTrader5 Python package only ships Windows binaries. On macOS/Linux
   you need Wine + Windows Python, which is fragile.

=============================================================================
"""

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import MetaTrader5 as mt5
import google.generativeai as genai
import pandas as pd
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Setup
# -----------------------------------------------------------------------------
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER = os.getenv("MT5_SERVER", "")
MT5_PATH = os.getenv("MT5_PATH", "")
SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "XAUUSD").split(",") if s.strip()]
FINNHUB_API_KEY = os.getenv("FINNHUB_API_KEY", "")

PLAN_DIR = Path(__file__).parent

# Gemini API cooldown — call at most once per hour per symbol/plan-type
GEMINI_COOLDOWN_HOURS = 1
_GEMINI_STAMP_FILE    = PLAN_DIR / ".gemini_timestamps.json"

# Timeframe mapping
TIMEFRAMES = {
    "MN1": (mt5.TIMEFRAME_MN1, 12),   # 12 months
    "W1":  (mt5.TIMEFRAME_W1,  20),   # 20 weeks
    "D1":  (mt5.TIMEFRAME_D1,  30),   # 30 days
    "H4":  (mt5.TIMEFRAME_H4,  60),   # 60 H4 candles ~ 10 days
    "H1":  (mt5.TIMEFRAME_H1,  100),  # 100 H1 ~ 4 days (more history for MSS detection)
    "M15": (mt5.TIMEFRAME_M15, 96),   # 96 M15 candles = 24 hours (scalp context)
}


# =============================================================================
# 1. MT5 CONNECTION
# =============================================================================
def init_mt5() -> bool:
    """Initialize MT5 connection. Returns True on success."""
    if MT5_PATH:
        ok = mt5.initialize(
            path=MT5_PATH,
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )
    else:
        ok = mt5.initialize(
            login=MT5_LOGIN,
            password=MT5_PASSWORD,
            server=MT5_SERVER,
        )

    if not ok:
        err = mt5.last_error()
        print(f"[MT5] initialize() failed: {err}", file=sys.stderr)
        return False

    info = mt5.account_info()
    if info is None:
        print("[MT5] account_info() returned None — login may have failed",
              file=sys.stderr)
        return False

    print(f"[MT5] Connected. Account={info.login}  Server={info.server}  "
          f"Balance={info.balance} {info.currency}")

    # Make sure all symbols are selected in Market Watch
    for sym in SYMBOLS:
        if not mt5.symbol_select(sym, True):
            print(f"[MT5] symbol_select({sym}) failed", file=sys.stderr)
            return False

    return True


def fetch_ohlc(symbol: str, tf_const: int, count: int) -> pd.DataFrame:
    """Fetch the last `count` candles of `symbol` at timeframe `tf_const`."""
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, count)
    if rates is None or len(rates) == 0:
        raise RuntimeError(
            f"copy_rates_from_pos failed for {symbol} tf={tf_const}: "
            f"{mt5.last_error()}"
        )
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df[["time", "open", "high", "low", "close", "tick_volume"]]


# =============================================================================
# 2. PRE-COMPUTE ICT CANDIDATES
# =============================================================================
# Why we do this in Python instead of asking Gemini:
# Gemini cannot reliably scan 500 candles and identify all FVGs/swings without
# missing or hallucinating. We compute REAL candidates here, then ask Gemini
# to PICK between them and reason about bias. This is the only way to keep
# the LLM from inventing price levels.
# =============================================================================

def find_fvgs(df: pd.DataFrame, max_results: int = 10) -> list[dict]:
    """
    Find Fair Value Gaps (3-candle imbalance pattern).

    Bullish FVG: candle[i-2].high < candle[i].low  (gap above)
    Bearish FVG: candle[i-2].low  > candle[i].high (gap below)

    Returns the most recent FVGs, with mitigation flag (whether price has
    revisited the gap since formation).
    """
    fvgs: list[dict] = []
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    t = df["time"].to_numpy()

    for i in range(2, len(df)):
        # Bullish FVG
        if h[i - 2] < l[i]:
            gap_low, gap_high = h[i - 2], l[i]
            # Check if mitigated: any candle AFTER i with low <= gap_high
            mitigated = bool((l[i + 1:] <= gap_high).any()) if i + 1 < len(df) else False
            fvgs.append({
                "type": "bullish",
                "time": str(t[i]),
                "low": float(gap_low),
                "high": float(gap_high),
                "mitigated": mitigated,
                "bar_index_from_end": len(df) - 1 - i,
            })
        # Bearish FVG
        elif l[i - 2] > h[i]:
            gap_low, gap_high = h[i], l[i - 2]
            mitigated = bool((h[i + 1:] >= gap_low).any()) if i + 1 < len(df) else False
            fvgs.append({
                "type": "bearish",
                "time": str(t[i]),
                "low": float(gap_low),
                "high": float(gap_high),
                "mitigated": mitigated,
                "bar_index_from_end": len(df) - 1 - i,
            })

    # Keep only most recent unmitigated + last few mitigated for context
    unmitigated = [f for f in fvgs if not f["mitigated"]][-max_results:]
    return unmitigated


def find_swing_points(df: pd.DataFrame, lookback: int = 3) -> dict:
    """
    Identify swing highs and lows (engineered liquidity targets).
    A swing high = a candle whose high is greater than `lookback` candles
    before AND after it.
    """
    swing_highs, swing_lows = [], []
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    t = df["time"].to_numpy()

    for i in range(lookback, len(df) - lookback):
        window_h = h[i - lookback:i + lookback + 1]
        window_l = l[i - lookback:i + lookback + 1]
        if h[i] == window_h.max():
            swing_highs.append({"time": str(t[i]), "price": float(h[i])})
        if l[i] == window_l.min():
            swing_lows.append({"time": str(t[i]), "price": float(l[i])})

    return {
        "recent_swing_highs": swing_highs[-5:],
        "recent_swing_lows":  swing_lows[-5:],
    }


def calculate_bias(symbol: str) -> str:
    """
    Real-time H1 bias from Python market structure. No LLM, no caching.
    Called at the start of every execution loop to override stale Gemini plans.

    Logic (in priority order):
      1. If last closed H1 candle breaks below most recent 3-bar pivot swing
         low  → "bearish" immediately.
      2. If it breaks above most recent swing high → "bullish" immediately.
      3. Fallback: majority of last 10 H1 candles are directional (≥ 7/10).
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates is None or len(rates) < 15:
        return "neutral"

    df = pd.DataFrame(rates)
    h  = df["high"].to_numpy()
    l  = df["low"].to_numpy()
    c  = df["close"].to_numpy()
    o  = df["open"].to_numpy()

    last_close = c[-2]   # last FULLY closed H1 candle (index -1 is still forming)

    # Backward search for the most recent 3-bar pivot swing high / swing low
    recent_swing_high: float | None = None
    recent_swing_low:  float | None = None

    for i in range(len(df) - 3, 2, -1):
        if recent_swing_high is None:
            if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
                recent_swing_high = float(h[i])
        if recent_swing_low is None:
            if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
                recent_swing_low = float(l[i])
        if recent_swing_high is not None and recent_swing_low is not None:
            break

    # Primary: MSS via swing break — immediate flip
    if recent_swing_low is not None and last_close < recent_swing_low:
        print(f"[{symbol}][BIAS] Bearish MSS — H1 close {last_close} < swing low {recent_swing_low}")
        return "bearish"
    if recent_swing_high is not None and last_close > recent_swing_high:
        print(f"[{symbol}][BIAS] Bullish MSS — H1 close {last_close} > swing high {recent_swing_high}")
        return "bullish"

    # Fallback: majority vote from last 10 closed H1 candles
    bear = sum(1 for i in range(-11, -1) if c[i] < o[i])
    if bear >= 7:
        return "bearish"
    if bear <= 3:
        return "bullish"

    return "neutral"


# =============================================================================
# GEMINI COOLDOWN HELPERS
# =============================================================================
def _load_gemini_stamps() -> dict:
    if _GEMINI_STAMP_FILE.exists():
        try:
            return json.loads(_GEMINI_STAMP_FILE.read_text())
        except Exception:
            pass
    return {}


def _save_gemini_stamps(stamps: dict):
    _GEMINI_STAMP_FILE.write_text(json.dumps(stamps, indent=2))


def _stamp_key(symbol: str, plan_type: str) -> str:
    return f"{symbol}_{plan_type}"


def should_call_gemini(symbol: str, plan_type: str) -> bool:
    """Return True if the cooldown has elapsed (or no previous call recorded)."""
    stamps = _load_gemini_stamps()
    key = _stamp_key(symbol, plan_type)
    ts_str = stamps.get(key)
    if not ts_str:
        return True
    try:
        last = datetime.fromisoformat(ts_str)
        elapsed_h = (datetime.now(timezone.utc) - last).total_seconds() / 3600
        return elapsed_h >= GEMINI_COOLDOWN_HOURS
    except Exception:
        return True


def mark_gemini_called(symbol: str, plan_type: str):
    stamps = _load_gemini_stamps()
    stamps[_stamp_key(symbol, plan_type)] = datetime.now(timezone.utc).isoformat()
    _save_gemini_stamps(stamps)


def remaining_cooldown_min(symbol: str, plan_type: str) -> float:
    """Minutes remaining in current cooldown window (0 if cooldown elapsed)."""
    stamps = _load_gemini_stamps()
    ts_str = stamps.get(_stamp_key(symbol, plan_type))
    if not ts_str:
        return 0.0
    try:
        last = datetime.fromisoformat(ts_str)
        elapsed_m = (datetime.now(timezone.utc) - last).total_seconds() / 60
        remaining = GEMINI_COOLDOWN_HOURS * 60 - elapsed_m
        return max(0.0, remaining)
    except Exception:
        return 0.0


def refresh_bias_in_plan(plan_file: Path, symbol: str):
    """
    Between Gemini API calls: load existing plan, update `bias` field using
    Python market structure (calculate_bias), and re-save.  Called every loop.
    """
    if not plan_file.exists():
        return
    try:
        plan = json.loads(plan_file.read_text())
    except Exception:
        return
    live_bias = calculate_bias(symbol)
    if live_bias != "neutral" and plan.get("bias") != live_bias:
        old = plan.get("bias")
        plan["bias"] = live_bias
        plan["_bias_refreshed_at"] = datetime.now(timezone.utc).isoformat()
        plan_file.write_text(json.dumps(plan, indent=2))
        print(f"[{symbol}][BIAS-REFRESH] {old} → {live_bias} (Python MSS, no API call)")


def build_htf_context(symbol: str) -> dict:
    """Build the full HTF analysis dict to feed to Gemini."""
    context: dict[str, Any] = {"symbol": symbol, "timeframes": {}}

    for tf_name, (tf_const, count) in TIMEFRAMES.items():
        df = fetch_ohlc(symbol, tf_const, count)
        last = df.iloc[-1]

        tf_data = {
            "current_price": float(last["close"]),
            "last_candle": {
                "time":  str(last["time"]),
                "open":  float(last["open"]),
                "high":  float(last["high"]),
                "low":   float(last["low"]),
                "close": float(last["close"]),
            },
            "fvgs":   find_fvgs(df),
            "swings": find_swing_points(df),
        }
        # Send only last 10 candles raw — Gemini doesn't need all 500
        tf_data["recent_candles"] = df.tail(10).to_dict(orient="records")
        # Convert timestamps to strings for JSON
        for c in tf_data["recent_candles"]:
            c["time"] = str(c["time"])

        context["timeframes"][tf_name] = tf_data

    return context


# =============================================================================
# 3. NEWS FETCHING
# =============================================================================
def fetch_news_finnhub() -> list[dict]:
    """Fetch high-impact economic events from Finnhub. Returns [] on failure."""
    if not FINNHUB_API_KEY:
        return []
    try:
        today = datetime.now(timezone.utc).date()
        tomorrow = today + timedelta(days=1)
        url = (
            f"https://finnhub.io/api/v1/calendar/economic"
            f"?from={today}&to={tomorrow}&token={FINNHUB_API_KEY}"
        )
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json().get("economicCalendar", [])
        # Filter to high impact only (impact == "high")
        high_impact = [e for e in data if e.get("impact") == "high"]
        return high_impact
    except Exception as e:
        print(f"[NEWS] Finnhub fetch failed: {e}", file=sys.stderr)
        return []


def fetch_news_forexfactory() -> list[dict]:
    """
    Scrape ForexFactory calendar (fallback).

    ⚠️ FRAGILE: This breaks when ForexFactory changes their HTML.
    Last verified structure: 2024. If your bot stops detecting news,
    inspect the page and update the selectors below.
    """
    try:
        url = "https://www.forexfactory.com/calendar"
        headers = {"User-Agent": "Mozilla/5.0"}
        r = requests.get(url, headers=headers, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        events: list[dict] = []
        for row in soup.select("tr.calendar__row"):
            impact_cell = row.select_one(".calendar__impact span")
            if not impact_cell:
                continue
            impact_class = " ".join(impact_cell.get("class", []))
            if "high" not in impact_class.lower():
                continue
            time_text     = (row.select_one(".calendar__time") or {}).get_text(strip=True) if row.select_one(".calendar__time") else ""
            currency_text = (row.select_one(".calendar__currency") or {}).get_text(strip=True) if row.select_one(".calendar__currency") else ""
            event_text    = (row.select_one(".calendar__event") or {}).get_text(strip=True) if row.select_one(".calendar__event") else ""
            events.append({
                "time": time_text,
                "currency": currency_text,
                "event": event_text,
                "impact": "high",
            })
        return events
    except Exception as e:
        print(f"[NEWS] ForexFactory scrape failed: {e}", file=sys.stderr)
        return []


def fetch_high_impact_news() -> list[dict]:
    """Try Finnhub first, fall back to ForexFactory."""
    news = fetch_news_finnhub()
    if not news:
        news = fetch_news_forexfactory()
    return news


def is_news_within_2h(news: list[dict]) -> bool:
    """Check if any high-impact news is within next 2 hours (Finnhub format)."""
    now = datetime.now(timezone.utc)
    cutoff = now + timedelta(hours=2)
    for ev in news:
        # Finnhub uses 'time' as 'YYYY-MM-DD HH:MM:SS'
        t_str = ev.get("time", "")
        try:
            t = datetime.strptime(t_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
            if now <= t <= cutoff:
                return True
        except (ValueError, TypeError):
            continue
    return False


# =============================================================================
# 4. GEMINI PROMPT
# =============================================================================
SYSTEM_PROMPT = """\
You are an elite ICT (Inner Circle Trader) / Smart Money Concepts analyst.

You will receive:
1. Pre-computed Fair Value Gaps (FVGs) and swing points across multiple
   timeframes (Monthly, Weekly, Daily, H4, H1). These are FACTS — do NOT
   invent new ones, do NOT change their prices. You may only PICK among them.
2. Upcoming high-impact news.

Your task:
- Determine the daily directional bias (bullish/bearish) using
  weekly + daily structure.
- CRITICAL: You MUST pick bullish or bearish if ANY unmitigated FVGs exist
  in H4 or H1. NEVER return "neutral" when there are tradeable FVG zones.
  "neutral" is ONLY for when there are literally ZERO unmitigated FVGs.
- Pick ONE high-probability HTF Point of Interest (POI) — preferably an
  unmitigated H4 or H1 FVG aligned with HTF bias.
- Identify the engineered liquidity (swing high or low) near the POI.
  This is `target_liquidity`. The sweep is OPTIONAL — if no clear liquidity
  target exists, set target_liquidity to the nearest swing point.
- Set `wait_for_sweep` to false if the POI is a strong overlap zone
  (Breaker + FVG confluence) where sweep is not necessary.
- INVALIDATION: If between current price and the chosen POI there exists
  a NEWER unmitigated FVG, the original POI is invalid — pick the newer
  FVG instead, and update target_liquidity accordingly.
- NEWS ALIGNMENT: If high-impact news is within 2 hours, set
  wait_for_news=true so the volatility can sweep liquidity first.

OUTPUT: Respond with ONLY a single valid JSON object, no markdown fences,
no prose. Schema:

{
  "bias": "bullish" | "bearish" | "neutral",
  "poi_zone_high": <float>,
  "poi_zone_low":  <float>,
  "poi_timeframe": "H4" | "H1" | "D1",
  "poi_type":      "FVG" | "OB",
  "target_liquidity": <float>,
  "target_liquidity_type": "swing_high" | "swing_low",
  "wait_for_news": <bool>,
  "wait_for_sweep": <bool>,
  "reasoning": "<2-3 sentence summary of your logic>"
}

ONLY if there are literally ZERO unmitigated FVGs across ALL timeframes:
{"bias": "neutral", "poi_zone_high": 0, "poi_zone_low": 0,
 "poi_timeframe": "none", "poi_type": "none", "target_liquidity": 0,
 "target_liquidity_type": "none", "wait_for_news": false,
 "wait_for_sweep": false, "reasoning": "no unmitigated FVGs found"}
"""

SCALP_SYSTEM_PROMPT = """\
You are an elite ICT (Inner Circle Trader) scalping analyst specializing in
LTF momentum and precision entries on M1/M5 timeframes.

You will receive pre-computed FVGs and swing points across all timeframes.
Focus ONLY on H1 (session bias) and M15 (POI selection).
These levels are FACTS — do NOT invent new prices. You may only PICK among them.

Your task:
- Determine the SESSION bias (bullish/bearish) using H1 structure only.
- CRITICAL: You MUST pick bullish or bearish if ANY unmitigated FVGs exist
  in H1 or M15. NEVER return "neutral" when there are tradeable FVG zones.
  "neutral" is ONLY for when there are literally ZERO unmitigated FVGs.
- Pick ONE high-probability M15 Point of Interest (POI) — preferably an
  unmitigated M15 FVG that aligns with the H1 session bias.
- The POI zone must be TIGHT — reachable within the current session.
  If no unmitigated M15 FVG aligns with H1 bias, fall back to an H1 FVG.
- Identify the nearest M15 engineered liquidity (swing high or low) near
  the POI. This is `target_liquidity`. The sweep is OPTIONAL.
- Set `wait_for_sweep` to false if the POI is strong (overlap/confluence).
- NEWS ALIGNMENT: If high-impact news is within 2 hours, set wait_for_news=true
  so the spike can sweep liquidity before entry.

OUTPUT: Respond with ONLY a single valid JSON object, no markdown fences,
no prose. Schema:

{
  "bias": "bullish" | "bearish" | "neutral",
  "poi_zone_high": <float>,
  "poi_zone_low":  <float>,
  "poi_timeframe": "M15" | "H1",
  "poi_type":      "FVG" | "OB",
  "target_liquidity": <float>,
  "target_liquidity_type": "swing_high" | "swing_low",
  "wait_for_news": <bool>,
  "wait_for_sweep": <bool>,
  "reasoning": "<1-2 sentence summary of session logic>"
}

ONLY if there are literally ZERO unmitigated FVGs across ALL timeframes:
{"bias": "neutral", "poi_zone_high": 0, "poi_zone_low": 0,
 "poi_timeframe": "none", "poi_type": "none", "target_liquidity": 0,
 "target_liquidity_type": "none", "wait_for_news": false,
 "wait_for_sweep": false, "reasoning": "no unmitigated FVGs found"}
"""


def call_gemini(htf_context: dict, news: list[dict], wait_news_flag: bool,
                system_prompt: str = SYSTEM_PROMPT) -> dict:
    """Send context to Gemini and parse the JSON response."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY missing in .env")

    genai.configure(api_key=GEMINI_API_KEY)
    # Use flash for higher rate limits; switch to pro if you want deeper reasoning
    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=system_prompt,
        generation_config={
            "temperature": 0.2,         # low = more deterministic
            "response_mime_type": "application/json",
        },
    )

    user_payload = {
        "htf_context": htf_context,
        "high_impact_news_next_24h": news,
        "news_within_2h": wait_news_flag,
        "current_utc_time": datetime.now(timezone.utc).isoformat(),
    }
    user_message = json.dumps(user_payload, default=str)

    # Try twice — Gemini sometimes returns malformed JSON on first try
    for attempt in (1, 2):
        try:
            resp = model.generate_content(user_message)
            text = resp.text.strip()
            # Strip ```json fences if Gemini ignored response_mime_type
            if text.startswith("```"):
                text = text.strip("`").lstrip("json").strip()
            plan = json.loads(text)
            print(f"[GEMINI] ✅ Response parsed (attempt {attempt}):")
            print(f"         Bias              = {plan.get('bias')}")
            print(f"         POI zone          = [{plan.get('poi_zone_low')} – {plan.get('poi_zone_high')}]  ({plan.get('poi_timeframe')} {plan.get('poi_type')})")
            print(f"         Target liquidity  = {plan.get('target_liquidity')}  ({plan.get('target_liquidity_type')})")
            print(f"         Wait for news     = {plan.get('wait_for_news')}")
            print(f"         Reasoning         = {plan.get('reasoning', 'N/A')}")
            return plan
        except (json.JSONDecodeError, ValueError) as e:
            print(f"[GEMINI] ❌ Attempt {attempt} — JSON parse failed: {e}",
                  file=sys.stderr)
            if attempt == 2:
                raise

    raise RuntimeError("unreachable")


# =============================================================================
# 5. PLAN VALIDATION
# =============================================================================
def validate_plan(plan: dict, current_price: float) -> bool:
    """
    Sanity-check Gemini's output — FORCE EXECUTION MODE (v3).

    Philosophy: If the AI detected a clear Bias, WRITE THE PLAN.
    The sniper/scalper will stay "armed" and wait for price to approach.
    We only reject plans with truly broken data (missing keys, zero POI).

    Auto-fixes applied:
    - Auto-sort inverted poi_zone_low / poi_zone_high (min/max)
    - Accept any POI where bias is bullish/bearish — the execution
      engine handles proximity, not validation
    - Only hard-reject if POI values are zero/negative (hallucination)
    """
    print(f"[VALIDATE] Running sanity checks... (current_price={current_price})")

    # ── Check required keys exist ──────────────────────────────────────────
    required = {
        "bias", "poi_zone_high", "poi_zone_low",
        "target_liquidity", "wait_for_news",
    }
    missing = required - set(plan.keys())
    if missing:
        print(f"[VALIDATE] ❌ REJECTED — missing keys: {missing}", file=sys.stderr)
        return False

    if plan["bias"] not in {"bullish", "bearish", "neutral"}:
        print(f"[VALIDATE] ❌ REJECTED — invalid bias='{plan['bias']}'", file=sys.stderr)
        return False

    if plan["bias"] == "neutral":
        print(f"[VALIDATE] ✅ bias=neutral — no trade, but plan is valid")
        return True

    # ── Auto-sort POI zones (min/max) — NEVER reject for inversion ────────
    poi_low  = float(plan["poi_zone_low"])
    poi_high = float(plan["poi_zone_high"])

    sorted_low  = min(poi_low, poi_high)
    sorted_high = max(poi_low, poi_high)
    if sorted_low != poi_low or sorted_high != poi_high:
        print(f"[VALIDATE] 🔧 AUTO-SORT — [{poi_low}–{poi_high}] → [{sorted_low}–{sorted_high}]")
        plan["poi_zone_low"]  = sorted_low
        plan["poi_zone_high"] = sorted_high
        poi_low, poi_high = sorted_low, sorted_high

    # ── Only reject truly broken POIs (zero or negative) ──────────────────
    if poi_low <= 0 or poi_high <= 0:
        print(f"[VALIDATE] ❌ REJECTED — POI has zero/negative value: [{poi_low}–{poi_high}]",
              file=sys.stderr)
        return False

    # ── Sanity warning (never rejection) — log if POI is far from price ───
    if current_price > 0:
        distance_pct = abs(poi_low - current_price) / current_price * 100
        if distance_pct > 30:
            print(f"[VALIDATE] ⚠️ WARNING — POI [{poi_low}–{poi_high}] is {distance_pct:.1f}% "
                  f"from current price {current_price} — ACCEPTING (bias={plan['bias']})")
        elif distance_pct > 10:
            print(f"[VALIDATE] ℹ️ POI [{poi_low}–{poi_high}] is {distance_pct:.1f}% from price — normal range")

    print(f"[VALIDATE] ✅ Plan ACCEPTED. Bias={plan['bias']} | POI=[{poi_low}–{poi_high}] | "
          f"Target={plan.get('target_liquidity')} | Sniper will stay ARMED")
    return True


# =============================================================================
# 6. MAIN
# =============================================================================
def main(standalone: bool = True) -> int:
    """
    Generate plans for all symbols.

    Args:
        standalone: If True (default, running as script), this function
                    will call init_mt5() and mt5.shutdown(). If False
                    (called from main.py orchestrator), it assumes MT5
                    is already initialized and does NOT shut it down.
    """
    print("=" * 70)
    print(f"AI BRAIN — Daily Plan Generation @ {datetime.now(timezone.utc)}")
    print("=" * 70)

    if standalone:
        if not init_mt5():
            return 1

    try:
        # Fetch news once — shared across all symbols
        print("[1/4] Fetching high-impact news...")
        news = fetch_high_impact_news()
        wait_flag = is_news_within_2h(news)
        print(f"      Found {len(news)} high-impact events; "
              f"news_within_2h={wait_flag}")

        overall_rc = 0
        for sym in SYMBOLS:
            print(f"\n{'─' * 60}")
            print(f"[{sym}] Processing...")
            print(f"{'─' * 60}")

            # Build HTF context with REAL pre-computed FVGs/swings (shared for both plans)
            print(f"[{sym}][2/4] Building HTF context (MN1→M15)...")
            htf = build_htf_context(sym)
            current_price = htf["timeframes"]["H1"]["current_price"]
            print(f"[{sym}]       Current price: {current_price}")

            intraday_file = PLAN_DIR / f"intraday_plan_{sym}.json"
            scalp_file    = PLAN_DIR / f"scalp_plan_{sym}.json"

            # ── Intraday plan (H4/D1 POIs, MAGIC 1000) ──────────────────────
            if should_call_gemini(sym, "intraday"):
                print(f"[{sym}][3/4] Calling Gemini for INTRADAY plan...")
                intraday_plan = call_gemini(htf, news, wait_flag, SYSTEM_PROMPT)
                if validate_plan(intraday_plan, current_price):
                    intraday_plan["wait_for_news"] = wait_flag
                    intraday_plan["_generated_at"] = datetime.now(timezone.utc).isoformat()
                    intraday_plan["_symbol"] = sym
                    intraday_plan["_current_price_at_plan"] = current_price
                    intraday_file.write_text(json.dumps(intraday_plan, indent=2))
                    mark_gemini_called(sym, "intraday")
                    print(f"\n✅ [{sym}] Intraday plan saved → {intraday_file}")
                    print(json.dumps(intraday_plan, indent=2))
                else:
                    print(f"[{sym}][FATAL] Intraday plan failed validation — NOT writing file",
                          file=sys.stderr)
                    overall_rc = 2
            else:
                rem = remaining_cooldown_min(sym, "intraday")
                print(f"[{sym}][3/4] Gemini INTRADAY cooldown active — {rem:.0f}min remaining")
                print(f"[{sym}]       Refreshing bias from Python MSS instead...")
                refresh_bias_in_plan(intraday_file, sym)

            # ── Scalp plan (H1/M15 POIs, MAGIC 2000) ────────────────────────
            if should_call_gemini(sym, "scalp"):
                print(f"[{sym}][4/4] Calling Gemini for SCALP plan...")
                scalp_plan = call_gemini(htf, news, wait_flag, SCALP_SYSTEM_PROMPT)
                if validate_plan(scalp_plan, current_price):
                    scalp_plan["wait_for_news"] = wait_flag
                    scalp_plan["_generated_at"] = datetime.now(timezone.utc).isoformat()
                    scalp_plan["_symbol"] = sym
                    scalp_plan["_current_price_at_plan"] = current_price
                    scalp_file.write_text(json.dumps(scalp_plan, indent=2))
                    mark_gemini_called(sym, "scalp")
                    print(f"\n✅ [{sym}] Scalp plan saved → {scalp_file}")
                    print(json.dumps(scalp_plan, indent=2))
                else:
                    print(f"[{sym}][FATAL] Scalp plan failed validation — NOT writing file",
                          file=sys.stderr)
                    overall_rc = 2
            else:
                rem = remaining_cooldown_min(sym, "scalp")
                print(f"[{sym}][4/4] Gemini SCALP cooldown active — {rem:.0f}min remaining")
                print(f"[{sym}]       Refreshing bias from Python MSS instead...")
                refresh_bias_in_plan(scalp_file, sym)

        return overall_rc

    finally:
        if standalone:
            mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main(standalone=True))
