"""
=============================================================================
mt5_sniper.py - The LTF Execution Engine
=============================================================================

Purpose:
    - Read `daily_plan_<SYMBOL>.json` produced by ai_brain.py
    - Stream M15 / M5 candles from MT5
    - Wait for the exact ICT entry sequence:
        1. Price reaches POI zone
        2. Wick sweep of target_liquidity
        3. CHoCH (Change of Character) with displacement
        4. IFVG (Inversion FVG) validated by full-body close through old FVG
    - Place a BuyLimit / SellLimit order at the IFVG with 1% risk

=============================================================================
⚠️ CRITICAL WARNINGS - READ BEFORE RUNNING WITH REAL MONEY
=============================================================================

1. LIMIT ORDER MAY NEVER FILL:
   You asked for a Buy/Sell LIMIT at the IFVG. After CHoCH, price often
   continues in the new direction WITHOUT pulling back to the IFVG. Your
   order will sit unfilled, then the move ends, and you missed it.
   MITIGATION: We add `LIMIT_EXPIRY_MINUTES` — if the order isn't filled
   within N minutes after IFVG validation, we cancel it and reset state.

2. SPREAD AND SLIPPAGE NOT MODELED HERE:
   On XAUUSD/BTCUSD news/Asian session spreads explode (5-20+ points).
   A "5 point SL" can be triggered by spread alone.
   MITIGATION: We require minimum stop distance from broker spec.

3. STATE MACHINE FRAGILITY:
   The bot tracks state across many ticks. If it crashes mid-sequence
   and restarts, it forgets where it was. For demo this is fine; for
   real money you need state persistence (Redis/SQLite).

4. ICT CONCEPTS ARE SUBJECTIVE:
   "Displacement", "valid CHoCH", "valid sweep" — every trader defines
   these differently. Our definitions:
     - Sweep:        candle wick > swing, body closes back inside
     - CHoCH:        close beyond the most recent opposing swing point
     - Displacement: CHoCH candle range > 1.5x average of last 10 ranges
     - IFVG:         a body close completely through an opposing FVG
   You will need to TUNE these or you'll get bad fills.

5. MULTI-SYMBOL:
   This bot trades all symbols listed in SYMBOLS (.env) within a single
   process. Each symbol has its own independent BotContext and plan file
   (daily_plan_XAUUSD.json, daily_plan_XAGUSD.json, etc.).

6. NO PARTIAL TPs / TRAILING:
   Exit is one TP only. For real ICT trading you'd want partial profits
   at intermediate liquidity. Add later.

=============================================================================
"""

import json
import os
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Optional

import MetaTrader5 as mt5
import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
MT5_LOGIN    = int(os.getenv("MT5_LOGIN", "0"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER", "")
MT5_PATH     = os.getenv("MT5_PATH", "")
SYMBOLS      = [s.strip() for s in os.getenv("SYMBOLS", "XAUUSD").split(",") if s.strip()]
RISK_PCT     = float(os.getenv("RISK_PER_TRADE", "0.01"))  # 1%

# News override: set FORCE_IGNORE_NEWS=true in .env to trade through news events
FORCE_IGNORE_NEWS = os.getenv("FORCE_IGNORE_NEWS", "true").lower() in ("true", "1", "yes")

PLAN_DIR = Path(__file__).parent

# How often to poll
LOOP_INTERVAL_SEC = 5

# How long a pending limit order is allowed to sit unfilled
LIMIT_EXPIRY_MINUTES = 30

# Displacement threshold: CHoCH candle's range must be > N * avg(last 10)
DISPLACEMENT_MULTIPLIER = 1.5

# Magic numbers — intraday bot owns 1000-series; scalper owns 2000-series
MAGIC        = 1000
MAGIC_MARKET = 1001  # Aggressive: immediate market fill at IFVG validation
MAGIC_LIMIT  = 1002  # Conservative: limit retracement into IFVG
MAGIC_POS_A  = 1011  # Split Pos A — scalp/intraday leg (1.5 RR / 500-pip cap)
MAGIC_POS_B  = 1012  # Split Pos B — runner leg (HTF target)

# Spread guard: if (ask - bid) / point > this value, skip execution
# Gold/Silver spreads are volatile; 55pts is the hard ceiling
MAX_SPREAD_POINTS    = 55
SPREAD_WARN_THRESHOLD = 40  # Warning printed but execution continues

# Proximity trigger: fire market order if price is within N points of POI edge
PROXIMITY_POINTS = 150

# Ghost-trade protection: SL must be at least this many points from entry
MIN_SL_POINTS = int(os.getenv("MIN_SL_POINTS", "100"))

# Scalp leg hard cap: max TP distance in broker points (500 pips × 10 pts)
MAX_SCALP_TP_POINTS = int(os.getenv("MAX_SCALP_TP_POINTS", "5000"))

# State refresh every N hours — clears stale FVG context
REFRESH_INTERVAL_HOURS = 3

# Minimum wait after a SL hit before re-entry (ghost-trade cooldown)
ENTRY_COOLDOWN_SEC = int(os.getenv("ENTRY_COOLDOWN_SEC", "300"))

# ICT Killzones (UTC hours) — new setups only hunted inside these windows
LONDON_START, LONDON_END = 8,  11
NY_START,     NY_END     = 13, 16

# OTE (Optimal Trade Entry) — 61.8%–78.6% Fibonacci retracement zone
OTE_LOW  = 0.618
OTE_HIGH = 0.786

# ADR: if today's range fills this fraction of the daily ADR, tighten Pos B
ADR_TIGHTEN_PCT = 0.80


# =============================================================================
# RISK MANAGER
# =============================================================================
class RiskManager:
    """
    Dynamic risk ladder.

    Win streak → risk increases by STEP_RISK up to MAX_RISK.
    Any SL hit OR reaching MAX_RISK → immediate reset to BASE_RISK.
    State is persisted to disk so restarts don't lose the streak.
    """
    BASE_RISK  = 0.005   # 0.5 %
    STEP_RISK  = 0.005   # +0.5 % per consecutive win
    MAX_RISK   = 0.020   # 2.0 % ceiling

    _STATE_FILE = Path(__file__).parent / "risk_state_sniper.json"

    def __init__(self):
        self.win_streak   = 0
        self.current_risk = self.BASE_RISK
        self._load()

    def _load(self):
        if self._STATE_FILE.exists():
            try:
                d = json.loads(self._STATE_FILE.read_text())
                self.win_streak   = int(d.get("win_streak",   0))
                self.current_risk = float(d.get("current_risk", self.BASE_RISK))
                return
            except Exception:
                pass
        self.win_streak   = 0
        self.current_risk = self.BASE_RISK

    def _save(self):
        self._STATE_FILE.write_text(json.dumps({
            "win_streak":   self.win_streak,
            "current_risk": self.current_risk,
        }, indent=2))

    def get_risk(self) -> float:
        return self.current_risk

    def record_win(self):
        self.win_streak  += 1
        self.current_risk = min(
            self.BASE_RISK + self.win_streak * self.STEP_RISK, self.MAX_RISK
        )
        print(f"[RISK] WIN #{self.win_streak} — next risk={self.current_risk*100:.1f}%")
        self._save()

    def record_loss(self):
        self.win_streak   = 0
        self.current_risk = self.BASE_RISK
        print(f"[RISK] LOSS — risk reset to {self.BASE_RISK*100:.1f}%")
        self._save()

    def record_max_hit(self):
        """Called when current_risk reaches MAX_RISK — force reset."""
        if self.current_risk >= self.MAX_RISK:
            self.record_loss()


risk_manager = RiskManager()


# =============================================================================
# STATE MACHINE
# =============================================================================
class TradeState(Enum):
    IDLE                = "idle"                  # waiting for price to reach POI
    POI_TOUCHED         = "poi_touched"           # price entered POI zone
    SWEEP_DONE          = "sweep_done"            # liquidity swept by wick
    CHOCH_CONFIRMED     = "choch_confirmed"       # market structure shifted
    IFVG_VALIDATED      = "ifvg_validated"        # IFVG formed, order placed
    FVG_TAPPED          = "fvg_tapped"            # aligned FVG tapped, awaiting respect candle
    POSITION_OPEN       = "position_open"         # order filled
    DONE                = "done"                  # closed (TP/SL/cancel)


@dataclass
class BotContext:
    """Tracks all state across ticks for a single symbol."""
    symbol: str = "XAUUSD"
    state: TradeState = TradeState.IDLE
    plan: dict = field(default_factory=dict)
    plan_loaded_at: Optional[datetime] = None

    sweep_wick_price: Optional[float] = None
    sweep_time:       Optional[datetime] = None
    structure_pivot:  Optional[float] = None

    ifvg_low:  Optional[float] = None
    ifvg_high: Optional[float] = None

    pending_ticket:    Optional[int]      = None
    pending_placed_at: Optional[datetime] = None
    market_ticket:     Optional[int]      = None

    stop_loss_price:   Optional[float] = None
    take_profit_price: Optional[float] = None

    fvg_low:      Optional[float] = None
    fvg_high:     Optional[float] = None
    fvg_midpoint: Optional[float] = None

    # ── Split position tracking ──────────────────────────────────────────────
    pos_a_ticket:       Optional[int]      = None   # scalp leg
    pos_b_ticket:       Optional[int]      = None   # runner leg
    pos_entry:          Optional[float]    = None   # entry price (both legs same)
    pos_b_sl_upgraded:  bool               = False  # True once Pos A TP → Pos B SL raised
    direction_at_entry: Optional[str]      = None   # used for opposing-MSS exit

    # ── Timing ──────────────────────────────────────────────────────────────
    last_refresh:  Optional[datetime] = None
    cooldown_until: Optional[datetime] = None

    # ── Risk override ────────────────────────────────────────────────────────
    half_risk: bool = False   # True when firing outside KZ (HP setup only)


# =============================================================================
# MT5 HELPERS
# =============================================================================
def init_mt5() -> bool:
    args = dict(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if MT5_PATH:
        args["path"] = MT5_PATH
    if not mt5.initialize(**args):
        print(f"[MT5] init failed: {mt5.last_error()}", file=sys.stderr)
        return False
    for sym in SYMBOLS:
        if not mt5.symbol_select(sym, True):
            print(f"[MT5] symbol_select({sym}) failed", file=sys.stderr)
            return False
    return True


def get_candles(tf: int, count: int, symbol: str) -> Optional[pd.DataFrame]:
    rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
    if rates is None or len(rates) == 0:
        return None
    df = pd.DataFrame(rates)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    return df


def get_tick(symbol: str) -> Optional[mt5.Tick]:
    return mt5.symbol_info_tick(symbol)


# =============================================================================
# ICT DETECTION FUNCTIONS
# =============================================================================
def detect_wick_sweep(df: pd.DataFrame, target_price: float, direction: str) -> Optional[dict]:
    """
    Detect a wick sweep of target liquidity.

    BULLISH SETUP (we want to BUY): we expect a sweep of a swing LOW.
        - candle.low  < target_price  (wick pierces below)
        - candle.close > target_price (body closes back above)

    BEARISH SETUP (we want to SELL): we expect a sweep of a swing HIGH.
        - candle.high  > target_price (wick pierces above)
        - candle.close < target_price (body closes back below)

    We check the LAST CLOSED candle (not the live one) to avoid false triggers
    from intra-candle wicks that fill back.
    """
    if len(df) < 2:
        return None

    last = df.iloc[-2]  # last CLOSED candle

    if direction == "bullish":
        # Sweeping a swing low
        if last["low"] < target_price and last["close"] > target_price:
            return {
                "wick_price": float(last["low"]),
                "time": last["time"].to_pydatetime(),
                "candle": last.to_dict(),
            }
    elif direction == "bearish":
        # Sweeping a swing high
        if last["high"] > target_price and last["close"] < target_price:
            return {
                "wick_price": float(last["high"]),
                "time": last["time"].to_pydatetime(),
                "candle": last.to_dict(),
            }
    return None


def detect_choch_with_displacement(df: pd.DataFrame, direction: str,
                                   sweep_time: datetime) -> Optional[float]:
    """
    Detect CHoCH (change of character) with displacement after a sweep.

    BULLISH CHoCH: after a low sweep, we want a candle that closes ABOVE
    the most recent swing high since the sweep.

    BEARISH CHoCH: after a high sweep, we want a candle that closes BELOW
    the most recent swing low since the sweep.

    DISPLACEMENT: the breakout candle's range must be > 1.5x average range
    of the last 10 candles. This filters out weak breaks.

    Returns the structure_pivot price if confirmed, else None.
    """
    if len(df) < 11:
        return None

    # sweep_time is already UTC-aware (to_pydatetime() from a utc=True DataFrame),
    # so drop the tz= argument — passing both a tz-aware object AND tz= raises ValueError.
    df_after = df[df["time"] > pd.Timestamp(sweep_time)]
    if len(df_after) < 2:
        return None

    # Average range of last 10 candles for displacement check
    last_10 = df.tail(11).iloc[:-1]  # exclude live candle
    avg_range = float((last_10["high"] - last_10["low"]).mean())

    last_closed = df.iloc[-2]
    last_range = float(last_closed["high"] - last_closed["low"])

    if last_range < DISPLACEMENT_MULTIPLIER * avg_range:
        return None  # not enough displacement

    if direction == "bullish":
        # Find highest high BEFORE the breakout candle but after the sweep
        prior = df_after.iloc[:-1]
        if len(prior) == 0:
            return None
        pivot = float(prior["high"].max())
        if last_closed["close"] > pivot:
            return pivot
    elif direction == "bearish":
        prior = df_after.iloc[:-1]
        if len(prior) == 0:
            return None
        pivot = float(prior["low"].min())
        if last_closed["close"] < pivot:
            return pivot

    return None


def find_opposing_fvg(df: pd.DataFrame, direction: str) -> Optional[dict]:
    """
    Find the most recent OPPOSING FVG that price might invert.

    For a BULLISH setup (post low-sweep, post CHoCH up), we look for a recent
    BEARISH FVG that price will close through. Once a candle body closes
    fully ABOVE that bearish FVG, it becomes an Inversion FVG (IFVG) — a
    bullish entry zone.

    Symmetric for bearish.
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    t = df["time"].to_numpy()

    candidates = []
    for i in range(2, len(df)):
        if direction == "bullish":
            # Looking for a BEARISH FVG (opposing): l[i-2] > h[i]
            if l[i - 2] > h[i]:
                candidates.append({"low": float(h[i]), "high": float(l[i - 2]),
                                   "time": str(t[i]), "type": "bearish"})
        elif direction == "bearish":
            # Looking for a BULLISH FVG (opposing): h[i-2] < l[i]
            if h[i - 2] < l[i]:
                candidates.append({"low": float(h[i - 2]), "high": float(l[i]),
                                   "time": str(t[i]), "type": "bullish"})

    return candidates[-1] if candidates else None


def is_ifvg_validated(df: pd.DataFrame, fvg: dict, direction: str) -> bool:
    """
    Check if the most recent CLOSED candle has a body that closes
    completely THROUGH the opposing FVG.

    BULLISH: candle body (open & close) both ABOVE fvg["high"]
    BEARISH: candle body (open & close) both BELOW fvg["low"]
    """
    last = df.iloc[-2]
    body_low  = min(last["open"], last["close"])
    body_high = max(last["open"], last["close"])

    if direction == "bullish":
        return body_low > fvg["high"]
    elif direction == "bearish":
        return body_high < fvg["low"]
    return False


def find_aligned_fvg(df: pd.DataFrame, direction: str) -> Optional[dict]:
    """
    Find the most recent unmitigated FVG aligned with the bias direction.
    Bullish bias → bullish FVG (h[i-2] < l[i]).
    Bearish bias → bearish FVG (l[i-2] > h[i]).
    Filters out FVGs already fully blown through so only tapable zones remain.
    """
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    candidates = []
    for i in range(2, len(df)):
        if direction == "bullish":
            if h[i - 2] < l[i]:
                gap_low, gap_high = float(h[i - 2]), float(l[i])
                blown = bool((l[i + 1:] < gap_low).any()) if i + 1 < len(df) else False
                if not blown:
                    candidates.append({
                        "low": gap_low, "high": gap_high,
                        "midpoint": (gap_low + gap_high) / 2,
                        "type": "bullish",
                    })
        elif direction == "bearish":
            if l[i - 2] > h[i]:
                gap_low, gap_high = float(h[i]), float(l[i - 2])
                blown = bool((h[i + 1:] > gap_high).any()) if i + 1 < len(df) else False
                if not blown:
                    candidates.append({
                        "low": gap_low, "high": gap_high,
                        "midpoint": (gap_low + gap_high) / 2,
                        "type": "bearish",
                    })
    return candidates[-1] if candidates else None


def detect_respect_candle(df: pd.DataFrame, fvg: dict, direction: str) -> bool:
    """
    Confirm FVG respect on the last closed candle.

    Bullish: wick dips into FVG (low <= fvg_high), candle closes bullishly
             (close > open), and close >= FVG midpoint (strong rejection).
    Bearish: wick pokes into FVG (high >= fvg_low), candle closes bearishly
             (close < open), and close <= FVG midpoint (strong rejection).
    """
    if len(df) < 2:
        return False
    last = df.iloc[-2]
    mid  = fvg["midpoint"]
    if direction == "bullish":
        return (last["low"]   <= fvg["high"]       # wick dipped into FVG
                and last["close"] >  last["open"]  # bullish candle
                and last["close"] >= mid)           # strong close above midpoint
    elif direction == "bearish":
        return (last["high"]  >= fvg["low"]        # wick poked into FVG
                and last["close"] <  last["open"]  # bearish candle
                and last["close"] <= mid)           # strong close below midpoint
    return False


# =============================================================================
# REAL-TIME BIAS, M1 MSS & TRADE MANAGEMENT
# =============================================================================

def calculate_bias(symbol: str) -> str:
    """
    Real-time H1 bias — Python market structure, no LLM, no caching.
    Re-computed every loop iteration to catch MSS the Gemini plan missed.

    Returns "bullish", "bearish", or "neutral".
    """
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_H1, 0, 100)
    if rates is None or len(rates) < 15:
        return "neutral"
    df = pd.DataFrame(rates)
    h  = df["high"].to_numpy()
    l  = df["low"].to_numpy()
    c  = df["close"].to_numpy()
    o  = df["open"].to_numpy()
    last_close = c[-2]
    recent_swing_high: Optional[float] = None
    recent_swing_low:  Optional[float] = None
    for i in range(len(df) - 3, 2, -1):
        if recent_swing_high is None:
            if h[i] >= h[i-1] and h[i] >= h[i-2] and h[i] >= h[i+1] and h[i] >= h[i+2]:
                recent_swing_high = float(h[i])
        if recent_swing_low is None:
            if l[i] <= l[i-1] and l[i] <= l[i-2] and l[i] <= l[i+1] and l[i] <= l[i+2]:
                recent_swing_low = float(l[i])
        if recent_swing_high is not None and recent_swing_low is not None:
            break
    if recent_swing_low is not None and last_close < recent_swing_low:
        print(f"[{symbol}][BIAS] Bearish MSS — H1 close {last_close:.5f} < swing low {recent_swing_low:.5f}")
        return "bearish"
    if recent_swing_high is not None and last_close > recent_swing_high:
        print(f"[{symbol}][BIAS] Bullish MSS — H1 close {last_close:.5f} > swing high {recent_swing_high:.5f}")
        return "bullish"
    bear = sum(1 for i in range(-11, -1) if c[i] < o[i])
    if bear >= 7:
        return "bearish"
    if bear <= 3:
        return "bullish"
    return "neutral"


def detect_m1_mss(df: pd.DataFrame, direction: str) -> bool:
    """
    M1 Market Structure Shift with displacement.

    Bearish: last closed M1 candle is a bearish bar that closes below the
             previous candle's low AND its range >= 1.2 × avg M1 range.
    Bullish: symmetric — close above previous high with displacement.
    """
    if len(df) < 5:
        return False
    last = df.iloc[-2]
    prev = df.iloc[-3]
    recent    = df.tail(11).iloc[:-1]
    avg_range = float((recent["high"] - recent["low"]).mean())
    last_range = float(last["high"] - last["low"])
    if avg_range <= 0:
        return False
    displaced = last_range >= 1.2 * avg_range
    if direction == "bearish":
        return (displaced
                and float(last["close"]) < float(prev["low"])
                and float(last["close"]) < float(last["open"]))
    elif direction == "bullish":
        return (displaced
                and float(last["close"]) > float(prev["high"])
                and float(last["close"]) > float(last["open"]))
    return False


def modify_sl(ticket: int, new_sl: float, tp: float, symbol: str) -> bool:
    """Move an open position's SL without touching TP."""
    info = mt5.symbol_info(symbol)
    if info is None:
        return False
    request = {
        "action":   mt5.TRADE_ACTION_SLTP,
        "position": ticket,
        "sl":       round(new_sl, info.digits),
        "tp":       round(tp,     info.digits),
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        print(f"[{symbol}][BE] ❌ SL modify failed retcode={code}", file=sys.stderr)
        return False
    print(f"[{symbol}][BE] ✅ SL moved to break-even {new_sl:.5f}")
    return True


def monitor_trade_management(ctx: BotContext, sym: str):
    """
    Break-even management for 0.01 lot (no lot splitting).

    When price reaches 1.5 RR → move SL to entry (break-even).
    TP remains at 3.0 RR — position runs free from there.
    Called every tick while state == POSITION_OPEN.
    """
    if ctx.state != TradeState.POSITION_OPEN:
        return

    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return

    positions = mt5.positions_get(symbol=sym) or []
    ours = [p for p in positions if p.magic in (MAGIC_MARKET, MAGIC_LIMIT)]

    for pos in ours:
        entry = float(pos.price_open)
        sl    = float(pos.sl)
        tp    = float(pos.tp)
        if sl <= 0 or entry <= 0:
            continue

        if pos.type == mt5.POSITION_TYPE_BUY:
            stop_dist  = entry - sl
            if stop_dist <= 0:
                continue
            be_trigger = entry + 1.5 * stop_dist
            current    = float(tick.bid)
            if sl >= entry:   # SL already at or above break-even
                continue
            if current >= be_trigger:
                print(f"[{sym}][BE] 1.5 RR hit (bid={current:.5f} >= trigger={be_trigger:.5f}) "
                      f"— moving SL to entry {entry:.5f}")
                modify_sl(pos.ticket, entry, tp, sym)

        elif pos.type == mt5.POSITION_TYPE_SELL:
            stop_dist  = sl - entry
            if stop_dist <= 0:
                continue
            be_trigger = entry - 1.5 * stop_dist
            current    = float(tick.ask)
            if sl <= entry:   # SL already at or below break-even
                continue
            if current <= be_trigger:
                print(f"[{sym}][BE] 1.5 RR hit (ask={current:.5f} <= trigger={be_trigger:.5f}) "
                      f"— moving SL to entry {entry:.5f}")
                modify_sl(pos.ticket, entry, tp, sym)


# =============================================================================
# KILLZONES, STATE REFRESH, OPPOSING MSS
# =============================================================================

def in_killzone() -> bool:
    """True if current UTC time is inside London or New York killzone."""
    h = datetime.now(timezone.utc).hour
    return (LONDON_START <= h < LONDON_END) or (NY_START <= h < NY_END)


def needs_state_refresh(ctx: BotContext) -> bool:
    """True if 3 hours have elapsed since the last structure refresh."""
    if ctx.last_refresh is None:
        return True
    elapsed_h = (datetime.now(timezone.utc) - ctx.last_refresh).total_seconds() / 3600
    return elapsed_h >= REFRESH_INTERVAL_HOURS


def detect_opposing_mss(df: pd.DataFrame, direction: str) -> bool:
    """
    Returns True if the market has structurally reversed against an open trade.

    For a short (bearish) position: opposing MSS = last closed M5 bar closes
    above the highest high of the past 20 bars (bullish break of structure).
    For a long (bullish) position: symmetric.

    Uses a loose 20-bar lookback to avoid premature exits on noise.
    """
    if len(df) < 22:
        return False
    last_close = float(df.iloc[-2]["close"])
    window = df.iloc[-22:-2]   # 20 bars before the last closed bar
    if direction == "bearish":
        return last_close > float(window["high"].max())
    elif direction == "bullish":
        return last_close < float(window["low"].min())
    return False


# =============================================================================
# ICT ADVANCED CONTEXT — OTE / PO3 / ADR / PREV DAY HL
# =============================================================================

def check_ote(sweep_price: float, choch_pivot: float, direction: str, price: float) -> bool:
    """
    OTE zone: 61.8%–78.6% Fibonacci retracement of the sweep→CHoCH move.

    Bullish: sweep_price = swing low swept; choch_pivot = prior swing high broken.
             Entry zone = [pivot - 0.786*rng, pivot - 0.618*rng]
    Bearish: sweep_price = swing high swept; choch_pivot = prior swing low broken.
             Entry zone = [pivot + 0.618*rng, pivot + 0.786*rng]
    """
    if direction == "bullish":
        rng = choch_pivot - sweep_price
        if rng <= 0:
            return False
        ote_lo = choch_pivot - OTE_HIGH * rng
        ote_hi = choch_pivot - OTE_LOW  * rng
    else:
        rng = sweep_price - choch_pivot
        if rng <= 0:
            return False
        ote_lo = choch_pivot + OTE_LOW  * rng
        ote_hi = choch_pivot + OTE_HIGH * rng
    in_ote = ote_lo <= price <= ote_hi
    tag = "✅" if in_ote else "—"
    print(f"  [OTE] zone=[{ote_lo:.5f}–{ote_hi:.5f}] price={price:.5f} {tag}")
    return in_ote


def get_midnight_open(symbol: str) -> Optional[float]:
    """00:00 EST (= 05:00 UTC) open price — PO3 Midnight Open reference."""
    now = datetime.now(timezone.utc)
    mo_utc = now.replace(hour=5, minute=0, second=0, microsecond=0)
    if now.hour < 5:
        mo_utc -= timedelta(days=1)
    rates = mt5.copy_rates_from(symbol, mt5.TIMEFRAME_H1, mo_utc, 1)
    if rates is None or len(rates) == 0:
        return None
    return float(rates[0]["open"])


def get_prev_day_hl(symbol: str) -> tuple:
    """(prev_day_high, prev_day_low) — primary HTF targets."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, 1)
    if rates is None or len(rates) == 0:
        return (0.0, 0.0)
    return (float(rates[0]["high"]), float(rates[0]["low"]))


def get_daily_adr(symbol: str, lookback: int = 14) -> float:
    """Average Daily Range (high − low) over the last N completed days."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 1, lookback)
    if rates is None or len(rates) == 0:
        return 0.0
    return float(sum(r["high"] - r["low"] for r in rates) / len(rates))


def get_today_range(symbol: str) -> float:
    """Today's high − low so far (live incomplete daily candle)."""
    rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_D1, 0, 1)
    if rates is None or len(rates) == 0:
        return 0.0
    return float(rates[0]["high"] - rates[0]["low"])


def is_high_probability_setup(ctx: BotContext, df: pd.DataFrame,
                               direction: str, price: float) -> bool:
    """
    High-probability outside-KZ criteria: Sweep confirmed + aligned FVG touch + OTE.
    All three must be true to allow a half-risk entry outside a killzone.
    """
    has_sweep = ctx.sweep_wick_price is not None
    fvg = find_aligned_fvg(df, direction)
    has_fvg = fvg is not None and fvg["low"] <= price <= fvg["high"]
    has_ote = False
    if ctx.sweep_wick_price and ctx.structure_pivot:
        has_ote = check_ote(ctx.sweep_wick_price, ctx.structure_pivot, direction, price)
    return has_sweep and has_fvg and has_ote


# =============================================================================
# RISK MANAGEMENT & ORDER PLACEMENT
# =============================================================================
def calculate_lot_size(stop_distance_price: float, risk_pct: float, symbol: str) -> float:
    """
    Calculate lot size such that loss at SL == risk_pct * account balance.

    lot = (balance * risk_pct) / (stop_distance_price * tick_value_per_point)

    Note: This uses tick_value, which depends on the symbol contract size
    and your account currency. We use mt5.symbol_info() and account_info()
    rather than hardcoding.
    """
    info = mt5.symbol_info(symbol)
    acc  = mt5.account_info()
    if info is None or acc is None:
        print(f"[{symbol}][LOT] ❌ symbol_info or account_info is None", file=sys.stderr)
        return 0.0

    risk_amount = acc.balance * risk_pct
    tick_value  = info.trade_tick_value
    tick_size   = info.trade_tick_size
    print(f"[{symbol}][LOT] Balance={acc.balance:.2f} | RiskAmt={risk_amount:.2f} | "
          f"TickValue={tick_value} | TickSize={tick_size} | StopDist={stop_distance_price:.5f}")

    if tick_size <= 0 or tick_value <= 0:
        print(f"[{symbol}][LOT] ❌ Invalid tick_size={tick_size} or tick_value={tick_value}", file=sys.stderr)
        return 0.0

    # Loss per 1.0 lot if price moves stop_distance_price against us
    loss_per_lot = (stop_distance_price / tick_size) * tick_value
    if loss_per_lot <= 0:
        print(f"[{symbol}][LOT] ❌ loss_per_lot={loss_per_lot:.4f} is zero or negative", file=sys.stderr)
        return 0.0

    raw_lot = risk_amount / loss_per_lot
    step = info.volume_step
    
    # ── HARD LIMIT FOR TESTING ──
    # The broker rejected 100 lots. We are capping this to 0.01 for testing.
    DEFAULT_LOT = 0.01
    
    lot  = max(info.volume_min, min(DEFAULT_LOT, round(raw_lot / step) * step))
    print(f"[{symbol}][LOT] LossPerLot={loss_per_lot:.2f} | RawLot={raw_lot:.4f} | "
          f"FinalLot={lot} (min={info.volume_min} max={DEFAULT_LOT} step={step})")
    return float(lot)


def place_limit_order(direction: str, price: float, sl: float, tp: float,
                      lot: float, symbol: str,
                      magic: int = MAGIC_LIMIT,
                      comment: str = "ICT_Limit") -> Optional[int]:
    """
    Place BuyLimit (bullish) or SellLimit (bearish) at IFVG price.
    Returns ticket on success, None on failure.
    """
    info = mt5.symbol_info(symbol)
    if info is None:
        return None

    digits = info.digits
    price = round(price, digits)
    sl    = round(sl, digits)
    tp    = round(tp, digits)

    order_type = mt5.ORDER_TYPE_BUY_LIMIT if direction == "bullish" else mt5.ORDER_TYPE_SELL_LIMIT

    request = {
        "action":       mt5.TRADE_ACTION_PENDING,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_RETURN,
    }

    order_type_str = "BUY_LIMIT" if direction == "bullish" else "SELL_LIMIT"
    print(f"[{symbol}][ORDER] ► Calling mt5.order_send() [CONSERVATIVE/LIMIT] with:")
    print(f"         Type     = {order_type_str}")
    print(f"         Symbol   = {symbol}")
    print(f"         Volume   = {lot}")
    print(f"         Price    = {price}")
    print(f"         SL       = {sl}")
    print(f"         TP       = {tp}")
    print(f"         Magic    = {magic}")
    print(f"         Comment  = {comment}")
    print(f"         Filling  = ORDER_FILLING_RETURN")

    result = mt5.order_send(request)
    if result is None:
        print(f"[{symbol}][ORDER] ❌ order_send returned None! MT5 error: {mt5.last_error()}",
              file=sys.stderr)
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}][ORDER] ❌ BROKER REJECTED ORDER!", file=sys.stderr)
        print(f"         retcode = {result.retcode}  comment = '{result.comment}'", file=sys.stderr)
        print(f"         Common codes: 10004=Requote 10006=Rejected 10014=InvalidVolume "
              f"10015=InvalidPrice 10016=Invalid SL/TP 10019=NoMoney 10033=PendingOrdersLimit",
              file=sys.stderr)
        return None

    print(f"[{symbol}][ORDER] ✅ ORDER PLACED! ticket={result.order} | {order_type_str} @ {price} | SL={sl} | TP={tp} | lot={lot}")
    return int(result.order)


def place_market_order(direction: str, sl: float, tp: float,
                       lot: float, symbol: str) -> Optional[int]:
    """
    Place an immediate market order (Aggressive entry at IFVG validation price).
    Returns ticket on success, None on failure.
    """
    info = mt5.symbol_info(symbol)
    tick = mt5.symbol_info_tick(symbol)
    if info is None or tick is None:
        return None

    digits = info.digits
    sl = round(sl, digits)
    tp = round(tp, digits)

    if direction == "bullish":
        order_type = mt5.ORDER_TYPE_BUY
        price = round(tick.ask, digits)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price = round(tick.bid, digits)

    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       symbol,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        MAGIC_MARKET,
        "comment":      "ICT_Market",
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }

    order_type_str = "BUY_MARKET" if direction == "bullish" else "SELL_MARKET"
    print(f"[{symbol}][ORDER] ► Calling mt5.order_send() [AGGRESSIVE/MARKET] with:")
    print(f"         Type     = {order_type_str}")
    print(f"         Symbol   = {symbol}")
    print(f"         Volume   = {lot}")
    print(f"         Price    = {price}")
    print(f"         SL       = {sl}")
    print(f"         TP       = {tp}")
    print(f"         Magic    = {MAGIC_MARKET}")
    print(f"         Comment  = ICT_Market")

    result = mt5.order_send(request)
    if result is None:
        print(f"[{symbol}][ORDER] ❌ Market order_send returned None! MT5 error: {mt5.last_error()}",
              file=sys.stderr)
        return None
    if result.retcode != mt5.TRADE_RETCODE_DONE:
        print(f"[{symbol}][ORDER] ❌ BROKER REJECTED MARKET ORDER!", file=sys.stderr)
        print(f"         retcode = {result.retcode}  comment = '{result.comment}'", file=sys.stderr)
        return None

    print(f"[{symbol}][ORDER] ✅ MARKET ORDER FILLED! ticket={result.order} | {order_type_str} @ {price} | SL={sl} | TP={tp} | lot={lot}")
    return int(result.order)


def cancel_order(ticket: int) -> bool:
    request = {
        "action": mt5.TRADE_ACTION_REMOVE,
        "order":  ticket,
    }
    res = mt5.order_send(request)
    return res is not None and res.retcode == mt5.TRADE_RETCODE_DONE


def order_still_pending(ticket: int) -> bool:
    orders = mt5.orders_get(ticket=ticket)
    return orders is not None and len(orders) > 0


def position_exists_for_magic(magic: int, symbol: str) -> bool:
    """Return True if at least one open position carries the given magic number."""
    positions = mt5.positions_get(symbol=symbol)
    if positions is None:
        return False
    return any(p.magic == magic for p in positions)


# =============================================================================
# SPLIT TRADE EXECUTION & POSITION MONITORING
# =============================================================================

def _place_split_order(direction: str, sl: float, tp: float, lot: float,
                       sym: str, magic: int, comment: str) -> Optional[int]:
    """Place one leg of a split trade as a market order."""
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None or tick is None:
        return None
    sl = round(sl, info.digits)
    tp = round(tp, info.digits)
    if direction == "bullish":
        order_type = mt5.ORDER_TYPE_BUY
        price      = round(tick.ask, info.digits)
    else:
        order_type = mt5.ORDER_TYPE_SELL
        price      = round(tick.bid, info.digits)
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       sym,
        "volume":       lot,
        "type":         order_type,
        "price":        price,
        "sl":           sl,
        "tp":           tp,
        "deviation":    20,
        "magic":        magic,
        "comment":      comment,
        "type_time":    mt5.ORDER_TIME_GTC,
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        print(f"[{sym}][SPLIT] ❌ {comment} failed retcode={code}", file=sys.stderr)
        return None
    print(f"[{sym}][SPLIT] ✅ {comment} ticket={result.order} @ {price:.5f} "
          f"SL={sl:.5f} TP={tp:.5f}")
    return int(result.order)


def execute_split_trade(ctx: BotContext, direction: str, entry: float,
                        sl: float, plan: dict, sym: str,
                        risk_multiplier: float = 1.0) -> bool:
    """
    Open Position A (scalp) + Position B (runner) simultaneously.

    Pos A TP = min(1.5 RR, MAX_SCALP_TP_POINTS).
    Pos B TP = nearest of: plan target_liquidity, prev day High/Low, capped by ADR remaining.
    risk_multiplier = 0.5 for outside-KZ half-risk entries.
    """
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None or tick is None:
        return False

    point     = info.point
    stop_dist = abs(entry - sl)

    # ── Ghost-trade protection ───────────────────────────────────────────────
    if stop_dist < MIN_SL_POINTS * point:
        print(f"[{sym}][GHOST] SL too tight "
              f"({stop_dist/point:.0f}pts < {MIN_SL_POINTS}pts) — skipped",
              file=sys.stderr)
        return False
    spread = tick.ask - tick.bid
    if stop_dist < 2.0 * spread:
        print(f"[{sym}][GHOST] SL ({stop_dist:.5f}) inside 2× spread "
              f"({2*spread:.5f}) — skipped", file=sys.stderr)
        return False

    # ── Pos A TP: min(1.5 RR, hard cap) ────────────────────────────────────
    cap_dist = MAX_SCALP_TP_POINTS * point
    if direction == "bullish":
        tp_a = min(entry + 1.5 * stop_dist, entry + cap_dist)
    else:
        tp_a = max(entry - 1.5 * stop_dist, entry - cap_dist)

    # ── Pos B TP: nearest valid HTF target (plan + prev day HL), ADR-capped ──
    tp_b_plan  = float(plan.get("target_liquidity", 0))
    prev_high, prev_low = get_prev_day_hl(sym)
    daily_adr   = get_daily_adr(sym)
    today_rng   = get_today_range(sym)
    adr_remain  = max(daily_adr - today_rng, stop_dist * 1.5) if daily_adr > 0 \
                  else stop_dist * 3.0

    if direction == "bullish":
        candidates = [x for x in [tp_b_plan, prev_high] if x > entry]
        tp_b = min(candidates) if candidates else entry + 3.0 * stop_dist
        tp_b = min(tp_b, entry + adr_remain)   # ADR cap
    else:
        candidates = [x for x in [tp_b_plan, prev_low] if 0 < x < entry]
        tp_b = max(candidates) if candidates else entry - 3.0 * stop_dist
        tp_b = max(tp_b, entry - adr_remain)   # ADR cap

    # ── Lot sizing (risk_multiplier < 1.0 for outside-KZ HP entries) ────────
    risk = risk_manager.get_risk() * risk_multiplier
    if risk_manager.get_risk() >= RiskManager.MAX_RISK:
        risk_manager.record_max_hit()
        risk = risk_manager.get_risk() * risk_multiplier
    lot = calculate_lot_size(stop_dist, risk / 2, sym)
    if lot <= 0:
        print(f"[{sym}][SPLIT] Lot size = 0 — skipped", file=sys.stderr)
        return False

    kz_tag = " (HALF-RISK outside KZ)" if risk_multiplier < 1.0 else ""
    print(f"[{sym}][SPLIT] direction={direction} entry={entry:.5f} "
          f"SL={sl:.5f} stop_dist={stop_dist/point:.0f}pts "
          f"risk={risk*100:.1f}%{kz_tag}")
    print(f"[{sym}][SPLIT] Pos A TP={tp_a:.5f} (1.5RR/{MAX_SCALP_TP_POINTS}pts cap) "
          f"Pos B TP={tp_b:.5f} (ADR remain={adr_remain/point:.0f}pts)")

    ticket_a = _place_split_order(direction, sl, tp_a, lot, sym, MAGIC_POS_A, "POS_A")
    if ticket_a is None:
        return False

    ticket_b = _place_split_order(direction, sl, tp_b, lot, sym, MAGIC_POS_B, "POS_B")
    if ticket_b is None:
        print(f"[{sym}][SPLIT] ⚠️ Pos B failed — running single Pos A leg")

    ctx.pos_a_ticket       = ticket_a
    ctx.pos_b_ticket       = ticket_b
    ctx.pos_entry          = entry
    ctx.pos_b_sl_upgraded  = False
    ctx.direction_at_entry = direction
    ctx.state              = TradeState.POSITION_OPEN
    return True


def _get_deal_profit(position_ticket: int) -> Optional[float]:
    """Sum the profit of all OUT-deals for a closed position ticket."""
    if position_ticket is None:
        return None
    try:
        deals = mt5.history_deals_get(position=position_ticket)
        if not deals:
            return None
        out_profit = sum(
            d.profit for d in deals if d.entry == mt5.DEAL_ENTRY_OUT
        )
        return float(out_profit)
    except Exception:
        return None


def _close_position(ticket: int, sym: str):
    """Close an open position at market immediately."""
    positions = mt5.positions_get(ticket=ticket)
    if not positions:
        return
    pos  = positions[0]
    info = mt5.symbol_info(sym)
    tick = mt5.symbol_info_tick(sym)
    if info is None or tick is None:
        return
    if pos.type == mt5.POSITION_TYPE_BUY:
        order_type = mt5.ORDER_TYPE_SELL
        price      = round(tick.bid, info.digits)
    else:
        order_type = mt5.ORDER_TYPE_BUY
        price      = round(tick.ask, info.digits)
    request = {
        "action":       mt5.TRADE_ACTION_DEAL,
        "symbol":       sym,
        "volume":       pos.volume,
        "type":         order_type,
        "price":        price,
        "position":     ticket,
        "deviation":    20,
        "magic":        pos.magic,
        "comment":      "FORCE_CLOSE",
        "type_filling": mt5.ORDER_FILLING_IOC,
    }
    result = mt5.order_send(request)
    if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
        code = result.retcode if result else "None"
        print(f"[{sym}][CLOSE] ❌ Force close failed retcode={code}", file=sys.stderr)
    else:
        print(f"[{sym}][CLOSE] ✅ Position {ticket} closed @ {price:.5f}")


def monitor_positions(ctx: BotContext, sym: str, m5: Optional[pd.DataFrame] = None):
    """
    Manages split Pos A / Pos B each tick while state == POSITION_OPEN.

    Rules:
    1. Pos A TP hit  → upgrade Pos B SL to entry + 1.0 RR profit; record WIN.
    2. Pos A SL hit  → force-close Pos B; record LOSS; start entry cooldown.
    3. Both closed   → determine outcome, update risk ladder, apply cooldown.
    4. Opposing MSS  → force-close both, record LOSS, apply cooldown.
    5. 500-pip move  → if Pos B SL not yet upgraded, secure 1.5 RR on Pos B.
    """
    if ctx.state != TradeState.POSITION_OPEN:
        return

    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return
    info  = mt5.symbol_info(sym)
    point = info.point if info else 0.0001

    positions = mt5.positions_get(symbol=sym) or []
    pos_a = next((p for p in positions if p.magic == MAGIC_POS_A), None)
    pos_b = next((p for p in positions if p.magic == MAGIC_POS_B), None)

    # ── Also track legacy single-leg magic numbers ───────────────────────────
    legacy = [p for p in positions if p.magic in (MAGIC_MARKET, MAGIC_LIMIT)]

    def _apply_cooldown():
        ctx.cooldown_until = datetime.now(timezone.utc) + timedelta(seconds=ENTRY_COOLDOWN_SEC)

    # ── Both split legs closed ───────────────────────────────────────────────
    if pos_a is None and pos_b is None and not legacy:
        a_profit = _get_deal_profit(ctx.pos_a_ticket)
        b_profit = _get_deal_profit(ctx.pos_b_ticket)
        total    = (a_profit or 0) + (b_profit or 0)
        if total > 0:
            risk_manager.record_win()
        else:
            risk_manager.record_loss()
            ctx.cooldown_until = datetime.now(timezone.utc) + \
                                 timedelta(seconds=ENTRY_COOLDOWN_SEC)
        print(f"[{sym}][SPLIT] Both legs closed — total P&L ≈ {total:.2f}; "
              f"next risk={risk_manager.get_risk()*100:.1f}%")
        ctx.state = TradeState.DONE
        return

    # ── Pos A closed, Pos B still open ─────────────────────────────────────
    if pos_a is None and pos_b is not None and not ctx.pos_b_sl_upgraded:
        a_profit = _get_deal_profit(ctx.pos_a_ticket)
        if a_profit is not None:
            if a_profit > 0:
                # Pos A TP → upgrade Pos B SL to 1.0 RR in profit
                b_entry    = float(pos_b.price_open)
                b_sl_orig  = float(pos_b.sl)
                b_stop     = abs(b_entry - b_sl_orig)
                if pos_b.type == mt5.POSITION_TYPE_BUY:
                    new_sl = round(b_entry + 1.0 * b_stop, info.digits)
                else:
                    new_sl = round(b_entry - 1.0 * b_stop, info.digits)
                if modify_sl(pos_b.ticket, new_sl, float(pos_b.tp), sym):
                    ctx.pos_b_sl_upgraded = True
                    risk_manager.record_win()
                    print(f"[{sym}][SPLIT] Pos A TP → Pos B SL upgraded to "
                          f"1.0 RR profit @ {new_sl:.5f}")
            else:
                # Pos A SL → close Pos B, record loss, start cooldown
                print(f"[{sym}][SPLIT] Pos A SL → force-closing Pos B")
                _close_position(pos_b.ticket, sym)
                risk_manager.record_loss()
                ctx.cooldown_until = datetime.now(timezone.utc) + \
                                     timedelta(seconds=ENTRY_COOLDOWN_SEC)
                ctx.state = TradeState.DONE
                return

    # ── Opposing MSS — exit all open legs early ─────────────────────────────
    if m5 is not None and ctx.direction_at_entry:
        if detect_opposing_mss(m5, ctx.direction_at_entry):
            print(f"[{sym}][MSS] Opposing structure detected — closing all positions")
            for p in ([pos_a] if pos_a else []) + ([pos_b] if pos_b else []) + legacy:
                _close_position(p.ticket, sym)
            risk_manager.record_loss()
            ctx.cooldown_until = datetime.now(timezone.utc) + \
                                 timedelta(seconds=ENTRY_COOLDOWN_SEC)
            ctx.state = TradeState.DONE
            return

    # ── 500-pip hard cap OR ADR 80% filled: secure 1.5 RR on Pos B ──────────
    if pos_b is not None and not ctx.pos_b_sl_upgraded and info:
        b_entry   = float(pos_b.price_open)
        b_sl      = float(pos_b.sl)
        b_stop    = abs(b_entry - b_sl)
        if pos_b.type == mt5.POSITION_TYPE_BUY:
            move    = float(tick.bid) - b_entry
            sl_15rr = round(b_entry + 1.5 * b_stop, info.digits)
        else:
            move    = b_entry - float(tick.ask)
            sl_15rr = round(b_entry - 1.5 * b_stop, info.digits)

        pip_cap_hit = move >= MAX_SCALP_TP_POINTS * point
        adr_cap_hit = False
        daily_adr   = get_daily_adr(sym)
        if daily_adr > 0:
            today_rng = get_today_range(sym)
            adr_cap_hit = (today_rng / daily_adr) >= ADR_TIGHTEN_PCT

        if pip_cap_hit or adr_cap_hit:
            reason = f"{move/point:.0f}pts pip cap" if pip_cap_hit \
                     else f"ADR {today_rng/daily_adr*100:.0f}% filled"
            print(f"[{sym}][CAP] {reason} — securing 1.5 RR on Pos B (SL → {sl_15rr:.5f})")
            if modify_sl(pos_b.ticket, sl_15rr, float(pos_b.tp), sym):
                ctx.pos_b_sl_upgraded = True


# =============================================================================
# PLAN LOADING
# =============================================================================
def load_plan(ctx: BotContext) -> bool:
    """Load intraday_plan_<symbol>.json. Returns True if a NEW or UPDATED plan was loaded."""
    plan_file = PLAN_DIR / f"intraday_plan_{ctx.symbol}.json"
    if not plan_file.exists():
        return False
    try:
        plan = json.loads(plan_file.read_text())
    except json.JSONDecodeError as e:
        print(f"[{ctx.symbol}][PLAN] invalid JSON: {e}", file=sys.stderr)
        return False

    mtime = datetime.fromtimestamp(plan_file.stat().st_mtime, tz=timezone.utc)
    if ctx.plan_loaded_at and mtime <= ctx.plan_loaded_at:
        return False  # not updated since last load

    # Check if the plan actually changed to avoid false resets
    if hasattr(ctx, 'plan') and ctx.plan:
        old_bias = ctx.plan.get('bias')
        old_low = ctx.plan.get('poi_zone_low')
        old_high = ctx.plan.get('poi_zone_high')
        if plan.get('bias') == old_bias and plan.get('poi_zone_low') == old_low and plan.get('poi_zone_high') == old_high:
            ctx.plan_loaded_at = mtime  # Update mtime but don't trigger a reset
            ctx.plan = plan
            return False

    ctx.plan = plan
    ctx.plan_loaded_at = mtime
    print(f"\n[{ctx.symbol}][PLAN] 🔄 New or updated intraday plan loaded")
    print(f"       bias={plan.get('bias')} POI=[{plan.get('poi_zone_low')}-"
          f"{plan.get('poi_zone_high')}] target={plan.get('target_liquidity')} "
          f"wait_news={plan.get('wait_for_news')}")
    return True


def reset_state(ctx: BotContext, reason: str):
    print(f"[{ctx.symbol}][STATE] Reset to IDLE — {reason}")
    ctx.state             = TradeState.IDLE
    ctx.sweep_wick_price  = None
    ctx.sweep_time        = None
    ctx.structure_pivot   = None
    ctx.ifvg_low          = None
    ctx.ifvg_high         = None
    ctx.pending_ticket    = None
    ctx.pending_placed_at = None
    ctx.market_ticket     = None
    ctx.stop_loss_price   = None
    ctx.take_profit_price = None
    ctx.fvg_low           = None
    ctx.fvg_high          = None
    ctx.fvg_midpoint      = None
    ctx.pos_a_ticket       = None
    ctx.pos_b_ticket       = None
    ctx.pos_entry          = None
    ctx.pos_b_sl_upgraded  = False
    ctx.direction_at_entry = None
    ctx.half_risk          = False


# =============================================================================
# MAIN LOOP — STATE MACHINE
# =============================================================================
def step(ctx: BotContext):
    """Run one iteration of the state machine for ctx.symbol."""
    sym = ctx.symbol

    # --- Always re-check plan file ----------------------------------------
    if load_plan(ctx):
        # Plan changed — reset
        if ctx.state != TradeState.POSITION_OPEN:
            reset_state(ctx, "plan updated")

    plan = ctx.plan
    if not plan:
        print(f"[{sym}][SCAN] ❌ No plan file found — run ai_brain.py first")
        return
    if plan.get("bias") == "neutral":
        print(f"[{sym}][SCAN] ⚪ bias=neutral — no trade today")
        print(f"              Reason: {plan.get('reasoning', 'N/A')}")
        return

    # --- Honour news blackout (overridden by FORCE_IGNORE_NEWS) -----------
    if plan.get("wait_for_news", False) and not FORCE_IGNORE_NEWS:
        if ctx.state in (TradeState.IDLE, TradeState.POI_TOUCHED,
                         TradeState.SWEEP_DONE, TradeState.CHOCH_CONFIRMED):
            print(f"[{sym}][NEWS] ⏸ wait_for_news=True — pausing entry logic until news passes")
            return
    elif plan.get("wait_for_news", False) and FORCE_IGNORE_NEWS:
        print(f"[{sym}][NEWS] ⚡ FORCE_IGNORE_NEWS=true — trading through news")

    direction = plan["bias"]

    # Re-compute H1 bias from fresh MT5 data every loop — no caching
    if ctx.state not in (TradeState.POSITION_OPEN, TradeState.DONE):
        live_bias = calculate_bias(sym)
        if live_bias != "neutral" and live_bias != direction:
            print(f"[{sym}][BIAS] ⚡ H1 MSS override: plan='{direction}' → live='{live_bias}'")
            if ctx.state not in (TradeState.SWEEP_DONE, TradeState.CHOCH_CONFIRMED,
                                 TradeState.IFVG_VALIDATED, TradeState.FVG_TAPPED):
                reset_state(ctx, f"bias flipped to {live_bias}")
            direction = live_bias

    if direction == "neutral":
        print(f"[{sym}][BIAS] ⚪ live bias = neutral — standing by")
        return

    # --- 3-hour state refresh — clear stale FVG context ----------------------
    if needs_state_refresh(ctx) and ctx.state == TradeState.IDLE:
        ctx.last_refresh = datetime.now(timezone.utc)
        reset_state(ctx, "3-hour state refresh")

    # --- Entry cooldown — minimum wait after a SL hit ------------------------
    if ctx.cooldown_until and datetime.now(timezone.utc) < ctx.cooldown_until:
        remaining = (ctx.cooldown_until - datetime.now(timezone.utc)).total_seconds()
        print(f"[{sym}][COOLDOWN] Entry cooldown active — {remaining:.0f}s remaining")
        return

    poi_low   = float(plan["poi_zone_low"])
    poi_high  = float(plan["poi_zone_high"])
    target_liq = float(plan["target_liquidity"])

    # Pull fresh candles — all fetched from scratch, no caching
    m1  = get_candles(mt5.TIMEFRAME_M1,  30,  sym)
    m5  = get_candles(mt5.TIMEFRAME_M5,  100, sym)
    m15 = get_candles(mt5.TIMEFRAME_M15, 50,  sym)
    if m1 is None or m5 is None or m15 is None:
        print(f"[{sym}][DATA] Failed to fetch candles", file=sys.stderr)
        return

    tick = get_tick(sym)
    if tick is None:
        print(f"[{sym}][DATA] ❌ get_tick() returned None — market may be closed")
        return
    price  = (tick.bid + tick.ask) / 2
    spread = tick.ask - tick.bid

    # ── Spread guard ───────────────────────────────────────────────────────────
    info  = mt5.symbol_info(sym)
    point = info.point if info else 0.0001
    spread_points = spread / point
    if spread_points > MAX_SPREAD_POINTS:
        print(f"[{sym}][SPREAD] ⛔ Spread {spread_points:.1f}pts > {MAX_SPREAD_POINTS}pts max — skipping tick")
        return
    elif spread_points > SPREAD_WARN_THRESHOLD:
        print(f"[{sym}][SPREAD] ⚠️ SPREAD WARNING: {spread_points:.1f}pts (warn threshold {SPREAD_WARN_THRESHOLD}pts) — proceeding with caution")

    # --- PO3 / Midnight Open context (price now defined) ---------------------
    midnight_open = get_midnight_open(sym)
    if midnight_open is not None:
        po3_ok = (direction == "bearish" and price > midnight_open) or \
                 (direction == "bullish" and price < midnight_open)
        po3_tag = "✅ manipulation zone" if po3_ok else "⚠️ outside manipulation zone"
        print(f"[{sym}][PO3]  MidnightOpen={midnight_open:.5f} | {po3_tag}")

    # --- Killzone gate — probability filter (price + candles now defined) ----
    if ctx.state in (TradeState.IDLE, TradeState.POI_TOUCHED,
                     TradeState.SWEEP_DONE, TradeState.CHOCH_CONFIRMED,
                     TradeState.IFVG_VALIDATED, TradeState.FVG_TAPPED):
        if not in_killzone():
            if is_high_probability_setup(ctx, m5, direction, price):
                ctx.half_risk = True
                print(f"[{sym}][KZ] Outside KZ — HP setup (sweep+FVG+OTE) → HALF-RISK entry allowed")
            else:
                ctx.half_risk = False
                print(f"[{sym}][KZ] Outside killzone — no HP setup, skipping")
                return
        else:
            ctx.half_risk = False

    # ── Terminal Cleanup (Throttle prints) ──────────────────────────────────
    if not hasattr(ctx, 'last_print_time'):
        ctx.last_print_time = 0
    if not hasattr(ctx, 'last_print_price'):
        ctx.last_print_price = price

    current_time = time.time()
    price_diff = abs(price - ctx.last_print_price) / point if point > 0 else 0

    should_print = (current_time - ctx.last_print_time > 15) or (price_diff > 50)

    if should_print:
        print(f"\n[{sym}][SCAN]  ──── {datetime.now().strftime('%H:%M:%S')} | State: {ctx.state.value} ────")
        print(f"[{sym}][MARKET] Price={price:.5f} | Spread={spread_points:.1f}pts | Bid={tick.bid:.5f} | Ask={tick.ask:.5f}")
        print(f"[{sym}][PLAN]   Bias={direction} | POI=[{poi_low:.5f}–{poi_high:.5f}]")
        ctx.last_print_time = current_time
        ctx.last_print_price = price

    # --- STATE: IDLE — wait for price to enter POI -----------------------
    if ctx.state == TradeState.IDLE:
        # ── Fast path: FVG touch + M1 MSS — no sweep required ────────────────
        # Active FVGs are re-computed from fresh MT5 data every tick (no cache).
        # If bias is strong and price enters an aligned FVG with M1 confirmation,
        # enter immediately without waiting for the engineered liquidity sweep.
        _fvg = find_aligned_fvg(m15, direction) or find_aligned_fvg(m5, direction)
        if _fvg is not None and _fvg["low"] <= price <= _fvg["high"]:
            if detect_m1_mss(m1, direction):
                print(f"[{sym}][FVG+MSS] Aligned FVG=[{_fvg['low']:.5f}–{_fvg['high']:.5f}] "
                      f"touched + M1 displacement confirmed — direct entry")
                _buffer = 5 * point
                if direction == "bullish":
                    _entry     = round(tick.ask, info.digits)
                    _sl        = _fvg["low"] - _buffer
                    _stop_dist = _entry - _sl
                    _tp        = _entry + 3.0 * _stop_dist
                else:
                    _entry     = round(tick.bid, info.digits)
                    _sl        = _fvg["high"] + _buffer
                    _stop_dist = _sl - _entry
                    _tp        = _entry - 3.0 * _stop_dist
                print(f"[{sym}][FVG+MSS] Entry={_entry:.5f} SL={_sl:.5f} TP={_tp:.5f} "
                      f"StopDist={_stop_dist:.5f} RR=3.0")
                if _stop_dist > 0:
                    _rm = 0.5 if ctx.half_risk else 1.0
                    if execute_split_trade(ctx, direction, _entry, _sl, plan, sym, _rm):
                        ctx.stop_loss_price   = _sl
                        ctx.take_profit_price = _tp
                        print(f"[{sym}][FVG+MSS] ✅ SPLIT TRADE FIRED "
                              f"A={ctx.pos_a_ticket} B={ctx.pos_b_ticket}")
                        return

        if poi_low <= price <= poi_high:
            print(f"[{sym}][STATE] ✅ POI touched @ {price:.5f} (zone {poi_low:.5f}–{poi_high:.5f})")
            ctx.state = TradeState.POI_TOUCHED
        else:
            dist_to_low  = abs(price - poi_low)
            dist_to_high = abs(price - poi_high)
            nearest = min(dist_to_low, dist_to_high)
            nearest_pts = nearest / point if point > 0 else 0
            side = "below" if price < poi_low else "above"
            print(f"[{sym}][IDLE]  Price {side} POI — {nearest:.5f} ({nearest_pts:.0f}pts) from nearest edge"
                  f" ({poi_low:.5f}–{poi_high:.5f})")

            # ── PROXIMITY EXECUTION EDGE ─────────────────────────────────────
            # If price is within PROXIMITY_POINTS of POI and bias is confirmed,
            # fire an aggressive market order immediately.
            if nearest_pts <= PROXIMITY_POINTS and nearest_pts > 0:
                print(f"[{sym}][PROXIMITY] 🎯 Price within {nearest_pts:.0f}pts of POI — PROXIMITY TRIGGER")
                buffer = 5 * point

                if direction == "bullish":
                    entry     = round(tick.ask, info.digits)
                    sl        = poi_low - buffer
                    stop_dist = entry - sl
                    tp        = entry + 3 * stop_dist
                else:
                    entry     = round(tick.bid, info.digits)
                    sl        = poi_high + buffer
                    stop_dist = sl - entry
                    tp        = entry - 3 * stop_dist

                print(f"[{sym}][PROXIMITY] Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} StopDist={stop_dist:.5f}")
                if stop_dist > 0:
                    _rm = 0.5 if ctx.half_risk else 1.0
                    if execute_split_trade(ctx, direction, entry, sl, plan, sym, _rm):
                        ctx.stop_loss_price   = sl
                        ctx.take_profit_price = tp
                        print(f"[{sym}][PROXIMITY] ✅ SPLIT TRADE FIRED "
                              f"A={ctx.pos_a_ticket} B={ctx.pos_b_ticket}")
                        return
                    else:
                        print(f"[{sym}][PROXIMITY] ❌ Split trade failed")
                else:
                    print(f"[{sym}][PROXIMITY] ❌ Invalid stop distance")

    # --- STATE: POI_TOUCHED — try sweep first, but DIRECT ENTRY if sweep doesn't come ---
    if ctx.state == TradeState.POI_TOUCHED:
        # Track how long we've been in POI_TOUCHED
        if not hasattr(ctx, '_poi_touched_ticks'):
            ctx._poi_touched_ticks = 0
        ctx._poi_touched_ticks += 1

        # Check if sweep is required (plan can override with wait_for_sweep: false)
        sweep_required = plan.get("wait_for_sweep", True)

        # Try to detect sweep normally
        sweep = detect_wick_sweep(m5, target_liq, direction)
        if sweep:
            print(f"[{sym}][STATE] ✅ Wick sweep confirmed @ {sweep['wick_price']:.5f} (target was {target_liq:.5f})")
            ctx.sweep_wick_price = sweep["wick_price"]
            ctx.sweep_time = sweep["time"]
            ctx.state = TradeState.SWEEP_DONE
            ctx._poi_touched_ticks = 0
        else:
            if len(m5) >= 2:
                c = m5.iloc[-2]
                if direction == "bullish":
                    print(f"[{sym}][SWEEP] Bullish — low={c['low']:.5f} vs target={target_liq:.5f} | Ticks in POI: {ctx._poi_touched_ticks}")
                else:
                    print(f"[{sym}][SWEEP] Bearish — high={c['high']:.5f} vs target={target_liq:.5f} | Ticks in POI: {ctx._poi_touched_ticks}")

            # ── DIRECT ENTRY OVERRIDE ─────────────────────────────────────────
            # If sweep hasn't happened after 10 ticks OR plan says sweep not required,
            # and price is STILL inside the POI zone → fire market order directly.
            skip_sweep = (not sweep_required) or (ctx._poi_touched_ticks >= 10)

            if skip_sweep and poi_low <= price <= poi_high:
                reason = "wait_for_sweep=false" if not sweep_required else f"{ctx._poi_touched_ticks} ticks without sweep"
                print(f"[{sym}][DIRECT ENTRY] 🔥 SWEEP BYPASSED ({reason}) — price {price:.5f} inside POI")
                print(f"[{sym}][DIRECT ENTRY] Bias={direction} — firing MARKET ORDER immediately")

                buffer = 5 * point

                if direction == "bullish":
                    entry     = round(tick.ask, info.digits)
                    sl        = poi_low - buffer
                    stop_dist = entry - sl
                    tp        = entry + 3 * stop_dist
                else:
                    entry     = round(tick.bid, info.digits)
                    sl        = poi_high + buffer
                    stop_dist = sl - entry
                    tp        = entry - 3 * stop_dist

                print(f"[{sym}][DIRECT ENTRY] Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} StopDist={stop_dist:.5f}")
                if stop_dist > 0:
                    _rm = 0.5 if ctx.half_risk else 1.0
                    if execute_split_trade(ctx, direction, entry, sl, plan, sym, _rm):
                        ctx.stop_loss_price    = sl
                        ctx.take_profit_price  = tp
                        ctx._poi_touched_ticks = 0
                        print(f"[{sym}][DIRECT ENTRY] ✅ SPLIT TRADE FIRED "
                              f"A={ctx.pos_a_ticket} B={ctx.pos_b_ticket}")
                        return
                    else:
                        print(f"[{sym}][DIRECT ENTRY] ❌ Split trade failed")
                else:
                    print(f"[{sym}][DIRECT ENTRY] ❌ Invalid stop distance")
            else:
                print(f"[{sym}][SWEEP] ❌ Sweep not confirmed — waiting ({ctx._poi_touched_ticks}/10 ticks before direct entry)")

    # --- STATE: SWEEP_DONE — wait for CHoCH with displacement ------------
    if ctx.state == TradeState.SWEEP_DONE:
        if len(m5) >= 11:
            last_10  = m5.tail(11).iloc[:-1]
            avg_rng  = float((last_10["high"] - last_10["low"]).mean())
            last_c   = m5.iloc[-2]
            last_rng = float(last_c["high"] - last_c["low"])
            required = DISPLACEMENT_MULTIPLIER * avg_rng
            disp_ok  = last_rng >= required
            print(f"[{sym}][CHOCH] Displacement: LastRange={last_rng:.5f} | AvgRange={avg_rng:.5f} | Required={required:.5f} | OK={disp_ok}")
        pivot = detect_choch_with_displacement(m5, direction, ctx.sweep_time)
        if pivot is not None:
            print(f"[{sym}][STATE] ✅ CHoCH confirmed — structure pivot={pivot:.5f}")
            ctx.structure_pivot = pivot
            ctx.state = TradeState.CHOCH_CONFIRMED
        else:
            print(f"[{sym}][CHOCH] ❌ No CHoCH yet — need displaced break of structure pivot")

    # --- STATE: CHOCH_CONFIRMED — Path A: IFVG | Path B: FVG Respect -------
    if ctx.state == TradeState.CHOCH_CONFIRMED:
        info   = mt5.symbol_info(sym)
        point  = info.point if info else 0.0001
        buffer = 5 * point

        # ── OTE quality check (informational — does not block inside KZ) ────
        if ctx.sweep_wick_price and ctx.structure_pivot:
            check_ote(ctx.sweep_wick_price, ctx.structure_pivot, direction, price)

        # ── Path A: IFVG (body close fully THROUGH an opposing FVG) ─────────
        opposing = find_opposing_fvg(m5, direction)
        if opposing:
            last_c    = m5.iloc[-2]
            body_low  = min(last_c["open"], last_c["close"])
            body_high = max(last_c["open"], last_c["close"])
            print(f"[{sym}][IFVG] Opposing FVG=[{opposing['low']:.5f}–{opposing['high']:.5f}] "
                  f"Body=[{body_low:.5f}–{body_high:.5f}]")
            ifvg_ok = (body_low > opposing["high"]) if direction == "bullish" \
                      else (body_high < opposing["low"])
            print(f"[{sym}][IFVG] Validated={ifvg_ok}")

            if ifvg_ok:
                print(f"[{sym}][STATE] ✅ IFVG confirmed (Path A — aggressive)")
                ctx.ifvg_low  = opposing["low"]
                ctx.ifvg_high = opposing["high"]
                entry = ctx.ifvg_high if direction == "bullish" else ctx.ifvg_low

                if direction == "bullish":
                    sl        = ctx.sweep_wick_price - buffer
                    stop_dist = entry - sl
                    tp        = entry + 3 * stop_dist
                else:
                    sl        = ctx.sweep_wick_price + buffer
                    stop_dist = sl - entry
                    tp        = entry - 3 * stop_dist

                print(f"[{sym}][ORDER-CALC] Entry={entry:.5f} SL={sl:.5f} TP={tp:.5f} StopDist={stop_dist:.5f} RR=3.0")
                if stop_dist <= 0:
                    reset_state(ctx, "invalid stop distance")
                    return

                ctx.stop_loss_price   = sl
                ctx.take_profit_price = tp

                _rm = 0.5 if ctx.half_risk else 1.0
                if execute_split_trade(ctx, direction, entry, sl, plan, sym, _rm):
                    print(f"[{sym}][IFVG] ✅ SPLIT TRADE FIRED (Path A) "
                          f"A={ctx.pos_a_ticket} B={ctx.pos_b_ticket}")
                else:
                    reset_state(ctx, "IFVG split trade failed")
                return  # Path A taken — skip Path B this tick
        else:
            print(f"[{sym}][IFVG] No opposing FVG found — checking regular FVG respect (Path B)")

        # ── Path B: Regular FVG Respect (tap + rejection candle) ────────────
        aligned = find_aligned_fvg(m5, direction)
        if not aligned:
            print(f"[{sym}][FVG]  ❌ No aligned {direction} FVG found post-CHoCH")
        else:
            last_c    = m5.iloc[-2]
            tapped    = (last_c["low"] <= aligned["high"]) if direction == "bullish" \
                        else (last_c["high"] >= aligned["low"])
            respected = detect_respect_candle(m5, aligned, direction)
            print(f"[{sym}][FVG]  Aligned {aligned['type']} FVG=[{aligned['low']:.5f}–"
                  f"{aligned['high']:.5f}] mid={aligned['midpoint']:.5f} | "
                  f"Tapped={tapped} Respected={respected}")

            if respected:
                print(f"[{sym}][STATE] ✅ FVG respected — market entry (Path B — conservative)")
                entry = round(tick.ask if direction == "bullish" else tick.bid, info.digits)

                if direction == "bullish":
                    sl        = ctx.sweep_wick_price - buffer
                    stop_dist = entry - sl
                    tp        = entry + 3 * stop_dist
                else:
                    sl        = ctx.sweep_wick_price + buffer
                    stop_dist = sl - entry
                    tp        = entry - 3 * stop_dist

                print(f"[{sym}][ORDER-CALC] Entry≈{entry:.5f} SL={sl:.5f} TP={tp:.5f} StopDist={stop_dist:.5f} RR=3.0")
                if stop_dist <= 0:
                    reset_state(ctx, "invalid stop distance")
                    return

                ctx.stop_loss_price   = sl
                ctx.take_profit_price = tp
                _rm = 0.5 if ctx.half_risk else 1.0
                if execute_split_trade(ctx, direction, entry, sl, plan, sym, _rm):
                    print(f"[{sym}][FVG] ✅ SPLIT TRADE FIRED (Path B) "
                          f"A={ctx.pos_a_ticket} B={ctx.pos_b_ticket}")
                else:
                    reset_state(ctx, "FVG respect split trade failed")

            elif tapped:
                print(f"[{sym}][FVG]  FVG tapped — waiting for respect candle → FVG_TAPPED")
                ctx.fvg_low      = aligned["low"]
                ctx.fvg_high     = aligned["high"]
                ctx.fvg_midpoint = aligned["midpoint"]
                ctx.state        = TradeState.FVG_TAPPED
            else:
                print(f"[{sym}][FVG]  FVG identified but not yet tapped — waiting")

    # --- STATE: FVG_TAPPED — waiting for respect (rejection) candle ----------
    if ctx.state == TradeState.FVG_TAPPED:
        if ctx.fvg_low is None:
            reset_state(ctx, "FVG context lost")
            return

        fvg    = {"low": ctx.fvg_low, "high": ctx.fvg_high, "midpoint": ctx.fvg_midpoint}
        last_c = m5.iloc[-2]
        info   = mt5.symbol_info(sym)
        point  = info.point if info else 0.0001
        buffer = 5 * point

        # Invalidate if price closed fully through the FVG (setup blown)
        if direction == "bullish" and last_c["close"] < fvg["low"]:
            reset_state(ctx, "FVG fully mitigated — bullish setup invalidated")
            return
        if direction == "bearish" and last_c["close"] > fvg["high"]:
            reset_state(ctx, "FVG fully mitigated — bearish setup invalidated")
            return

        respected = detect_respect_candle(m5, fvg, direction)
        print(f"[{sym}][FVG_TAPPED] FVG=[{fvg['low']:.5f}–{fvg['high']:.5f}] "
              f"mid={fvg['midpoint']:.5f} | Respected={respected}")

        if ctx.sweep_wick_price and ctx.structure_pivot:
            check_ote(ctx.sweep_wick_price, ctx.structure_pivot, direction, price)

        if not respected:
            print(f"[{sym}][FVG_TAPPED] ❌ No respect candle yet — holding")
            return

        print(f"[{sym}][STATE] ✅ FVG respect candle confirmed — entering market")
        entry = round(tick.ask if direction == "bullish" else tick.bid, info.digits)

        if direction == "bullish":
            sl        = ctx.sweep_wick_price - buffer
            stop_dist = entry - sl
            tp        = entry + 3 * stop_dist
        else:
            sl        = ctx.sweep_wick_price + buffer
            stop_dist = sl - entry
            tp        = entry - 3 * stop_dist

        print(f"[{sym}][ORDER-CALC] Entry≈{entry:.5f} SL={sl:.5f} TP={tp:.5f} StopDist={stop_dist:.5f} RR=3.0")
        if stop_dist <= 0:
            reset_state(ctx, "invalid stop distance")
            return

        ctx.stop_loss_price   = sl
        ctx.take_profit_price = tp
        _rm = 0.5 if ctx.half_risk else 1.0
        if execute_split_trade(ctx, direction, entry, sl, plan, sym, _rm):
            print(f"[{sym}][FVG_TAPPED] ✅ SPLIT TRADE FIRED "
                  f"A={ctx.pos_a_ticket} B={ctx.pos_b_ticket}")
        else:
            reset_state(ctx, "FVG tapped split trade failed")

    # --- STATE: IFVG_VALIDATED — manage pending limit order ---------------
    if ctx.state == TradeState.IFVG_VALIDATED:
        limit_filled = (ctx.pending_ticket is not None and
                        position_exists_for_magic(MAGIC_LIMIT, sym))
        if limit_filled:
            print(f"[{sym}][STATE] Limit leg filled — both legs now open → POSITION_OPEN")
            ctx.state = TradeState.POSITION_OPEN
        elif ctx.pending_ticket and ctx.pending_placed_at:
            age = (datetime.now(timezone.utc) - ctx.pending_placed_at).total_seconds() / 60
            if age > LIMIT_EXPIRY_MINUTES:
                print(f"[{sym}][ORDER] Limit leg expired after {age:.1f}min — cancelling")
                cancel_order(ctx.pending_ticket)
                ctx.pending_ticket = None
                # Market leg may still be running — keep managing it
                if position_exists_for_magic(MAGIC_MARKET, sym):
                    print(f"[{sym}][STATE] Market leg still open → POSITION_OPEN")
                    ctx.state = TradeState.POSITION_OPEN
                else:
                    reset_state(ctx, "limit expired, market leg already closed")

    # --- STATE: POSITION_OPEN — manage split positions, wait for close -------
    if ctx.state == TradeState.POSITION_OPEN:
        monitor_positions(ctx, sym, m5)

    # --- STATE: DONE — wait for next plan to reset -----------------------
    if ctx.state == TradeState.DONE:
        # Idle here until a new plan loads (handled at top of step())
        pass


def main() -> int:
    print("=" * 70)
    print(f"MT5 SNIPER — {', '.join(SYMBOLS)} — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    if not init_mt5():
        return 1

    contexts = [BotContext(symbol=sym) for sym in SYMBOLS]

    try:
        while True:
            for ctx in contexts:
                try:
                    step(ctx)
                except Exception as e:
                    # Catch-all: never crash the loop on a single bad tick
                    print(f"[{ctx.symbol}][LOOP] step() raised: {e!r}", file=sys.stderr)
            time.sleep(LOOP_INTERVAL_SEC)
    except KeyboardInterrupt:
        print("\n[EXIT] Ctrl+C received, shutting down.")
        return 0
    finally:
        mt5.shutdown()


if __name__ == "__main__":
    sys.exit(main())
