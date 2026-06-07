"""
=============================================================================
main.py - Unified Autonomous Execution Orchestrator (v3)
=============================================================================

Purpose:
    - Single entry point: python main.py
    - NO subprocess, NO Start-Job, NO background processes.
    - Everything runs IN THIS PROCESS, IN THIS TERMINAL.
    - Phase 1: Run AI Brain to generate POI plans for all symbols.
    - Phase 2: Start continuous 1-second execution loop running BOTH
      the scalper and sniper state machines.
    - Phase 3: Re-run the Brain every 15 minutes to refresh POIs.

Path Safety:
    - Pins working directory to the script's own folder on startup.
    - All file paths use Path(__file__).parent — never relies on CWD.
    - This fixes the PowerShell Start-Job Documents path issue.

=============================================================================
"""

import json
import os
import sys
import time
import threading
from datetime import datetime, timezone
from pathlib import Path

# ═══════════════════════════════════════════════════════════════════════════
# PATH FIX: Pin CWD to the script's directory BEFORE any imports.
# This prevents PowerShell/subprocess path confusion.
# ═══════════════════════════════════════════════════════════════════════════
SCRIPT_DIR = Path(__file__).resolve().parent
os.chdir(SCRIPT_DIR)
sys.path.insert(0, str(SCRIPT_DIR))

from dotenv import load_dotenv
load_dotenv(SCRIPT_DIR / ".env")  # Explicit path to .env

# ---------------------------------------------------------------------------
# Module imports (now guaranteed to find them in SCRIPT_DIR)
# ---------------------------------------------------------------------------
import MetaTrader5 as mt5
import ai_brain
import mt5_scalper
import mt5_sniper

SYMBOLS = [s.strip() for s in os.getenv("SYMBOLS", "XAUUSD").split(",") if s.strip()]
PLAN_DIR = SCRIPT_DIR  # Same as script directory

LOOP_INTERVAL_SEC = 1
BRAIN_REFRESH_MINUTES = 15
MT5_MAX_RETRIES = 3

# News override — defaults to True (ignore news)
FORCE_IGNORE_NEWS = os.getenv("FORCE_IGNORE_NEWS", "true").lower() in ("true", "1", "yes")


# =============================================================================
# MT5 CONNECTION (single shared instance with retry logic)
# =============================================================================
def init_mt5_shared() -> bool:
    """
    Initialize MT5 with up to 3 retry attempts.
    Waits 5s, 10s, 20s between retries (exponential backoff).
    """
    MT5_LOGIN = int(os.getenv("MT5_LOGIN", "0"))
    MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
    MT5_SERVER = os.getenv("MT5_SERVER", "")
    MT5_PATH = os.getenv("MT5_PATH", "")

    args = dict(login=MT5_LOGIN, password=MT5_PASSWORD, server=MT5_SERVER)
    if MT5_PATH:
        args["path"] = MT5_PATH

    for attempt in range(1, MT5_MAX_RETRIES + 1):
        print(f"[MAIN][MT5] Connection attempt {attempt}/{MT5_MAX_RETRIES}...")

        if mt5.initialize(**args):
            info = mt5.account_info()
            if info is not None:
                print(f"[MAIN][MT5] ✅ Connected — Account={info.login} Server={info.server} "
                      f"Balance={info.balance} {info.currency}")

                for sym in SYMBOLS:
                    if not mt5.symbol_select(sym, True):
                        print(f"[MAIN][MT5] ⚠️ symbol_select({sym}) failed — continuing", file=sys.stderr)
                return True
            else:
                print(f"[MAIN][MT5] ❌ account_info() returned None", file=sys.stderr)
        else:
            print(f"[MAIN][MT5] ❌ init failed: {mt5.last_error()}", file=sys.stderr)

        if attempt < MT5_MAX_RETRIES:
            wait_sec = 5 * (2 ** (attempt - 1))
            print(f"[MAIN][MT5] ⏳ Retrying in {wait_sec}s...")
            time.sleep(wait_sec)

    print(f"[MAIN][MT5] ☠️ FATAL — Failed to connect after {MT5_MAX_RETRIES} attempts",
          file=sys.stderr)
    return False


def reconnect_mt5() -> bool:
    """Attempt to reconnect MT5 if the connection drops mid-loop."""
    print("[MAIN][MT5] 🔄 Connection lost — attempting reconnect...")
    try:
        mt5.shutdown()
    except Exception:
        pass
    return init_mt5_shared()


# =============================================================================
# BRAIN RUNNER (in-process, no subprocess)
# =============================================================================
def run_brain_safe() -> bool:
    """
    Run ai_brain logic WITHOUT calling mt5.shutdown().
    Returns True if at least one plan was generated successfully.
    """
    print("\n" + "═" * 70)
    print(f"  🧠 AI BRAIN — Generating Plans @ {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S')} UTC")
    print("═" * 70)

    try:
        news = ai_brain.fetch_high_impact_news()
        wait_flag = ai_brain.is_news_within_2h(news)

        # If FORCE_IGNORE_NEWS is on, override the flag
        if FORCE_IGNORE_NEWS:
            print(f"[BRAIN] ⚡ FORCE_IGNORE_NEWS=true — overriding wait_for_news to False")
            wait_flag = False

        print(f"[BRAIN] News events: {len(news)} | wait_for_news: {wait_flag}")

        plans_generated = 0

        for sym in SYMBOLS:
            print(f"\n{'─' * 60}")
            print(f"[BRAIN][{sym}] Building HTF context...")
            print(f"{'─' * 60}")

            try:
                htf = ai_brain.build_htf_context(sym)
                current_price = htf["timeframes"]["H1"]["current_price"]
                print(f"[BRAIN][{sym}] Current price: {current_price}")

                # Intraday plan
                print(f"[BRAIN][{sym}] Calling Gemini for INTRADAY plan...")
                intraday_plan = ai_brain.call_gemini(htf, news, wait_flag, ai_brain.SYSTEM_PROMPT)

                if ai_brain.validate_plan(intraday_plan, current_price):
                    intraday_plan["wait_for_news"] = wait_flag
                    intraday_plan["_generated_at"] = datetime.now(timezone.utc).isoformat()
                    intraday_plan["_symbol"] = sym
                    intraday_plan["_current_price_at_plan"] = current_price
                    intraday_file = PLAN_DIR / f"intraday_plan_{sym}.json"
                    intraday_file.write_text(json.dumps(intraday_plan, indent=2))
                    print(f"[BRAIN][{sym}] ✅ Intraday plan saved → {intraday_file.name}")
                    plans_generated += 1
                else:
                    print(f"[BRAIN][{sym}] ⚠️ Intraday plan failed validation")

                # Scalp plan
                print(f"[BRAIN][{sym}] Calling Gemini for SCALP plan...")
                scalp_plan = ai_brain.call_gemini(htf, news, wait_flag, ai_brain.SCALP_SYSTEM_PROMPT)

                if ai_brain.validate_plan(scalp_plan, current_price):
                    scalp_plan["wait_for_news"] = wait_flag
                    scalp_plan["_generated_at"] = datetime.now(timezone.utc).isoformat()
                    scalp_plan["_symbol"] = sym
                    scalp_plan["_current_price_at_plan"] = current_price
                    scalp_file = PLAN_DIR / f"scalp_plan_{sym}.json"
                    scalp_file.write_text(json.dumps(scalp_plan, indent=2))
                    print(f"[BRAIN][{sym}] ✅ Scalp plan saved → {scalp_file.name}")
                    plans_generated += 1
                else:
                    print(f"[BRAIN][{sym}] ⚠️ Scalp plan failed validation")

            except Exception as e:
                print(f"[BRAIN][{sym}] ❌ Error: {e!r}", file=sys.stderr)

        return plans_generated > 0

    except Exception as e:
        print(f"[BRAIN] ❌ Fatal error: {e!r}", file=sys.stderr)
        return False


# =============================================================================
# PLAN VERIFICATION & SYNC STATUS
# =============================================================================
def verify_plans() -> dict:
    """Check which plan files exist. Returns status dict."""
    status = {}
    for sym in SYMBOLS:
        sym_status = {"intraday": None, "scalp": None}

        intraday_file = PLAN_DIR / f"intraday_plan_{sym}.json"
        if intraday_file.exists():
            try:
                plan = json.loads(intraday_file.read_text())
                sym_status["intraday"] = plan.get("bias", "unknown")
            except Exception:
                sym_status["intraday"] = "error"

        scalp_file = PLAN_DIR / f"scalp_plan_{sym}.json"
        if scalp_file.exists():
            try:
                plan = json.loads(scalp_file.read_text())
                sym_status["scalp"] = plan.get("bias", "unknown")
            except Exception:
                sym_status["scalp"] = "error"

        status[sym] = sym_status

    return status


def print_execution_ready(status: dict):
    """Print the EXECUTION READY banner."""
    print("\n")
    print("═" * 70)
    print("  ██████╗ ██╗  ██╗███████╗ ██████╗██╗   ██╗████████╗██╗ ██████╗ ███╗   ██╗")
    print("  ██╔════╝╚██╗██╔╝██╔════╝██╔════╝██║   ██║╚══██╔══╝██║██╔═══██╗████╗  ██║")
    print("  █████╗   ╚███╔╝ █████╗  ██║     ██║   ██║   ██║   ██║██║   ██║██╔██╗ ██║")
    print("  ██╔══╝   ██╔██╗ ██╔══╝  ██║     ██║   ██║   ██║   ██║██║   ██║██║╚██╗██║")
    print("  ███████╗██╔╝ ██╗███████╗╚██████╗╚██████╔╝   ██║   ██║╚██████╔╝██║ ╚████║")
    print("  ╚══════╝╚═╝  ╚═╝╚══════╝ ╚═════╝ ╚═════╝    ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝")
    print("")
    print("                    ██ EXECUTION READY ██")
    print("═" * 70)

    for sym, s in status.items():
        intraday_str = f"✅ {s['intraday']}" if s['intraday'] else "❌ missing"
        scalp_str    = f"✅ {s['scalp']}" if s['scalp'] else "❌ missing"
        print(f"  {sym}:")
        print(f"    Sniper  (intraday): {intraday_str}")
        print(f"    Scalper (scalp):    {scalp_str}")

    news_str = "⚡ DISABLED (trading through)" if FORCE_IGNORE_NEWS else "🛡️ Active"
    print(f"\n  Working dir:      {SCRIPT_DIR}")
    print(f"  Loop interval:    {LOOP_INTERVAL_SEC}s")
    print(f"  Brain refresh:    Every {BRAIN_REFRESH_MINUTES} minutes")
    print(f"  Max spread:       55 points")
    print(f"  News filter:      {news_str}")
    print(f"  Sweep required:   OPTIONAL (direct entry after 10s)")
    print(f"  MT5 reconnect:    {MT5_MAX_RETRIES} retries with backoff")
    print("═" * 70)
    print("")


# =============================================================================
# BACKGROUND BRAIN REFRESH (every 15 minutes)
# =============================================================================
_brain_lock = threading.Lock()
_last_brain_run = None


def brain_refresh_loop():
    """Runs in a background thread. Re-generates plans every 15 minutes."""
    global _last_brain_run
    while True:
        time.sleep(BRAIN_REFRESH_MINUTES * 60)
        print(f"\n[BRAIN-REFRESH] ⏰ {BRAIN_REFRESH_MINUTES}min elapsed — regenerating plans...")
        with _brain_lock:
            try:
                run_brain_safe()
                _last_brain_run = datetime.now(timezone.utc)
                status = verify_plans()
                print_execution_ready(status)
            except Exception as e:
                print(f"[BRAIN-REFRESH] ❌ Error: {e!r}", file=sys.stderr)


# =============================================================================
# MAIN — THE AUTONOMOUS LOOP (everything in one process, one terminal)
# =============================================================================
def main() -> int:
    global _last_brain_run

    print("╔" + "═" * 68 + "╗")
    print("║" + "  ICT AUTONOMOUS TRADING SYSTEM v3".center(68) + "║")
    print("║" + f"  Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}".center(68) + "║")
    print("║" + f"  Symbols: {', '.join(SYMBOLS)}".center(68) + "║")
    print("║" + f"  Path: {SCRIPT_DIR}".center(68) + "║")
    print("╚" + "═" * 68 + "╝")
    print(f"\n[MAIN] Working directory pinned to: {SCRIPT_DIR}")

    # ── Step 1: Initialize MT5 with retry logic ────────────────────────────
    print("\n[MAIN] Phase 1: Initializing MT5 connection...")
    if not init_mt5_shared():
        print("[MAIN] ☠️ MT5 initialization failed after all retries. Exiting.", file=sys.stderr)
        return 1

    # ── Step 2: Run the Brain to generate plans ────────────────────────────
    print("\n[MAIN] Phase 2: Running AI Brain...")
    brain_ok = run_brain_safe()
    _last_brain_run = datetime.now(timezone.utc)

    if not brain_ok:
        print("[MAIN] ⚠️ Brain generated 0 new plans — using existing plans on disk...")

    # ── Step 3: Verify sync status ─────────────────────────────────────────
    status = verify_plans()
    any_plan_exists = any(
        s.get("intraday") or s.get("scalp")
        for s in status.values()
    )

    if not any_plan_exists:
        print("[MAIN] ❌ No plan files found at all. Cannot start execution.", file=sys.stderr)
        print("[MAIN]    Check your Gemini API key and MT5 data connection.", file=sys.stderr)
        mt5.shutdown()
        return 2

    print_execution_ready(status)

    # ── Step 4: Create bot contexts for both engines ───────────────────────
    scalper_contexts = [mt5_scalper.BotContext(symbol=sym) for sym in SYMBOLS]
    sniper_contexts  = [mt5_sniper.BotContext(symbol=sym) for sym in SYMBOLS]

    # ── Step 5: Start brain refresh thread ─────────────────────────────────
    brain_thread = threading.Thread(target=brain_refresh_loop, daemon=True)
    brain_thread.start()
    print(f"[MAIN] 🧠 Brain refresh thread started (every {BRAIN_REFRESH_MINUTES}min)\n")

    # ── Step 6: Main execution loop (ALL IN THIS PROCESS) ──────────────────
    print(f"[MAIN] 🚀 Execution loop started — scalper + sniper every {LOOP_INTERVAL_SEC}s")
    print(f"[MAIN] ℹ️ NO subprocess, NO Start-Job — everything runs here.\n")

    tick_count = 0
    consecutive_errors = 0

    try:
        while True:
            tick_count += 1

            # Print heartbeat every 60 ticks (~1 min)
            if tick_count % 60 == 0:
                elapsed_brain = ""
                if _last_brain_run:
                    mins = (datetime.now(timezone.utc) - _last_brain_run).total_seconds() / 60
                    elapsed_brain = f" | Brain age: {mins:.0f}min"
                print(f"\n[HEARTBEAT] Tick #{tick_count} | "
                      f"{datetime.now().strftime('%H:%M:%S')}{elapsed_brain}")

            # ── Check connection health ──────────────────────────────────────
            # If another script runs and calls mt5.shutdown(), this connection dies.
            # We must detect it early and reconnect immediately.
            term_info = mt5.terminal_info()
            if term_info is None:
                print(f"[MAIN] ⚠️ MT5 Connection Lost — reconnecting immediately...")
                if reconnect_mt5():
                    print("[MAIN] ✅ Reconnected successfully")
                else:
                    print("[MAIN] ☠️ Reconnection failed — will try again next tick", file=sys.stderr)
                    time.sleep(LOOP_INTERVAL_SEC)
                    continue

            # ── Run Scalper steps ──────────────────────────────────────────
            for ctx in scalper_contexts:
                try:
                    mt5_scalper.step(ctx)
                except Exception as e:
                    print(f"[SCALPER][{ctx.symbol}] error: {e!r}", file=sys.stderr)

            # ── Run Sniper steps ───────────────────────────────────────────
            for ctx in sniper_contexts:
                try:
                    mt5_sniper.step(ctx)
                except Exception as e:
                    print(f"[SNIPER][{ctx.symbol}] error: {e!r}", file=sys.stderr)

            time.sleep(LOOP_INTERVAL_SEC)

    except KeyboardInterrupt:
        print("\n[MAIN] 🛑 Ctrl+C — shutting down gracefully.")
        return 0
    finally:
        mt5.shutdown()
        print("[MAIN] MT5 connection closed.")


if __name__ == "__main__":
    sys.exit(main())
