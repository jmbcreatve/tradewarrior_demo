# TradeWarrior Canon
_Last updated: 2025-12-03 (VPS → GitHub → local sync completed)_

---

## 1. Overview

TradeWarrior is an automated trading agent built in Python.  
It runs a loop of: **config/state → data adapters → snapshot → gatekeeper → GPT policy → risk engine → execution adapter → state**, plus a replay engine for backtesting and analytics.

Primary goals:

- Research‑grade replay harness and parity with live engine.
- Explicit, testable contract between snapshot → GPT → risk → execution.
- Strict, downward‑only risk envelopes around any GPT suggestion.
- Eventually: safe Hyperliquid testnet/live modes behind hard limits.

---

## 2. Roles & Workflow (for LLMs + human)

- **Architect** – designs phases, invariants, and contracts. Never edits code directly.
- **Foreman** – turns architecture into concrete tasks and prompts for code workers.
- **Craftsman** – edits code (Cursor / Codex / local dev) following this CANON.
- **Inspector** – audits the repo and tests for drift vs this document and flags risks.

When starting a new chat, the model should be told which role it is playing and to treat this file as the source of truth.

---

## 3. Invariants (do not casually break)

- The **spine** always follows:  
  `config/state → data adapters → build_snapshot → gatekeeper → GPT → risk_engine → execution_engine → state`.
- **Risk is downward‑only capped**: risk %, leverage, and notional exposure can only be reduced relative to config/envelope, never increased.
- GPT **never talks directly to an exchange**. All external side‑effects go through `execution_engine.py` and an execution adapter.
- `state_memory.py` is the **only module** that reads/writes persistent JSON state on disk.
- Replay harness must reuse the same snapshot → gatekeeper → GPT → risk → execution flow as live (only data + execution adapters differ).
- Tests in `tests/` are treated as contracts; they must pass before merging significant changes.

---

## 4. Module Map (high level)

| File / Dir                     | Purpose |
|--------------------------------|---------|
| `config.py`                    | Runtime configuration (symbol, risk caps, intervals, paths, testnet mode, initial equity). |
| `engine.py`                    | Main orchestrator loop for live/demo trading. |
| `build_features.py`            | Builds snapshot dict from raw market data + shapes. Computes liquidity context from fractal swing points, market session classification, and timing state. |
| `shapes_module.py`             | Deterministic microstructure features (sweeps, FVGs, CHoCH, etc.). |
| `gatekeeper.py`                | Decides whether to call GPT on a given tick (movement, timing, danger). Enforces rate limits (12 calls/hour, 60s spacing) and includes slow trend timeout (15min) to avoid silent periods during slow drifts. |
| `gpt_client.py`                | Wraps GPT call + prompt + JSON parsing into a `GptDecision`. Parses "rationale" field from GPT response (with fallback to "notes" for compatibility) and maps to `GptDecision.notes`. |
| `risk_envelope.py`             | Defines `RiskEnvelope` and helpers for computing envelope limits. |
| `risk_engine.py`               | Applies risk rules and envelopes to a GPT decision → `RiskDecision`. Logs full risk_envelope (including note) and RiskDecision details (size, leverage, stop distances) for every trade decision. |
| `execution_engine.py`         | Takes a `RiskDecision` and routes to an execution adapter. Logs execution events and traces including risk_envelope and RiskDecision info for approved trades. |
| `adapters/base_execution_adapter.py` | Base interface for execution adapters. |
| `adapters/mock_execution_adapter.py` | Demo/mock execution adapter for testing. |
| `adapters/replay_execution_adapter.py` | Replay adapter for backtests; simulates fills & positions. |
| `adapters/example_data_adapter.py`     | Example candle data adapter. |
| `adapters/liqdata.py`         | Hyperliquid data adapter for real market data (testnet/mainnet). |
| `adapters/liqexec.py`         | Hyperliquid execution adapter for real testnet order placement (market orders only, testnet-only safety). |
| `replay_engine.py`            | Backtest harness using the same spine plus replay adapters. |
| `state_memory.py`             | Persistent JSON state management (load/validate/save). |
| `logger_utils.py`             | Central logging setup. |
| `run_testnet.py`              | Testnet runner script for Hyperliquid testnet trading with safety banners and configuration. |
| `tests/`                      | Unit and integration tests for snapshot, gatekeeper, GPT client, risk envelope/engine, execution, replay parity, Hyperliquid adapters, etc. |

---

## 5. Core Contracts

These contracts are what matter for GPT/risk/execution integration.  
If you change them, update this file and the tests.

### 5.1 Snapshot schema (v1)

`build_snapshot` must produce a dict including at least:

- `symbol`: str
- `timestamp`: float (unix seconds)
- `price`: float (last/close)
- `trend`: str (`"up" | "down" | "sideways" | "unknown"`)
- `range_position`: str (`"extreme_low" | "low" | "mid" | "high" | "extreme_high" | "unknown"`)
- `volatility_mode`: str (`"low" | "normal" | "high" | "explosive" | "unknown"`)
- `flow`: dict with `funding`, `open_interest`, `skew`, `skew_bias`
- `microstructure`: dict with keys such as `sweep_up`, `sweep_down`, `fvg_up`, `fvg_down`, `choch_direction`, `compression_active`, `shape_bias`, `shape_score`
- `liquidity_context`: dict with `liquidity_above` (float or None) and `liquidity_below` (float or None) - computed from fractal swing points and recent extremes
- `fib_context`: dict with `macro_zone` and `micro_zone` (e.g. `"macro_discount" | "macro_premium"`)
- `htf_context`: dict with `trend_1h` and `range_pos_1h` (higher-timeframe context; currently placeholder)
- `danger_mode`: bool (true when environment is dangerous; risk should clamp hard)
- `timing_state`: str (`"avoid" | "cautious" | "normal" | "aggressive" | "unknown"`) - derived from market session
- `market_session`: str (`"ASIA" | "EUROPE" | "US" | "OFF_HOURS"`) - raw session label based on UTC hour
- `recent_price_path`: dict with `ret_1`, `ret_5`, `ret_15`, `impulse_state`, `lookback_bars`
- `risk_context`: dict with `equity`, `max_drawdown`, `open_positions_summary`, `last_action`, `last_confidence`
- `risk_envelope`: dict with risk limits (`max_notional`, `max_leverage`, `max_risk_per_trade_pct`, etc.)
- `since_last_gpt`: dict with time/price/equity changes since last GPT call
- `gpt_state_note`: str or None (short text note from prior GPT decisions)

Implementation detail may extend this; tests and GPT prompt should always assume these fields exist.

---

### 5.2 GPT Action schema (v1)

GPT must return a JSON object shaped as:

```json
{
  "action": "long" | "short" | "flat",
  "size": 0.0_to_1.0,
  "confidence": 0.0_to_1.0,
  "rationale": "short explanation string"
}
```

**Field consistency:**
- The `rationale` field is the canonical name in the GPT prompt (`brain.txt`) and JSON response.
- `gpt_client.py` parses `rationale` from the GPT response and maps it to `GptDecision.notes` internally.
- The rationale flows through: GPT response → `GptDecision.notes` → `state["gpt_state_note"]` → next snapshot's `gpt_state_note` field.
- For backward compatibility, `gpt_client.py` falls back to parsing `notes` if `rationale` is missing.

**Notes:**
- `size` is a fraction of max allowed size, not absolute notional.
- `confidence` is interpreted as model conviction; risk engine can further scale based on regime.
- On any parse failure, `gpt_client` must fall back to `action="flat"`, `size=0.0`, `confidence=0.0` with a clear log entry.

5.3 RiskEnvelope schema (v1)
RiskEnvelope defines caps for this decision:

max_leverage: float

max_notional: float (absolute value cap for position notional)

max_risk_per_trade_pct: float (max % of equity to risk on the trade)

min_stop_distance_pct: float (minimum stop distance as fraction of price)

max_stop_distance_pct: float (maximum stop distance as fraction of price)

max_daily_loss_pct: float (maximum daily loss percentage)

note: string (explains why envelope was tightened, e.g. "baseline_vol;timing_normal", "trim_for_vol;timing_cautious", "danger_mode")

Risk engine must never exceed these limits even if config and GPT suggest otherwise. The envelope and its note are logged for every trade decision to enable audit of risk behavior during testnet runs.

5.4 RiskDecision / Decision object (v1)
Internal object passed from risk engine to execution engine:

python
Copy code
{
  "approved": bool,
  "side": "long" | "short" | "flat",
  "position_size": float,        # units/contracts
  "leverage": float,
  "notional": float,
  "stop_loss_price": float | null,
  "take_profit_price": float | null,
  "reason": str,                 # why approved/rejected
  "envelope": { ... },           # snapshot of RiskEnvelope used
  "gpt_action": { ... },         # raw GPT action payload for logging
}
Execution engine must treat approved=False or side="flat" as “do nothing”.

6. Phases & Tasks
6.1 Phase 1 – Core Spine Hardening & Observability
Status: ✅ Complete

Highlights:

Introduced run_id and snapshot_id plumbing and logging.

Implemented snapshot validation and state versioning in state_memory.py.

Hardened engine loop with safe error handling and timing probes.

Ensured replay vs live parity path and cleaned up logging.

(Phase 1 is frozen; only adjust if a serious bug is found.)

6.2 Phase 2 – GPT Policy + Risk Brain v1
Status: 🚧 In progress

Active tasks:

 Restore config.py and get pytest collecting tests without import errors.
(We now have a stable baseline after the HL refactor; this unblocks all other work.)

 ✅ Lock GPT Action schema v1 in code and prompt
(Ensure gpt_client.py and the brain prompt both enforce the action/size/confidence/rationale contract so future changes don't silently break parsing.)
**Status:** Complete. Fixed inconsistency where brain.txt specified "rationale" but gpt_client.py parsed "notes". Now parses "rationale" with fallback to "notes" for compatibility. Verified end-to-end flow with tests.

 Stabilize RiskEnvelope schema and validation
(Align risk_envelope.py and risk_engine.py on the exact fields and make tests fail loudly if an envelope is missing or malformed.)

 Implement unified Decision object through engine → risk → execution
(Replace any ad‑hoc dicts with one normalized structure so logs, replay, and live runs all see the same decision shape.)

 Harden GPT JSON parsing + add tests
(Update gpt_client.call_gpt to handle malformed responses gracefully and add tests so that any format drift is caught immediately.)
**Progress:** Added tests for rationale/notes field parsing and end-to-end flow verification. General malformed response handling may need additional hardening.

 Ensure replay parity trace is implemented and tested
(Have replay_engine emit a parity trace structure and make tests assert that live vs replay see identical decisions for the same data.)

 Expand and document snapshot schema in build_features.py
(Make sure all required fields listed above are actually present and add tests so new features don’t break the snapshot shape.)

 Introduce Safe Mode for GPT failures
(If GPT errors repeatedly, engine should switch into a “flat only” safe mode and log this state until manually cleared.)

When this phase is complete, the snapshot → GPT → risk → execution contract should be stable enough that Phase 3 (analytics/sweeps) can rely on it.

6.3 Phase 3 – Research‑Grade Replay & Analytics
Status: ⏳ Not started (planning only)

High‑level goals:

Multiple replay modes (single run, parameter sweeps, brain variants).

Exportable results (CSV/JSON) for post‑analysis.

Visualizations / summaries for equity curve, drawdown, regime performance.

Detailed tasks for Phase 3 will be defined after Phase 2 is locked.

6.4 Later Phases (stubs)
Phase 4 – Stateful “Moving Picture” GPT Agent
Add rolling memory and regime summaries for GPT while preserving v1 contract.

Phase 5 – Hyperliquid SIM Integration
**Status:** 🚧 Partially complete
- ✅ Hyperliquid data adapter implemented (testnet/mainnet)
- ✅ Hyperliquid testnet execution adapter implemented (market orders only, testnet-only)
- ⏳ Replay integration with Hyperliquid data adapter
- ⏳ Mainnet execution adapter (currently disabled for safety)

Phase 6 – Hyperliquid Testnet / Live under Strict Envelopes
**Status:** ⏳ Not started
- Add budget caps, daily loss limits, circuit breakers, and kill switch guarding all external orders.
- Mainnet execution requires explicit approval and additional safety rails beyond testnet.

7. Editing Policy (for Craftsman)
Tier 1 files (core, high‑risk):
engine.py, build_features.py, gatekeeper.py, gpt_client.py, risk_envelope.py, risk_engine.py, execution_engine.py, state_memory.py, replay_engine.py
→ Prefer full‑file replacements when making non‑trivial changes. Keep structure clean.

Tier 2 files (support/tests/utilities):
config.py, adapters, tests under tests/, small utils
→ Localized patches are allowed; still keep functions cohesive.

All changes must keep tests in tests/ passing or add/update tests as part of the same task.

8. LLM Usage Notes
When using LLMs with this repo:

Always read TW_CANON.md and TW_LOG.md first and confirm “synced”.

Respect the invariants and contracts in this file; do not invent new architecture without an explicit Architect‑mode decision.

Foreman‑mode should update the Phase/Tasks section here when tasks are completed or added.

Craftsman should, when finished with a task, propose diff updates to this file (Phase/Tasks) and to TW_LOG.md summarizing what changed.

---

## 9. Standard Sync / “Commit and Push” Ritual

When I say “commit and push” or “sync repos”, workers must follow this workflow unless explicitly told otherwise.

### 9.1 Phase3-node (VPS → GitHub)

Repository: `/root/tradewarrior_demo` on phase3-node.

Standard sync command sequence:

```bash
cd /root/tradewarrior_demo
git status
git add -A
git commit -m "update from phase3-node" || echo "no changes to commit"
git push origin main
This stages all changes, creates a simple sync commit if needed, and pushes main to GitHub via SSH.

9.2 Local Windows repo (GitHub → laptop)
Repository: C:\Dev\tradewarrior_demo on my Windows laptop.

Standard sync command sequence:

powershell
Copy code
cd C:\Dev\tradewarrior_demo
git status
git pull origin main
This updates the local main branch to match GitHub (and therefore phase3-node).

9.3 LLM Behavior Requirements
When asked to “commit and push”, “sync repos”, or similar, assume this ritual by default.

Ask for permission only when running important actions (e.g., destructive operations, force-push, stash/drop, or changing remotes).

After any significant sync or code change, update TW_LOG.md with a new one-line event and ensure TW_CANON.md stays accurate if any contracts or invariants were touched.

This document is the single source of truth for how TradeWarrior is supposed to work.