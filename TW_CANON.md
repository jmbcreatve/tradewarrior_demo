TradeWarrior Canon

Last updated: 2025-12-04

1. Overview

TradeWarrior is an automated trading agent in Python.

Core spine (must stay intact):

config/state → data adapters → build_snapshot → gatekeeper → GPT → risk_engine → execution_engine → state

Replay uses the same spine with replay adapters.

Goals:

Research‑grade replay + live parity.

Explicit contract between snapshot → GPT → risk → execution.

Strict downward‑only risk envelopes around GPT suggestions.

Eventually: safe Hyperliquid testnet/live behind hard limits.

2. Roles & Workflow

We use Architect → Construction Crew:

Architect: defines phases, invariants, and contracts as short worker prompts.

Construction Crew: edits code, runs tests, updates TW_CANON.md + TW_LOG.md following those prompts.

All LLM‑driven code changes:

Must come from an Architect prompt.

Must preserve the spine + downward‑only risk rules.

Must treat tests as non‑negotiable contracts.

3. Invariants (do not break)

**Spine order is fixed:**
config/state → data adapters → snapshot → gatekeeper → GPT → risk → execution → state.

**Shared tick function for live/replay parity:**
Both engine.py and replay_engine.py use the same `run_spine_tick()` function (defined in engine.py) for the core decision spine: gatekeeper → GPT → risk → execution. This ensures identical decision logic regardless of whether running live or in replay mode. The only differences are the adapters provided (live data/execution vs mock/replay).

**Risk is capped downward‑only:** risk %, leverage, and notional can only be ≤ config/envelope caps, never above.

**GPT never talks directly to exchanges.** All side‑effects go through execution_engine.py and an execution adapter.

**state_memory.py is the ONLY module that reads/writes persistent JSON state.** No other module may write to state files. Replay exports (CSV, JSONL) go to analytics directories and are clearly separate from state.

**Replay harness reuses the exact snapshot → gatekeeper → GPT → risk → execution flow** via the shared `run_spine_tick()` function. Only adapters differ (replay execution adapter, optional GPT stub).

**Parity trace includes full decision context:** timestamp, price, gpt_action, approved, side, position_size, leverage, risk_envelope summary, and execution_status. This enables auditing and verification that replay matches live decisions.

Tests under tests/ are contracts; significant changes must keep them green or update tests in the same task.

4. Module Map (short)

config.py – Runtime config (symbol, risk caps, intervals, paths, testnet mode, initial equity). Environment-agnostic: does not read .env files or environment variables directly (except optional GPT model override). All secrets must come from OS environment variables at runtime, not from .env files.

engine.py – Main live/demo loop. **Exports `run_spine_tick()` and `SpineTickResult`** - the shared single-tick function used by both live and replay to ensure decision parity.

build_features.py – Builds snapshot dict from data + shapes; adds liquidity/session/timing context. Validates against TW_CANON 5.1 schema.

schemas.py – Defines MarketSnapshot, GptDecision, RiskDecision dataclasses and `validate_snapshot_dict()` for schema validation.

shapes_module.py – Deterministic microstructure features (sweeps, FVGs, CHoCH, etc.).

gatekeeper.py – Decides whether to call GPT; enforces ≤ 12 calls/hour, ≥ 60s spacing, slow‑trend timeout (~15 min). Returns dict with should_call_gpt and reason.

gpt_client.py – Wraps GPT call + prompt + JSON parse into GptDecision. Uses rationale → GptDecision.notes, with fallback to notes. Validates OPENAI_API_KEY at startup. Tracks errors for safe mode.

risk_envelope.py – Defines RiskEnvelope and helpers to compute limits.

risk_engine.py – Applies envelopes and rules to GPT decisions → RiskDecision. **Logs full RiskEnvelope (including note) and RiskDecision for every trade decision**, both approved and rejected.

execution_engine.py – Consumes RiskDecision, calls execution adapter. **Logs full risk_envelope and RiskDecision** for auditability.

adapters/*.py – Data and execution adapters:

mock_execution_adapter.py – Demo/mock execution.

replay_execution_adapter.py – Simulated fills for replay (defined in replay_engine.py).

example_data_adapter.py – Example candle data.

liqdata.py – Hyperliquid data (testnet/mainnet).

liqexec.py – Hyperliquid testnet execution (market‑only, reads HL_TESTNET_PRIVATE_KEY from OS environment variables). Places protective stop/TP trigger orders (grouped OCO); fails safe by flattening/returning no_trade if protection cannot be installed.

replay_engine.py – Backtest harness; **uses shared `run_spine_tick()` for live/replay parity**. Emits trades/equity + parity trace with full decision context.

state_memory.py – **ONLY module that writes persistent JSON state.** Load/validate/save state, GPT safe mode helpers, daily tracking.

logger_utils.py – Logging setup.

safety_utils.py – Kill switch and circuit breaker checks.

run_testnet.py – Hyperliquid testnet runner, continuous loop, daily P&L + safety logs, loss limits, kill switch. Reads secrets from OS environment variables (HL_TESTNET_PRIVATE_KEY, OPENAI_API_KEY). Optional .env file loading is a convenience helper only (explicit flag/twp3_testnet.env); no baked-in local_env.txt fallback.

tests/ – Snapshot schema compliance, gatekeeper, GPT client, risk envelope/engine, execution, **replay parity**, adapters, safety utils, GPT safe mode. Tests are contracts.

5. Core Contracts

If these change, update this file and tests.

5.1 Snapshot schema (v1)

build_snapshot must return a dict with at least:

symbol: str

timestamp: float (unix seconds)

price: float

trend: "up" | "down" | "sideways" | "unknown"

range_position: "extreme_low" | "low" | "mid" | "high" | "extreme_high" | "unknown"

volatility_mode: "low" | "normal" | "high" | "explosive" | "unknown"

flow: dict with funding, open_interest, skew, skew_bias

microstructure: dict with at least: sweep_up, sweep_down, fvg_up, fvg_down, choch_direction, compression_active, shape_bias, shape_score

liquidity_context: dict with liquidity_above, liquidity_below (floats or None)

fib_context: dict with macro_zone, micro_zone

htf_context: dict with trend_1h, range_pos_1h (placeholder is fine but keys must exist)

danger_mode: bool

timing_state: "avoid" | "cautious" | "normal" | "aggressive" | "unknown"

market_session: "ASIA" | "EUROPE" | "US" | "OFF_HOURS"

recent_price_path: dict with ret_1, ret_5, ret_15, impulse_state, lookback_bars

risk_context: dict with equity, max_drawdown, open_positions_summary, last_action, last_confidence (last_action/confidence come from the last risk/execution outcome, not raw GPT proposals)

risk_envelope: dict (see 5.3)

since_last_gpt: dict with time_since_last_gpt_sec, price_change_pct_since_last_gpt, equity_change_since_last_gpt, trades_since_last_gpt (state is refreshed after every GPT call; trades_since_last_gpt increments only on executed trades)

gpt_state_note: str or None

Implementation can add more fields, but tests/GPT prompt assume these exist.

5.2 GPT Action schema (v1)

GPT must return JSON:

{
  "action": "long" | "short" | "flat",
  "size": 0.0_to_1.0,
  "confidence": 0.0_to_1.0,
  "rationale": "short explanation"
}


Rules:

rationale is the canonical field in prompt + parse.

gpt_client.py maps rationale → GptDecision.notes, falling back to notes if needed.

Rationale path: GPT → GptDecision.notes → state["gpt_state_note"] → next snapshot’s gpt_state_note.

size is a fraction of max allowed size, not absolute.

confidence is conviction; risk engine may scale further.

On parse failure, gpt_client must fall back to:
action="flat", size=0.0, confidence=0.0 and log clearly.

5.3 RiskEnvelope schema (v1)

Caps for a single decision:

max_leverage: float

max_notional: float

max_risk_per_trade_pct: float (of equity)

min_stop_distance_pct: float (of price)

max_stop_distance_pct: float

max_daily_loss_pct: float

note: str (e.g. "baseline_vol;timing_normal", "trim_for_vol;timing_cautious", "danger_mode")

Risk engine must never exceed these limits even if config or GPT suggests more.
Stop distances are clamped into [min_stop_distance_pct, max_stop_distance_pct] before deriving stop_loss/take_profit.
Envelope + note must be logged for every trade decision.

5.4 RiskDecision / Decision object (v1)

Internal object from risk → execution:

{
  "approved": bool,
  "side": "long" | "short" | "flat",
  "position_size": float,       # units/contracts
  "leverage": float,
  "notional": float,
  "stop_loss_price": float | None,
  "take_profit_price": float | None,
  "reason": str,                # why approved/rejected
  "envelope": { ... },          # snapshot of RiskEnvelope
  "gpt_action": { ... }         # raw GPT action for logging
}


Execution engine must treat approved=False or side="flat" as “do nothing”.

6. Phases & Tasks
Phase 1 – Spine + Observability

Status: ✅ Complete & frozen

run_id/snapshot_id plumbing.

Snapshot validation + state versioning.

Hardened engine loop, timing probes.

Replay vs live parity and logging.

Only change if serious bug.

Phase 2 – GPT Policy + Risk Brain v1

Status: 🚧 In progress**

Key tasks:

Restore config.py, get pytest collecting cleanly.

Schema lock: GPT Action v1 enforced in gpt_client.py and brain prompt (done: rationale vs notes consistency).

Stabilize RiskEnvelope schema + validation in risk_envelope.py / risk_engine.py; tests must fail loudly on bad envelopes.

Implement unified Decision object engine → risk → execution (no ad‑hoc dicts in the spine).

Harden GPT JSON parsing + tests for malformed responses.

Ensure replay parity trace exists and is tested (live vs replay same decisions on same data).

Expand + test snapshot schema in build_features.py to match section 5.1.

✅ **Safe Mode for GPT failures**: When GPT returns 3+ consecutive errors within 5 minutes, the system automatically enters safe mode—skipping all GPT calls and forcing FLAT decisions with clear logging. Safe mode persists until manually cleared via `--clear-gpt-safe-mode` flag or by editing state. Error counts reset on successful GPT calls but do not auto-clear safe mode.

When Phase 2 is done, snapshot → GPT → risk → execution should be stable enough for research sweeps.

Phase 3 – Replay & Analytics

Status: ⏳ Planned**

Multiple replay modes (single run, parameter sweeps, brain variants).

Exportable results (CSV/JSON).

Summaries/plots for equity, drawdown, regime performance.

Phase 4 – Stateful GPT Agent

Status: Stub**

Rolling memory + regime summaries for GPT.

Preserve v1 contracts while adding statefulness.

Phase 5 – Hyperliquid SIM Integration

Status: 🚧 Partial**

✅ HL data adapter (testnet/mainnet).

✅ HL testnet execution adapter (market‑only, Wallet+Exchange, HL_TESTNET_PRIVATE_KEY).

⏳ Replay integration with HL data.

⏳ Mainnet adapter (disabled until extra safety complete).

Phase 6 – Hyperliquid Testnet / Live

Status: ⏳ Not started**

Hard budget caps, daily loss limits, circuit breakers, kill switch for all external orders.

Mainnet requires explicit approval + extra safety rails.

7. Editing Policy

Tier 1 (core/high‑risk):
engine.py, build_features.py, gatekeeper.py, gpt_client.py, risk_envelope.py, risk_engine.py, execution_engine.py, state_memory.py, replay_engine.py
→ Prefer full‑file replacements for non‑trivial changes; keep structure clean.

Tier 2 (support/tests/utils):
config.py, adapters/, tests/, small utils
→ Local patches allowed; still keep functions cohesive.

All changes must keep tests/ green or include updated tests in the same task.

8. LLM Usage

Always read TW_CANON.md + TW_LOG.md first and confirm they’re “synced”.

Do not invent new architecture without explicit Architect instructions.

Foreman mode updates Phase/Tasks here when tasks complete or change.

Craftsman mode, after finishing a task:

Propose updates to Phase/Tasks in this file.

Append a one‑line event to TW_LOG.md describing the change.

9. Sync / “Commit and Push” Ritual

When asked to “commit and push”, “sync repos”, etc., assume:

9.1 VPS (phase3-node: /root/tradewarrior_demo)
cd /root/tradewarrior_demo
git status
git add -A
git commit -m "update from phase3-node" || echo "no changes to commit"
git push origin main

9.2 Local Windows (C:\Dev\tradewarrior_demo)
cd C:\Dev\tradewarrior_demo
git status
git pull origin main

9.3 LLM behavior

Ask for confirmation only for risky actions (force‑push, destructive ops, remotes).

After significant sync/code changes:

Update TW_LOG.md with a one‑liner.

Ensure TW_CANON.md matches any changed contracts.

This document is the single source of truth for how TradeWarrior is supposed to work.
