# TradeWarrior Canon
_Last updated: 2025-12-03_

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
| `build_features.py`            | Builds snapshot dict from raw market data + shapes. |
| `shapes_module.py`             | Deterministic microstructure features (sweeps, FVGs, CHoCH, etc.). |
| `gatekeeper.py`                | Decides whether to call GPT on a given tick (movement, timing, danger). |
| `gpt_client.py`                | Wraps GPT call + prompt + JSON parsing into a `GptDecision`. |
| `risk_envelope.py`             | Defines `RiskEnvelope` and helpers for computing envelope limits. |
| `risk_engine.py`               | Applies risk rules and envelopes to a GPT decision → `RiskDecision`. |
| `execution_engine.py`         | Takes a `RiskDecision` and routes to an execution adapter. |
| `adapters/base_execution_adapter.py` | Base interface for execution adapters. |
| `adapters/mock_execution_adapter.py` | Demo/mock execution adapter for testing. |
| `adapters/replay_execution_adapter.py` | Replay adapter for backtests; simulates fills & positions. |
| `adapters/example_data_adapter.py`     | Example candle data adapter. |
| `adapters/hyperliquid_*`      | Hyperliquid data/execution adapters (SIM/TESTNET). |
| `replay_engine.py`            | Backtest harness using the same spine plus replay adapters. |
| `state_memory.py`             | Persistent JSON state management (load/validate/save). |
| `logger_utils.py`             | Central logging setup. |
| `run_testnet.py`              | Testnet runner script for Hyperliquid testnet trading with safety banners and configuration. |
| `tests/`                      | Unit and integration tests for snapshot, gatekeeper, GPT client, risk envelope/engine, execution, replay parity, etc. |

---

## 5. Core Contracts

These contracts are what matter for GPT/risk/execution integration.  
If you change them, update this file and the tests.

### 5.1 Snapshot schema (v1)

`build_snapshot` must produce a dict including at least:

- `symbol`: str
- `timestamp`: float or ISO string
- `price`: float (last/close)
- `trend`: str (`"up" | "down" | "flat"`)
- `range_position`: float in `[0.0, 1.0]` (where price sits in recent range)
- `volatility_mode`: str (e.g. `"low" | "normal" | "high"`)
- `shapes`: dict with keys such as `sweep_up`, `sweep_down`, `fvg_up`, `fvg_down`, `choch_direction`, `compression_active`, `shape_bias`, `shape_score`
- `recent_price_path`: small struct or list of recent candles/prices
- `fib_context`: any fib/structural context used
- `liquidity_context`: levels/liquidity hints
- `flow_context`: basic flow/volume hints
- `danger_mode`: bool (true when environment is dangerous; risk should clamp hard)
- `timing_state`: str (e.g. `"OK" | "AVOID"`)
- `gpt_state_note`: short text note coming from prior GPT decisions (agent memory stub)
- `risk_envelope`: optional pre‑computed `RiskEnvelope` snapshot

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
Notes:

size is a fraction of max allowed size, not absolute notional.

confidence is interpreted as model conviction; risk engine can further scale based on regime.

On any parse failure, gpt_client must fall back to action="flat", size=0.0, confidence=0.0 with a clear log entry.

5.3 RiskEnvelope schema (v1)
RiskEnvelope defines caps for this decision:

max_leverage: float

max_notional: float (absolute value cap for position notional)

max_risk_pct: float (max % of equity to risk on the trade)

danger_flag: bool (if true, risk engine should be extremely conservative or flat)

notes: optional string for debugging / logging

Risk engine must never exceed these limits even if config and GPT suggest otherwise.

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

 Lock GPT Action schema v1 in code and prompt
(Ensure gpt_client.py and the brain prompt both enforce the action/size/confidence/rationale contract so future changes don’t silently break parsing.)

 Stabilize RiskEnvelope schema and validation
(Align risk_envelope.py and risk_engine.py on the exact fields and make tests fail loudly if an envelope is missing or malformed.)

 Implement unified Decision object through engine → risk → execution
(Replace any ad‑hoc dicts with one normalized structure so logs, replay, and live runs all see the same decision shape.)

 Harden GPT JSON parsing + add tests
(Update gpt_client.call_gpt to handle malformed responses gracefully and add tests so that any format drift is caught immediately.)

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
Wire Hyperliquid data + execution adapters into both live and replay in SIM/testnet mode.

Phase 6 – Hyperliquid Testnet / Live under Strict Envelopes
Add budget caps, daily loss limits, circuit breakers, and kill switch guarding all external orders.

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

This document is the single source of truth for how TradeWarrior is supposed to work.

yaml
Copy code

---

### `TW_LOG.md` (full replacement)

```markdown
# TradeWarrior Log

This file is a terse, chronological log of important decisions and milestones.  
Each line should be one clear event; details belong in commits and TW_CANON.md.

---

- 2025-11-26 – Initialized TradeWarrior repo, added gitignore and removed pycache.  
- 2025-11-30 – Added Hyperliquid adapters (placeholders), refactored data routing, and cleaned up stray SSH config and state/log files.  
- 2025-12-01 – Completed initial HL refactor: introduced risk envelope, replay execution adapter, and expanded tests for replay and risk engine.  
- 2025-12-01 – Phase 1 spine hardening commit: added run_id/snapshot_id, improved logging, and replay exports.  
- 2025-12-02 – Fixed broken local environment by restoring `config.py` and getting pytest to collect tests cleanly again.  
- 2025-12-02 – Created TW_CANON.md and TW_LOG.md as the canonical truth layer for TradeWarrior.  
- 2025-12-02 – Entered Phase 2 (GPT Policy + Risk Brain v1) and defined v1 contra