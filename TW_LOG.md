# TradeWarrior Canon

_Last updated: 2025-12-02_

This file is the **single source of truth** for how TradeWarrior works and how AI should interact with it.

---

## 1. What TradeWarrior Is

TradeWarrior is an automated trading agent built in Python.

It runs a deterministic loop:

`config + state → data adapters → build_snapshot → gatekeeper → GPT policy → risk_engine → execution_engine → updated state`

The same spine is reused by:

- **Live/Demo engine** (`engine.py`)
- **Replay/Backtest engine** (`replay_engine.py`)

---

## 2. Spine & Invariants (Do Not Break)

**Core spine (must stay in this order):**

1. Load `Config` (`config.py`) and state (`state_memory.py`).
2. Fetch market data via data adapters (`data_router` / adapters).
3. Build a `snapshot` (`build_features.py` + `shapes_module.py`).
4. Let `gatekeeper.py` decide if GPT should be called this tick.
5. If yes, call GPT via `gpt_client.py` → get a `GptDecision`.
6. Run `risk_engine.evaluate_risk(...)` with snapshot + GPT + envelope.
7. Pass the resulting `RiskDecision` into `execution_engine.py` and its execution adapter.
8. Update and save state via `state_memory.py`.

**Invariants:**

- GPT **never** talks directly to an exchange or execution API.
- Only `execution_engine.py` calls execution adapters (mock, replay, Hyperliquid, etc.).
- **Risk is downward-only:** `risk_engine` can reduce size/leverage or veto the trade, never increase risk beyond config/envelope.
- `state_memory.py` is the **only** module that reads/writes JSON state on disk.
- Replay must use the same snapshot → gatekeeper → GPT → risk → execution logic as live (only data/execution adapters differ).
- Tests in `tests/` are contracts; they must stay green (or be updated deliberately when contracts evolve).

---

## 3. Module Map

| File / Dir                          | Responsibility |
|------------------------------------|----------------|
| `config.py`                        | Runtime configuration: symbol, timeframe, risk caps, adapter IDs, paths, loop sleep, etc. |
| `engine.py`                        | Live/demo orchestrator; runs the core loop around one symbol/timeframe. |
| `build_features.py`                | Builds the `snapshot` dict from raw market data + state (trend, range_position, vol mode, shapes, etc.). |
| `shapes_module.py`                 | Deterministic, side-effect-free microstructure detection (sweeps, FVGs, CHOCH, bias/score, compression flags). |
| `gatekeeper.py`                    | Decides whether to call GPT based on movement, timing, danger, and call rate limits. |
| `gpt_client.py`                    | Builds the GPT prompt from brain + snapshot, calls the LLM, parses JSON into a `GptDecision`. |
| `risk_envelope.py`                 | Defines `RiskEnvelope` and computes envelope caps from config + snapshot. |
| `risk_engine.py`                   | Applies risk rules and envelope caps to GPT’s suggestion → returns `RiskDecision`. |
| `execution_engine.py`             | Translates `RiskDecision` into orders via an execution adapter and logs the result. |
| `adapters/base_execution_adapter.py` | Interface for execution adapters (mock, replay, Hyperliquid). |
| `adapters/mock_*`                 | Local mock adapters for demo/testing. |
| `adapters/hyperliquid_*`          | Hyperliquid SIM/testnet data + execution adapters (in progress). |
| `data_router.py` & data adapters  | Fetch recent candles and related market data for a symbol/timeframe. |
| `replay_engine.py`                | Backtest harness; runs candles through the spine using `ReplayExecutionAdapter`. |
| `state_memory.py`                 | Atomic JSON state load/save; schema normalization, defaults, and repair. |
| `logger_utils.py`                 | Central logging: console + `tradewarrior.log`. |
| `tests/`                          | Tests for snapshot, shapes, gatekeeper, GPT client, risk envelope/engine, execution, replay, parity. |

---

## 4. Core Contracts

These interfaces are the backbone between components.  
If you change them, you **must** update tests and this file.

### 4.1 Snapshot (input to gatekeeper/GPT/risk)

A `snapshot` dict MUST contain at least:

- `symbol`: `str`
- `timestamp`: float or ISO string
- `price`: float (current/close)
- `trend`: `"up" | "down" | "flat"`
- `range_position`: float in `[0.0, 1.0]`
- `volatility_mode`: `"low" | "normal" | "high"` (or similar)
- `shapes`: `dict` including keys like:
  - `shape_bias`: `"bull" | "bear" | "none"`
  - `shape_score`: float in `[0.0, 1.0]`
  - `sweep_up`, `sweep_down`, `fvg_up`, `fvg_down`: bool
  - `choch_direction`: `"up" | "down" | "none"`
  - `compression_active`: bool
- `recent_price_path`: compact representation of recent price movement
- `fib_context`, `liquidity_context`, `flow_context`: optional dicts/fields used by the brain
- `danger_mode`: bool
- `timing_state`: `"OK" | "AVOID"` (or similar enum)
- `risk_context`: includes at least equity, drawdown, open positions summary
- `gpt_state_note`: short text note carried from prior GPT calls (GPT “memory” stub)

Internally you can add more, but **anything referenced by gatekeeper, GPT, or risk must not disappear silently.**

---

### 4.2 GPT Action (output of `gpt_client`)

GPT is instructed to return strict JSON. After parsing, `gpt_client` must produce a `GptDecision` with:

- `action`: `"long" | "short" | "flat"`
- `size`: float in `[0.0, 1.0]`  
  _(fraction of max allowed position size for this context)_
- `confidence`: float in `[0.0, 1.0]`
- `rationale`: short text string (for logs/state, not for math)

Rules:

- On any parse/format failure, `gpt_client` returns a safe decision: `action="flat"`, `size=0.0`, `confidence=0.0`, with an error note and logs.
- `gpt_client` clamps invalid values (unknown `action` → `flat`, out-of-range `confidence` → clamp to `[0, 1]`).

---

### 4.3 RiskEnvelope (from `risk_envelope.py`)

`RiskEnvelope` is the per-decision cap object. It must have at least:

- `max_leverage`: float
- `max_notional`: float (max absolute notional exposure)
- `max_risk_pct`: float (max allowed risk as % of equity for this trade)
- `danger_flag`: bool (true if envelope effectively says “no new exposure”)
- `notes`: optional string for logs

Risk engine must **never exceed any of these** even if config/GPT are more aggressive.

---

### 4.4 RiskDecision / Decision object (from `risk_engine` to `execution_engine`)

The risk engine returns a `RiskDecision`/Decision dict with:

- `approved`: bool
- `side`: `"long" | "short" | "flat"`
- `position_size`: float (size in units/contracts)
- `leverage`: float
- `notional`: float
- `stop_loss_price`: float or `None`
- `take_profit_price`: float or `None`
- `reason`: short string (e.g. `"no_equity"`, `"envelope_forbids"`, `"ok"`)
- `envelope`: snapshot of the `RiskEnvelope` used
- `gpt_action`: the normalized GPT action dict (for logging/debugging)

`execution_engine` treats `approved=False` or `side="flat"` as “do nothing”.

---

## 5. Phases & Tasks (High-Level)

### Phase 1 – Core Spine Hardening & Observability  
**Status:** ✅ Complete (do not change unless there’s a critical bug)

- Centralized logging and log directory.
- Stable JSON state schema with atomic saves.
- Snapshot builder and validator wired into engine.
- Gatekeeper v1 with timing/danger + rate limits.
- Replay harness reusing the core spine.
- RiskEnvelope + RiskEngine enforcing downward-only caps.
- Basic test suite for snapshot, shapes, gatekeeper, GPT client, risk, replay, execution.

---

### Phase 2 – GPT Policy + Risk Brain v1  
**Status:** 🚧 In progress (Architect and Cursor should focus here)

Key outcomes:

- A **stable GPT → risk → execution contract** (Action schema, RiskEnvelope, RiskDecision).
- Robust GPT parsing and safe fallbacks.
- Replay and live parity for decisions.

Working task list (to be updated by Architect and Cursor):

- [x] Restore `config.py` and get `pytest` collecting tests without import errors.  
- [ ] Lock Action schema v1 in `gpt_client` and prompt (enforce `action/size/confidence/rationale`).  
- [ ] Stabilize RiskEnvelope fields and add validation tests.  
- [ ] Introduce unified Decision object and wire it through `engine.py`, `risk_engine.py`, `execution_engine.py`.  
- [ ] Harden GPT JSON parsing and add tests for malformed/partial responses.  
- [ ] Implement replay parity trace and tests (live vs replay see same decisions for same data + stubbed GPT).  
- [ ] Ensure `build_features.py` always emits the required snapshot fields and add snapshot schema tests.  
- [ ] Add Safe Mode for repeated GPT failures (auto “flat only” mode + log/state flag).

---

### Phase 3 – Research-Grade Replay & Analytics  
**Status:** ⏳ Not started

To be designed once Phase 2 is locked: multi-run sweeps, metrics exports, richer analytics.

### Phase 4 – Stateful “Moving-Picture” GPT  
**Status:** ⏳ Not started

Add rolling context/memory while preserving the v1 contract so replay and analytics don’t break.

### Phase 5 – Hyperliquid SIM Integration  
**Status:** ⏳ Not started

Wire HL data + execution in SIM/testnet mode only, behind the existing risk envelopes.

### Phase 6 – Hyperliquid Testnet/Live Under Strict Envelopes  
**Status:** ⏳ Not started

Hard budget caps, daily loss limits, circuit breakers, kill-switches around any real external trading.

---

## 6. Roles & Workflow (Simplified)

### 6.1 Architect (ChatGPT – this UI)

- **Only role for ChatGPT now.**
- Owns this file (CANON) and the high-level plan.
- Duties:
  - Define/adjust architecture, invariants, and contracts.
  - Define/adjust phases and high-level task lists.
  - Occasionally inspect designs/logs and propose course corrections.
- Never writes code or shell commands.  
- When architecture or phase plan changes, Architect updates this file and asks Cursor to adjust code/tests accordingly.

### 6.2 Cursor Worker (Agent Chat + Composer)

In practice, Cursor runs both “Foreman” and “Craftsman” behaviors:

- **Agent Chat:**
  - Reads this CANON file.
  - Breaks Phase 2 tasks into concrete code edits.
  - Plans changes across files.
  - Responds to your “DOC-UPDATE” ritual (see below) to keep CANON and the log in sync.

- **Composer / code editing:**
  - Applies the planned changes to code and tests.
  - Shows diffs and respects Tier 1 vs Tier 2 edit policy.
  - Runs tests when asked.

---

## 7. Editing Policy (for Cursor)

- **Tier 1 (core spine, high risk):**  
  `engine.py`, `build_features.py`, `gatekeeper.py`, `gpt_client.py`,  
  `risk_envelope.py`, `risk_engine.py`, `execution_engine.py`,  
  `state_memory.py`, `replay_engine.py`.

  → For non-trivial changes, **prefer full-file replacements** to keep structure clean and avoid patchy half-refactors.

- **Tier 2 (support/testing):**  
  `config.py`, adapters in `adapters/`, utilities, `logger_utils.py`, `tests/`.

  → Localized patches are allowed, but changes must stay coherent.

All significant changes must be accompanied by updated or new tests where appropriate.

---

## 8. Rituals (How Tools Sync to This Canon)

### 8.1 Architect Session (ChatGPT)

When starting / resuming work on TradeWarrior in ChatGPT:

1. Load the `tradewarrior_demo` repo (via GitHub connection).
2. Read `TW_CANON.md` (this file) and, if needed, the log file.
3. Confirm: “Synced with CANON as of <date>.”
4. Operate only as **Architect**:
   - Update phases/tasks.
   - Clarify or change contracts and invariants.
   - Produce high-level task lists for Cursor (but not code).

### 8.2 Cursor Session (Agent Chat)

In a new Cursor chat for this repo, first message:

> 1. Open `TW_CANON.md` in the repo root and read it fully.  
> 2. Acknowledge with: “Synced with CANON; ready for tasks.”  
> 3. You are the Cursor Worker (Foreman + Craftsman): you plan and apply code changes following CANON.

### 8.3 DOC-UPDATE Ritual (keeping this file evolving)

After a chunk of code work + passing tests, you tell Cursor:

> **DOC-UPDATE**:  
> 1. Look at the current Git diff (uncommitted changes).  
> 2. Update the “Phase 2 – GPT Policy + Risk Brain v1” checklist in `TW_CANON.md`:  
>    - Mark finished tasks as `[x]`, and/or refine/add bullets if the work doesn’t exactly match an existing line.  
> 3. Append a one-line summary to the log file (e.g., `TW_LOG.md`):  
>    “YYYY-MM-DD – short description of what changed.”  
> 4. Show me diffs for CANON and the log; do not touch other files.

You review diffs → commit code + updated CANON + log together.

---

This document is the **only architectural brain** for TradeWarrior.  
ChatGPT (Architect) updates the plan; Cursor reads and implements it.  
As long as this file stays current, you can kill any chat and still recover the whole project state in one shot.
