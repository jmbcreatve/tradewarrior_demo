TradeWarrior Canon

Last updated: 2025-12-09

1. Overview

TradeWarrior is an automated trading agent in Python.

There are now two engines in this repo:

- Classic spine engine (v1):
  config/state → data adapters → build_snapshot → gatekeeper → GPT → risk → execution → state  
  This is the original research/testnet engine driven by `engine.py` + `replay.py`.

- TW-5 thin engine:
  data → Tw5Snapshot → TW-5 gatekeeper → GPT/stub → TW-5 risk_clamp → (future executor) → state  
  This lives under `tw5/` and is designed to be a smaller, GPT-first planning layer that still uses the same
  config, state, and safety rails.

Replay for both engines uses the same idea: re-use the decision spine with replay adapters.

Goals:

- Research-grade replay + live parity for the classic spine.
- A tiny, human-like panel (Tw5Snapshot) and structured OrderPlan contract for TW-5.
- Explicit contracts between snapshot → GPT → risk → execution.
- Strict downward-only risk envelopes around GPT suggestions.
- Testnet-only outbound execution, behind hard limits and kill switches.

2. Roles & Workflow

We use Architect → Construction Crew:

- Architect: defines phases, invariants, and contracts as short worker prompts.
- Construction Crew: edits code, runs tests, updates TW_CANON.md + TW_LOG.md following those prompts.

All LLM-driven code changes:

- Must come from an Architect prompt.
- Must preserve the spine + downward-only risk rules.
- Must treat tests as non-negotiable contracts.
- Must respect TW-5 contracts when touching `tw5/`.

3. Invariants (do not break)

Classic spine:

- Spine order is fixed:
  config/state → data adapters → snapshot → gatekeeper → GPT → risk → execution → state.

- Shared tick function for live/replay parity:
  Both `engine.py` and `replay.py` use `run_spine_tick()` (defined in `engine.py`) for the core decision spine:
  gatekeeper → GPT → risk → execution. Only the adapters differ (live vs mock/replay).

- Risk is capped downward-only:
  risk %, leverage, and notional can only be ≤ config/envelope caps, never above.

- GPT never talks directly to exchanges:
  all side-effects go through `execution_engine.py` and an execution adapter.

- `state_memory.py` is the ONLY module that reads/writes persistent JSON state.
  No other module may write to state files. Replay exports (CSV, JSONL) go to analytics directories.

- Replay harness reuses the exact snapshot → gatekeeper → GPT → risk → execution flow via `run_spine_tick()`.

- Parity trace includes full decision context:
  timestamp, price, gpt_action, approved, side, position_size, leverage, risk_envelope summary, and execution_status.

TW-5 engine:

- TW-5 uses the same `Config` and `state` dict as the classic spine. TW-5-specific fields in state are namespaced
  (`tw5_last_snapshot`, `tw5_last_plan`, `tw5_last_clamp_result`, `tw5_gpt_call_history`, etc.).

- TW-5 may NOT write JSON directly; it must call `save_state()` from `state_memory.py` if persistence is desired.

- TW-5 risk clamp (`tw5/risk_clamp.py`) must obey:
  - kill switch + `trading_halted` circuit breaker
  - GPT safe mode (`is_gpt_safe_mode(state)`) → no new exposure
  - equity floor (downside only)
  - daily loss limit (downside only)
  - risk_per_trade + max_leverage caps from Config (optionally overridden by dedicated TW-5 config fields)
  It may only shrink size/stops or veto; it may never widen stops or increase leverage vs config.

- TW-5 GPT client must:
  - use OPENAI_API_KEY from OS env (no .env loading)
  - return a structured OrderPlan or a flat “error” plan on failure
  - feed GPT error/success into state via `record_gpt_error/record_gpt_success` (same safe-mode machinery as classic).

Safety (shared):

- Kill switch file (`.tradewarrior_kill_switch`) + `trading_halted` flag must be honoured before placing real orders.
- Daily loss caps, GPT safe mode, and kill switch are higher priority than any trading signal.
- No Hyperliquid mainnet execution is allowed until a dedicated future phase explicitly enables it.

Tests under `tests/` are contracts; significant changes must keep them green or update tests in the same task.

4. Module Map (short)

Classic spine:

- config.py  
  Runtime config (symbol, risk caps, intervals, paths, testnet mode, initial equity). Environment-agnostic:
  does not read .env files. All secrets enter via OS env. `load_testnet_config()` builds a Hyperliquid testnet Config
  with conservative risk and `state_testnet.json` as the state file.

- engine.py  
  Main live/demo loop. Exports:
  - `run_spine_tick()` and `SpineTickResult` (shared live/replay decision spine).
  - `_run_tick()`, `run_once()`, `run_forever()` wrappers.
  Handles kill switch / `trading_halted` checks, daily tracking init, execution adapter selection,
  equity updates from execution, daily loss circuit breaker, and state persistence.

- build_features.py  
  Builds the classic `snapshot` dict from candles + shapes; adds liquidity/session/timing context and
  risk_context / since_last_gpt fields. Enforced by snapshot schema tests.

- schemas.py  
  Defines:
  - MarketSnapshot schema validation helper
  - GptDecision, RiskDecision dataclasses
  - `validate_snapshot_dict()` for schema validation.

- shapes_module.py  
  Deterministic microstructure features (sweeps, FVGs, CHoCH, equal highs/lows, etc.).

- gatekeeper.py  
  Decides whether to call GPT for the classic spine; enforces:
  - ≤ 12 calls/hour
  - ≥ 60s spacing
  - slow-trend timeout (~15 min) so GPT is eventually woken in grind regimes.

- gpt_client.py  
  Wraps GPT call + prompt + JSON parse into a `GptDecision`. Uses `brain.txt` as the system prompt.
  - Validates OPENAI_API_KEY at startup.
  - Maps GPT `rationale` → `GptDecision.notes`.
  - Logs a compact decision event.
  - On failure, logs and returns a flat “error fallback” decision, feeding errors into GPT safe mode tracking.

- risk_envelope.py  
  Defines `RiskEnvelope` and `compute_risk_envelope()`:
  - Caps leverage/notional/risk_per_trade/daily_loss, with volatility + timing trims.
  - All envelopes are downward-only vs Config caps.

- risk_engine.py  
  Applies envelopes and rules to GPT decisions → `RiskDecision`. Logs:
  - full RiskEnvelope (including `note` explaining trims like `trim_for_vol;timing_cautious`)
  - full RiskDecision (approved + rejected)
  Implements daily loss checks and replay-only chop/high guards.

- execution_engine.py  
  Consumes `RiskDecision`, calls the active execution adapter.
  - Logs risk_envelope and RiskDecision for auditing.
  - Treats `approved=False` or `side="flat"` as “do nothing”.

- adapters/  
  Data + execution adapters:
  - `mock_data_adapter.py`, `mock_execution_adapter.py` – local SIM.
  - `liqdata.py` – Hyperliquid data (testnet/mainnet endpoints, but only testnet is wired in runners).
  - `liqexec.py` – Hyperliquid testnet execution via current SDK and `HL_TESTNET_PRIVATE_KEY` env var.
  - replay helpers for offline backtests.

- replay.py  
  Backtest harness for the classic spine:
  - loads candles (CSV/JSON)
  - builds snapshots
  - uses `run_spine_tick()` for parity with live
  - tracks equity/trades
  - writes per-run exports under `analytics/runs/<run_id>/` (trades/equity/parity/features/config/summary).

- state_memory.py  
  Single source of truth for JSON state I/O. Responsibilities:
  - defaults + schema normalisation
  - daily P&L tracking and reset
  - `trading_halted` circuit breaker flag
  - GPT safe mode: error counting, activation, manual clearing
  - run_id/snapshot_id plumbing
  No one else writes state files.

- logger_utils.py  
  Logging setup: console + single rotating file (`logs/tradewarrior.log`) with rollover disabled for Windows safety.

- safety_utils.py  
  Kill switch + `trading_halted` helpers:
  - `check_kill_switch()` on `.tradewarrior_kill_switch` in project root.
  - `check_trading_halted(state)` merges kill switch + circuit breaker.

- run_testnet.py  
  Hyperliquid testnet runner. Responsibilities:
  - print a loud TESTNET banner
  - optional .env loading for local dev (does not override OS env)
  - validate OPENAI_API_KEY
  - build testnet Config (`load_testnet_config`)
  - load/reset state, initialise daily tracking
  - continuous loop with periodic safety + P&L status prints
  - respect kill switch, daily loss cap, and GPT safe mode
  - single-tick mode for testing.

- tests/  
  Contracts for:
  - snapshot schema
  - gatekeeper
  - GPT client
  - risk envelope / risk engine
  - execution engine
  - replay parity + exports
  - safety (kill switch, daily loss)
  - GPT safe mode
  - Hyperliquid adapters (mocked).

TW-5 modules (`tw5/`):

- `tw5/schemas.py`
  Defines TW-5 contracts:
  - `RunMode` enum: `ADVISOR`, `AGENT_CONFIRM`, `AUTO`.
  - `Tw5Snapshot`: tiny human-like snapshot panel (trend_1h/4h, 7d range position, swing/fibs, vol_mode + atr_pct,
    last_impulse, simple sweep flags, and a light position view).
  - `OrderPlan`: GPT/stub output (mode/side + legs + max_total_size_frac + confidence + rationale).
  - `OrderLeg` + `TPLevel`: fib-style legs with limit/market entries, stops, and TP ladder.
  - `RiskClampResult`: approved/vetoed, reason, original_plan, clamped_plan.
  - `PendingOrder`: tracks unfilled limit orders in replay (plan, clamp_result, created_ts, created_idx, expires_idx, snapshot).

- `tw5/snapshot_builder.py`
  Builds `Tw5Snapshot` from:
  - config (symbol/timeframe)
  - market_data["candles"]
  - state (via `last_execution_result` to infer position side/size/entry)
  Implements:
  - trend_1h/4h via simple close-to-close change over fixed bar windows
  - 7d-style range and range_position_7d (extreme_low/low/mid/high/extreme_high)
  - swing_low/high from HISTORICAL bars only (current bar excluded to prevent hindsight bias)
  - fib levels from swing
  - ATR-like volatility from avg |ret| and vol_mode (low/normal/high/explosive)
  - last_impulse_direction/size_pct over a short lookback
  - sweeps off in v1.

- `tw5/stub.py`  
  Deterministic baseline policy:
  - follows 4h trend (up → fib-based longs, down → fib-based shorts, else flat)
  - uses fib pullbacks for limit entries, stops beyond swing extremes
  - builds 1R/2R TP ladders per leg
  - leaves notional sizing to the risk clamp.

- `tw5/gatekeeper.py`  
  Simple TW-5 wake-up logic:
  - snapshot sanity (price > 0, vol_mode known, at least one trend known)
  - per-hour GPT call cap
  - minimum seconds between calls
  - triggers on:
    - |price move| in ATR units ≥ configurable threshold
    - OR range_position_7d at extremes (extreme_low/extreme_high).

- `tw5/risk_clamp.py`  
  Down-only risk clamp for TW-5 OrderPlans:
  - first checks kill switch / `trading_halted`
  - blocks non-flat plans in GPT safe mode
  - blocks if price invalid, equity ≤ 0, equity floor breached, or daily loss exceeded
  - shrinks stops that are too wide (never widens stops)
  - normalises leg size_fracs
  - computes a risk factor from (entry/stop distances) and caps `max_total_size_frac` so
    trade risk ≤ equity * risk_per_trade_pct with leverage ≤ effective max_leverage
  - respects optional TW-5 overrides: `tw5_risk_per_trade_pct`, `tw5_max_leverage`, `tw5_max_stop_pct`,
    `tw5_equity_floor_pct`, `tw5_max_daily_loss_pct`
  - returns an approved flat plan when vetoing.

- `tw5/gpt_client.py`  
  TW-5-specific GPT client:
  - Uses its own inline system prompt (not `brain.txt`) and `Tw5Snapshot`/OrderPlan schema.
  - Requires OPENAI_API_KEY; otherwise returns a flat “error” plan and records error.
  - On success, parses JSON into an OrderPlan and records GPT success for safe-mode tracking.
  - On failure (API error / JSON error), returns a flat error plan and records GPT error.

- `tw5/engine_tw5.py`  
  TW-5 engine tick:
  - `run_tw5_tick()`:
    1) fetches market data via `get_market_data()`
    2) builds a `Tw5Snapshot`
    3) reconstructs previous Tw5Snapshot from state (if present)
    4) computes GPT call rate (last_gpt_ts + history) and reads TW-5 thresholds from Config
    5) calls `should_call_gpt_tw5()`; if it says no, returns a flat TickResult and only updates TW-5 state
    6) if yes, calls stub or TW-5 GPT client to produce an OrderPlan
    7) passes plan to TW-5 `risk_clamp`
    8) (for now) does not execute anything; execution_result is always None
    9) writes TW-5 fields back into state and optionally persists via `save_state()`.
  - `TickResult` dataclass summarises snapshot, plan, clamp_result, execution_result, and gatekeeper status.

- `tw5/executor.py`  
  Skeleton for a future TW-5 execution adapter wrapper (currently no-op).

- `tw5/replay.py`
  TW-5 replay harness with realistic limit order simulation:
  - Loads candles from CSV or in-memory sequences
  - Runs structural replay (no PnL): candles → Tw5Snapshot → gatekeeper → GPT/stub → risk_clamp → TickResult[]
  - Runs PnL replay with persistent limit orders:
    - Tracks pending orders across bars (PendingOrder dataclass)
    - Orders expire after configurable lifetime (default: 100 bars)
    - Fills when entry price touched within bar's OHLC range
    - Gap-aware stop/TP execution with slippage (3 bps) and fees (4 bps per side)
    - One position at a time, max hold bars enforcement
  - Returns replay statistics: equity curve, trade list, win rate, avg R, max drawdown.

5. Core Contracts

5.1 Classic Snapshot schema (v1)

`build_snapshot()` must return a dict with at least:

- symbol: str  
- timestamp: float (unix seconds)  
- price: float  

- trend: "up" | "down" | "sideways" | "unknown"  
- range_position: "extreme_low" | "low" | "mid" | "high" | "extreme_high" | "unknown"  
- volatility_mode: "low" | "normal" | "high" | "explosive" | "unknown"  

- flow: dict with funding, open_interest, skew, skew_bias  
- microstructure: dict with at least:
  sweep_up, sweep_down, fvg_up, fvg_down, choch_direction, compression_active, shape_bias, shape_score  

- liquidity_context: dict with liquidity_above, liquidity_below  
- fib_context: dict with macro_zone, micro_zone  
- htf_context: dict with trend_1h, range_pos_1h  

- danger_mode: bool  
- timing_state: "avoid" | "cautious" | "normal" | "aggressive" | "unknown"  
- market_session: "ASIA" | "EUROPE" | "US" | "OFF_HOURS"  

- recent_price_path: dict with ret_1, ret_5, ret_15, impulse_state, lookback_bars  

- risk_context: dict with equity, max_drawdown, open_positions_summary, last_action, last_confidence  
- risk_envelope: dict (see 5.3)  

- since_last_gpt: dict with time_since_last_gpt_sec, price_change_pct_since_last_gpt,
  equity_change_since_last_gpt, trades_since_last_gpt  

- gpt_state_note: str or None  

Implementation can add more fields, but tests and the brain prompt assume these exist.

5.2 GPT Action schema (classic, v1)

GPT must return JSON:

```json
{
  "action": "long" | "short" | "flat",
  "size": 0.0_to_1.0,
  "confidence": 0.0_to_1.0,
  "rationale": "short explanation"
}
Rules:

rationale is the canonical reasoning field (stored as GptDecision.notes).

size is a fraction of max allowed size, not an absolute quantity.

confidence is conviction; risk engine may further scale.

On parse failure, the client must fall back to a flat, zero-confidence decision and log clearly.

5.3 RiskEnvelope schema (classic, v1)

Caps for a single decision:

max_leverage: float

max_notional: float

max_risk_per_trade_pct: float (of equity)

min_stop_distance_pct: float (of price)

max_stop_distance_pct: float

max_daily_loss_pct: float

note: str

Risk engine must never exceed these limits. Stop distances are clamped into
[min_stop_distance_pct, max_stop_distance_pct] before deriving stop_loss/take_profit.

5.4 RiskDecision / Decision object (classic, v1)

Internal object from risk → execution:

approved: bool

side: "long" | "short" | "flat"

position_size: float (units/contracts)

leverage: float

notional: float

stop_loss_price: float | None

take_profit_price: float | None

reason: str (why approved/rejected)

envelope: dict (snapshot of RiskEnvelope)

gpt_action: dict (raw GPT action for logging)

Execution engine must treat approved=False or side="flat" as “do nothing”.

5.5 TW-5 Snapshot schema (v0)

build_tw5_snapshot() returns a Tw5Snapshot dataclass with fields:

Identity / timing:

symbol: str

timeframe: str

timestamp: float

price: float

Trend / range:

trend_1h: "up" | "down" | "sideways" | "unknown"

trend_4h: same

range_low_7d: float

range_high_7d: float

range_position_7d: "low" | "mid" | "high" | "extreme_low" | "extreme_high" | "unknown"

Swing / fib:

swing_low, swing_high: floats

fib_0_382, fib_0_5, fib_0_618, fib_0_786: floats

Volatility / path:

vol_mode: "low" | "normal" | "high" | "explosive" | "unknown"

atr_pct: float (fraction of price)

Microstructure-lite:

last_impulse_direction: "up" | "down" | "chop" | "unknown"

last_impulse_size_pct: float (abs change fraction)

swept_prev_high: bool

swept_prev_low: bool

Position:

position_side: "flat" | "long" | "short"

position_size: float

position_entry_price: float | None

5.6 TW-5 OrderPlan schema (v0)

Top-level OrderPlan object:

mode: "enter" | "manage" | "flat"

side: "long" | "short" | "flat"

legs: list[OrderLeg]

max_total_size_frac: float in [0, 1]

confidence: float in [0, 1]

rationale: str

Each OrderLeg:

id: str

entry_type: "limit" | "market"

entry_price: float > 0

entry_tag: str (e.g. "fib_0.382", "price")

size_frac: float in (0, 1]

stop_loss: float > 0

take_profits: list[TPLevel]

Each TPLevel:

price: float > 0

size_frac: float in (0, 1]

tag: str (e.g. "1R", "2R", "partial")

The GPT client normalises invalid values conservatively. Non-flat plans with no valid legs
must be turned into a flat plan with an error rationale.

5.7 TW-5 RiskClampResult schema (v0)

RiskClampResult:

approved: bool

reason: str

original_plan: OrderPlan

clamped_plan: OrderPlan | None (flat plan when vetoed)

The clamp must never "expand" risk vs original; it can only veto or shrink.

5.8 TW-5 PendingOrder schema (v0)

PendingOrder (for replay only):

plan: OrderPlan (the original plan that generated this order)

clamp_result: RiskClampResult (the risk clamp result for this plan)

created_ts: float (timestamp when order was created)

created_idx: int (bar index when order was created)

expires_idx: int | None (bar index when order expires, None = no expiry)

snapshot: Tw5Snapshot | None (snapshot at creation time for context)

Helpers:

is_expired(current_idx: int) -> bool: checks if order has expired

PendingOrder objects allow limit orders to persist across bars until filled or expired,
enabling realistic backtesting where pullback entries can fill on future bars when price
reaches the limit level.

Phases & Tasks

Phase 1 – Spine + Observability (classic)

Status: ✅ Complete & frozen

run_id/snapshot_id plumbing.

Snapshot validation + state versioning.

Hardened engine loop, timing probes.

Replay vs live parity and logging.

Change only for serious bugs.

Phase 2 – GPT Policy + Risk Brain v1 (classic)

Status: 🚧 In progress

Key tasks:

Schema lock: classic GPT Action v1 enforced in gpt_client.py + brain.txt.

Stabilise RiskEnvelope schema + validation in risk_envelope.py / risk_engine.py.

Maintain unified Decision object engine → risk → execution (no ad-hoc dicts).

Harden GPT JSON parsing + malformed response tests.

Keep replay parity tests green (live/replay same decisions on same data).

Keep snapshot schema in sync with section 5.1.

Keep GPT Safe Mode robust and well-tested.

When Phase 2 is done, classic snapshot → GPT → risk → execution should be stable
enough for broad research sweeps.

Phase 3 – Replay & Analytics (classic)

Status: 🚧 In progress

Multiple replay modes (single run, parameter sweeps, brain variants).

Per-run exports (already in place).

Summaries/plots for equity, drawdown, regime performance.

TEST_INDEX.md as the compact index of important runs.

Phase 4 – Stateful GPT Agent (classic)

Status: Stub

Rolling memory + regime summaries for GPT.

Preserve v1 contracts while adding statefulness.

Phase 5 – Hyperliquid SIM Integration

Status: 🚧 Partial

✅ HL data adapter (testnet/mainnet).

✅ HL testnet execution adapter (Wallet + Exchange).

⏳ Replay integration with HL data.

⏳ Mainnet adapter (disabled until extra safety complete).

Phase 6 – Hyperliquid Testnet / Live

Status: ⏳ Not started

Hard budget caps, daily loss limits, circuit breakers, kill switch for all external orders.

Mainnet requires explicit approval + extra safety rails.

Phase TW-5 – Thin GPT-First Engine

Status: 🚧 In progress

Completed:

Defined Tw5Snapshot + OrderPlan + RiskClampResult + PendingOrder schemas.

Implemented build_tw5_snapshot() with trend/range/fib/vol/impulse logic.

Fixed hindsight bias: swing_low/swing_high now computed from historical bars only.

Implemented deterministic TW-5 stub policy (fib-based trend follower).

Implemented TW-5 gatekeeper (ATR-scaled move + range extremes, rate-limited).

Implemented TW-5 GPT client with dedicated prompt + parser.

Implemented TW-5 risk clamp with equity floor + daily loss + max_stop_pct + down-only sizing.

Implemented run_tw5_tick() engine wrapper for advisor-style ticks with optional state persistence.

Implemented TW-5 replay harness with persistent limit order simulation, gap-aware exits, slippage, and fees.

Planned:

TW-5 executor wrapper to translate clamped plans into classic RiskDecision/exec calls.

Tests for TW-5 snapshot builder, gatekeeper, GPT client parsing, risk clamp, and engine wiring.

UI/UX layer for RunMode (advisor vs agent_confirm vs auto).

Editing Policy

Tier 1 (core/high-risk):

engine.py, build_features.py, gatekeeper.py, gpt_client.py,
risk_envelope.py, risk_engine.py, execution_engine.py,
state_memory.py, replay.py,
tw5/engine_tw5.py, tw5/risk_clamp.py, tw5/gpt_client.py, tw5/snapshot_builder.py

→ Prefer full-file replacements for non-trivial changes; keep structure clean.

Tier 2 (support/tests/utils):

config.py, adapters/, tests/, safety_utils.py, logger_utils.py,
tw5/stub.py, tw5/gatekeeper.py, tw5/executor.py, tw5/replay.py

→ Local patches allowed; still keep functions cohesive.

All changes must keep tests green or include updated tests in the same task.

LLM Usage

Always read README.md, TW_CANON.md, TW_LOG.md first and confirm they’re in sync.

Do not invent new architecture without explicit Architect instructions.

Foreman mode updates Phases/Tasks here when tasks complete or change.

Craftsman mode, after finishing a task:

Propose updates to Phase/Tasks in this file.

Append a one-line event to TW_LOG.md describing the change.

Sync / “Commit and Push” Ritual

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

Ask for confirmation only for risky actions (force-push, destructive ops, remotes).

After significant sync/code changes:

Update TW_LOG.md with a one-liner.

Ensure TW_CANON.md matches any changed contracts.

This document is the single source of truth for how TradeWarrior (classic spine + TW-5) is supposed to work.