# TradeWarrior – Agent Rules (READ THIS FIRST)

This README defines how **humans and LLMs** must work with this repo.  
Before any change, tools must read:

1. `README.md`  (these rules)
2. `TW_CANON.md` (architecture + invariants + schemas)
3. `TW_LOG.md`  (chronological change log)

---

## 1. Sources of Truth

- **Architecture & invariants:** `TW_CANON.md`
- **History of changes:** `TW_LOG.md`
- **Implementation:** code on branch `main` in this repo

Rules:

- Do **not** contradict CANON.
- Do **not** rewrite history in LOG.
- If code behavior changes in a way that affects CANON, update CANON and add a LOG entry.

---

## 2. Editing Rules for Any Agent (Codex, ChatGPT, etc.)

When asked to change code:

1. Read `README.md`, `TW_CANON.md`, `TW_LOG.md`.
2. Confirm the requested change does **not** violate any invariant in CANON.
3. Prefer the **smallest correct change** that preserves:
   - Spine: `config → data adapters → snapshot → gatekeeper → GPT → risk → execution → state`
   - Downward-only risk: risk/leverage/notional can only be reduced vs config/envelope.
   - Single state writer: only `state_memory.py` writes persistent JSON state.
   - Replay/live parity: replay uses the same `run_spine_tick` flow as live.

Never:

- Introduce new architecture without CANON updates.
- Edit env/secret handling in core modules (no `.env` or `local_env.txt` loading).
- Touch Hyperliquid mainnet endpoints.
- Remove or weaken safety features (daily loss limit, kill switch, GPT safe mode).

---

## 3. Environment & Secrets

- All secrets come from **OS environment variables** (e.g. `OPENAI_API_KEY`, `HL_TESTNET_PRIVATE_KEY`).
- Core modules must **not** load `.env` or `local_env.txt`.
- Optional env loading is allowed only in top-level runner scripts and must be clearly marked as optional.

If you see code that violates this, treat it as a bug.

---

## 4. Hyperliquid Adapters

- Execution is **testnet-only**.
- Uses the current Hyperliquid SDK + `eth_account.Account.from_key`.
- Credentials: `HL_TESTNET_PRIVATE_KEY` from OS env.
- On any error, adapters must fail safe (log and skip trade), not crash the engine.

Do not add mainnet execution without an explicit new phase in CANON.

---

## 5. TW_LOG and TW_CANON Editing Discipline

### TW_LOG.md

- Append-only, one line per event:
  - `YYYY-MM-DD – short description of what changed and why`
- Only update when code or architecture actually changes.
- No paragraphs, no design notes, no future tasks.

### TW_CANON.md

- Update only when:
  - Architecture, invariants, schemas, or module responsibilities change, **and**
  - Code on `main` has already been updated (or is being updated in the same change).
- Keep it factual, minimal, and consistent with the current code.

---

## 6. Workflow Expectations (for humans)

Typical safe loop:

1. `git pull --ff-only`
2. Make small, CANON-consistent edits (manually or via Codex).
3. Run tests (e.g. `pytest`).
4. Update `TW_LOG.md` (and `TW_CANON.md` if needed).
5. `git add -A && git commit -m "short message" && git push origin main`

Any agent asked to “edit code” should assume this workflow and not run git commands itself.

---

## 7. Replay Outputs & Indexing

- Each replay writes to `analytics/runs/<run_id>/` with:
  - `trades.csv`, `equity.csv`, `parity.jsonl`
  - `features_at_entry.csv`, `features_at_exit.csv`
  - `run_config.json` (full config + metadata) and `summary.json` (stats + side breakdown)
- `TEST_INDEX.md` is a compact Markdown table linking run_id → tape/config/stats; one row per run.
- Future analysis/scripts must read per-run folders (not a single global CSV).
- When adding a new replay, record the run_id/path in `TEST_INDEX.md` and add a brief note only; details live in the run folder.
`
