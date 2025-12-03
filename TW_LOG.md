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
- 2025-12-02 – Entered Phase 2 (GPT Policy + Risk Brain v1) and defined v1 contracts.  
- 2025-12-02 – Fixed test_gatekeeper assertions to match dict return type.  
- 2025-12-03 – Updated requirements, improved logging, and added testnet configuration support (load_testnet_config, run_testnet.py, state_testnet.json).  
- 2025-12-03 – Enhanced config.py with testnet mode, env file loading, initial_equity support, and improved state_memory.py schema handling.
