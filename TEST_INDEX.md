# Test & Replay Index

One row per replay/test run. Detailed artifacts live in `analytics/runs/<run_id>/`.

| date       | run_id                               | tape path                               | config/tag                | trades | win_rate | final_equity | max_dd | note |
|------------|--------------------------------------|-----------------------------------------|---------------------------|--------|----------|--------------|--------|------|
| 2025-12-05 | legacy_baseline                      | dumps/hl_mainnet_btcusdt_1m_60d.csv     | Baseline_Mainnet_A (stub) | 238    | 40.8%    | 7627.89      | 25.94% | Stub lost materially; large DD; needs policy changes. |
| 2025-12-05 | 61c88c5310eb44bab3b0e149f2ffa22d     | analytics/parity_hl_mainnet_btcusdt_1m_60d_antichop.jsonl | AntiChop_v1 (stub)        | 60     | 51.7%    | 9447.09      | 7.10%  | Chop/high gate reduced losses vs baseline; still all-long. |
| 2025-12-06 | ReplayStub_v2                        | dumps/hl_mainnet_btcusdt_1m_60d.csv     | ReplayStub_v2 (stub)      | 30     | 50.0%    | 9239.64      | 8.77%  | All-short run (30 shorts, no longs); v2 stub flipped bias on this tape. |

## AntiChop_v1 (replay-only chop/high long block)
- Tape/Data: dumps/hl_mainnet_btcusdt_1m_60d.csv
- Config/Preset: GPT stub + replay gate blocking longs in low-vol chop at highs
- Trades: 60
- Win Rate: 51.7%
- Final Equity: 9447.09 (total PnL -552.91)
- Max Drawdown: 7.10%
- Side breakdown: longs=60, avg pnl -9.22, total pnl -552.91, hit rate 51.7%; shorts=0
- Notes: Cuts loss vs Baseline_Mainnet_A (-2.37k → -0.55k) and shrinks max DD (25.9% → 7.1%) with fewer trades (238 → 60); still all-long behavior but the chop-high gate stops most bad entries.
