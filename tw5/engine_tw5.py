"""TW-5 engine wiring.

High-level loop (one tick):
- Get data
- Build Tw5Snapshot
- Gatekeeper: should we call GPT?
- GPT or stub -> OrderPlan
- Risk clamp -> RiskClampResult
- Executor -> send orders / no-op
- Update state
"""

# Empty skeleton; implementation to be added.

