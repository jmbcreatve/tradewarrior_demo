import json
from dataclasses import dataclass, field
from typing import List

from enums import ExecutionMode


@dataclass
class Config:
    """Top-level configuration for the TradeWarrior demo system.

    Defaults are DEMO-safe: mock data, mock execution, paper trading enabled.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = "1m"

    # Risk
    risk_per_trade: float = 0.005
    max_leverage: float = 3.0
    max_leverage_10x_mode: float = 10.0

    # Loop / orchestration
    # How long run_forever sleeps between iterations.
    loop_sleep_seconds: float = 3.0

    # Adapters
    primary_data_source_id: str = "mock"
    backup_data_source_ids: List[str] = field(default_factory=list)
    primary_execution_id: str = "mock"
    backup_execution_ids: List[str] = field(default_factory=list)

    # Modes
    execution_mode: ExecutionMode = ExecutionMode.SIM
    paper_trading: bool = True

    # GPT / brain config (model + token cap). Model can be overridden via env var.
    gpt_model: str = "gpt-5-mini"
    gpt_max_tokens: int = 256

    # Paths
    state_file: str = "state.json"
    log_dir: str = "logs"


def load_config(path: str | None = None) -> Config:
    """Return a Config instance with demo-safe defaults or load from a file.

    This function MUST NOT require environment variables. All defaults are safe
    to run in a local or demo environment with no API keys.
    """
    if path is None:
        return Config()

    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError("Config file must contain a JSON object at the top level.")
    return Config(**payload)
