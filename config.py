import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from enums import ExecutionMode


def _default_backup_data_sources() -> List[str]:
    return ["mock"]


def _load_env_file(path: str) -> None:
    """Load environment variables from a file (simple .env parser)."""
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, _, value = line.partition("=")
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:  # Don't override existing env vars
                    os.environ[key] = value


def _str_to_bool(val: str) -> bool:
    """Convert string to boolean."""
    return val.lower() in ("true", "1", "yes", "on")


@dataclass
class Config:
    """Top-level configuration for the TradeWarrior demo system.

    Defaults use Hyperliquid for real data, mock execution for safety.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = "1m"

    # Risk
    risk_per_trade: float = 0.005
    max_leverage: float = 3.0
    max_leverage_10x_mode: float = 10.0

    # Account / equity
    initial_equity: float = 10_000.0  # Starting equity for position sizing

    # Loop / orchestration
    # How long run_forever sleeps between iterations.
    loop_sleep_seconds: float = 3.0

    # Adapters - use Hyperliquid for real data, fallback to mock
    primary_data_source_id: str = "hl"
    backup_data_source_ids: List[str] = field(default_factory=_default_backup_data_sources)
    primary_execution_id: str = "mock"
    backup_execution_ids: List[str] = field(default_factory=list)

    # Modes
    execution_mode: ExecutionMode = ExecutionMode.SIM
    paper_trading: bool = True

    # Testnet configuration
    is_testnet: bool = False  # Master testnet flag - affects API endpoints

    # GPT / brain config (model + token cap). Model from env var or default.
    gpt_model: str = field(default_factory=lambda: os.getenv("TRADEWARRIOR_GPT_MODEL", "gpt-4o-mini"))
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
    
    # Handle execution_mode string -> enum conversion
    if "execution_mode" in payload and isinstance(payload["execution_mode"], str):
        payload["execution_mode"] = ExecutionMode(payload["execution_mode"])
    
    return Config(**payload)


def load_testnet_config(
    env_file: str = "twp3_testnet.env",
    equity: float = 1000.0,
) -> Config:
    """
    Create a Config specifically configured for Hyperliquid testnet trading.
    
    Args:
        env_file: Path to env file with HL_TESTNET_* credentials
        equity: Initial equity for position sizing (default $1000 for testnet)
    
    Returns:
        Config instance configured for testnet with conservative position sizing
    """
    # Load environment variables from testnet env file
    _load_env_file(env_file)
    
    # Validate required env vars are present
    required_env_vars = [
        "HL_TESTNET_MAIN_WALLET_ADDRESS",
        "HL_TESTNET_API_WALLET_PRIVATE_KEY",
    ]
    missing = [v for v in required_env_vars if not os.getenv(v)]
    if missing:
        raise ValueError(
            f"Missing required testnet env vars: {missing}. "
            f"Please set them in {env_file} or environment."
        )
    
    return Config(
        symbol="BTCUSDT",
        timeframe="1m",
        
        # Conservative risk settings for testnet
        risk_per_trade=0.01,  # 1% risk per trade
        max_leverage=2.0,     # Conservative 2x leverage for testing
        max_leverage_10x_mode=5.0,  # Cap at 5x even in aggressive mode
        
        # Position sizing based on testnet equity
        initial_equity=equity,
        
        # Loop timing
        loop_sleep_seconds=5.0,  # Slightly slower for testnet monitoring
        
        # Adapters - use Hyperliquid testnet for both data and execution
        primary_data_source_id="hl",
        backup_data_source_ids=["mock"],
        primary_execution_id="hl",  # Use Hyperliquid for execution
        backup_execution_ids=["mock"],
        
        # Execution mode - testnet
        execution_mode=ExecutionMode.HL_TESTNET,
        paper_trading=False,  # Real testnet orders (with play money)
        
        # Master testnet flag
        is_testnet=True,
        
        # GPT config
        gpt_model=os.getenv("TRADEWARRIOR_GPT_MODEL", "gpt-4o-mini"),
        gpt_max_tokens=256,
        
        # Paths - separate state file for testnet
        state_file="state_testnet.json",
        log_dir="logs",
    )
