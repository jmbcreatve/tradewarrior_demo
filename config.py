import json
import os
from dataclasses import dataclass, field
from typing import List, Optional

from enums import ExecutionMode


def _default_backup_data_sources() -> List[str]:
    return ["mock"]


def _str_to_bool(val: str) -> bool:
    """Convert string to boolean."""
    return val.lower() in ("true", "1", "yes", "on")


def _get_env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _get_env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _get_env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return _str_to_bool(raw)


def _get_env_float_list(name: str, default: List[float]) -> List[float]:
    raw = os.getenv(name)
    if raw is None:
        return list(default)

    pieces = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    try:
        floats = [float(p) for p in pieces]
    except ValueError:
        return list(default)

    return floats if floats else list(default)


@dataclass
class Config:
    """Top-level configuration for the TradeWarrior demo system.

    This is the single canonical description of runtime configuration including:
    - Risk caps (risk_per_trade, max_leverage, etc.)
    - Testnet mode settings (is_testnet, execution_mode)
    - Initial equity for position sizing and state initialization
    - Data/execution adapter selection
    - Loop timing and paths

    All secrets (API keys, private keys) must come from OS environment variables,
    not from .env files. This class keeps environment reads minimal: safe TW-5
    execution defaults and the GPT model can be overridden via env vars for
    convenience.

    Defaults use Hyperliquid for real data, mock execution for safety.
    """

    symbol: str = "BTCUSDT"
    timeframe: str = "1m"

    # Risk caps (downward-only limits enforced by risk_engine.py)
    risk_per_trade: float = 0.005  # Risk per trade as fraction of equity (0.5% default)
    max_leverage: float = 3.0  # Maximum leverage multiplier
    max_leverage_10x_mode: float = 10.0  # Maximum leverage in aggressive mode
    # Replay-only risk tuning (ignored in live/testnet). Lets us size replays meaningfully.
    replay_risk_floor_pct: float = 0.02  # 2% floor for replay sizing
    replay_risk_cap_pct: float = 0.05    # 5% cap for replay sizing
    replay_notional_cap_pct: float = 10.0  # Allow larger notionals in replay so risk_pct can express
    replay_mode: bool = False            # Flag set by replay_engine to enable replay-only risk tweaks

    # Account / equity
    initial_equity: float = 10_000.0  # Starting equity for position sizing and state initialization

    # Loop / orchestration
    loop_sleep_seconds: float = 3.0  # How long run_forever sleeps between iterations

    # Adapters - use Hyperliquid for real data, fallback to mock
    primary_data_source_id: str = "hl"
    backup_data_source_ids: List[str] = field(default_factory=_default_backup_data_sources)
    primary_execution_id: str = "mock"
    backup_execution_ids: List[str] = field(default_factory=list)

    # Execution mode and testnet settings
    execution_mode: ExecutionMode = ExecutionMode.SIM  # Execution mode (SIM, HL_TESTNET, etc.)
    paper_trading: bool = True  # Paper trading flag (legacy, may be redundant with execution_mode)
    is_testnet: bool = False  # Master testnet flag - affects API endpoints in adapters

    # GPT / brain config (model + token cap). Model from env var or default.
    gpt_model: str = field(default_factory=lambda: os.getenv("TRADEWARRIOR_GPT_MODEL", "gpt-4o-mini"))
    gpt_max_tokens: int = 256

    # TW-5 prompt profile selector: "conservative" or "aggressive"
    # Controls which TW-5 brain file is loaded (tw5/tw5_brain_<profile>.txt)
    # Default: "conservative" for safer, flatter trading
    tw5_prompt_profile: str = "conservative"

    # TW-5 execution cadence and exit defaults (env-overridable, demo-safe)
    tw5_manage_interval_sec: float = field(
        default_factory=lambda: _get_env_float("TW5_MANAGE_INTERVAL_SEC", 180.0)
    )
    tw5_tp_r_multipliers: List[float] = field(
        default_factory=lambda: _get_env_float_list("TW5_TP_R_MULTIPLIERS", [1.2, 2.0, 3.0])
    )
    tw5_tp_remaining_fracs: List[float] = field(
        default_factory=lambda: _get_env_float_list("TW5_TP_REMAINING_FRACS", [0.30, 0.30, 1.00])
    )
    tw5_trail_early_trigger_r: float = field(
        default_factory=lambda: _get_env_float("TW5_TRAIL_EARLY_TRIGGER_R", 0.7)
    )
    tw5_trail_early_stop_r: float = field(
        default_factory=lambda: _get_env_float("TW5_TRAIL_EARLY_STOP_R", -0.25)
    )
    tw5_trail_after_tp1_stop_r: float = field(
        default_factory=lambda: _get_env_float("TW5_TRAIL_AFTER_TP1_STOP_R", 0.10)
    )
    tw5_trail_after_tp2_stop_r: float = field(
        default_factory=lambda: _get_env_float("TW5_TRAIL_AFTER_TP2_STOP_R", 1.00)
    )
    tw5_trail_runner_giveback_r: float = field(
        default_factory=lambda: _get_env_float("TW5_TRAIL_RUNNER_GIVEBACK_R", 1.0)
    )
    tw5_stop_order_mode: str = field(
        default_factory=lambda: os.getenv("TW5_STOP_ORDER_MODE", "stop_market")
    )
    tw5_stop_limit_offset_bps: int = field(
        default_factory=lambda: _get_env_int("TW5_STOP_LIMIT_OFFSET_BPS", 10)
    )
    tw5_live_force_market_entry: bool = field(
        default_factory=lambda: _get_env_bool("TW5_LIVE_FORCE_MARKET_ENTRY", True)
    )

    # Paths
    state_file: str = "state.json"  # Path to state JSON file
    log_dir: str = "logs"  # Directory for log files


def load_config(path: str | None = None) -> Config:
    """Return a Config instance with demo-safe defaults or load from a file.

    This function is environment-agnostic and MUST NOT read environment variables
    or .env files beyond the env-aware default factories on Config. All defaults are
    safe to run in a local or demo environment with no API keys. Secrets must be
    provided via OS environment variables at runtime.
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


def load_testnet_config(equity: float = 1000.0) -> Config:
    """
    Create a Config specifically configured for Hyperliquid testnet trading.
    
    This function does NOT load .env files. It expects required secrets to be
    available as OS environment variables (e.g., HL_TESTNET_PRIVATE_KEY).
    
    Args:
        equity: Initial equity for position sizing (default $1000 for testnet)
    
    Returns:
        Config instance configured for testnet with conservative position sizing
    
    Raises:
        ValueError: If required environment variables are missing
    """
    # Validate required env vars are present (from OS environment, not .env file)
    required_env_vars = [
        "HL_TESTNET_PRIVATE_KEY",
    ]
    missing = [v for v in required_env_vars if not os.getenv(v)]
    if missing:
        raise ValueError(
            f"Missing required testnet env vars: {missing}. "
            f"Please set them as OS environment variables (e.g., export HL_TESTNET_PRIVATE_KEY=0x...)."
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
