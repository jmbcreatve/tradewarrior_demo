from typing import Dict, Any, List, Optional

from config import Config
from adapters.base_data_adapter import BaseDataAdapter
from adapters.base_execution_adapter import BaseExecutionAdapter
from adapters.mock_data_adapter import MockDataAdapter
from adapters.mock_execution_adapter import MockExecutionAdapter
from adapters.example_data_adapter import ExampleDataAdapter
from adapters.example_execution_adapter import ExampleExecutionAdapter
from logger_utils import get_logger

logger = get_logger(__name__)


def build_data_adapters(config: Config) -> Dict[str, BaseDataAdapter]:
    adapters: Dict[str, BaseDataAdapter] = {
        "mock": MockDataAdapter(),
        "example": ExampleDataAdapter(),
    }
    return adapters


def build_execution_adapters(config: Config) -> Dict[str, BaseExecutionAdapter]:
    adapters: Dict[str, BaseExecutionAdapter] = {
        "mock": MockExecutionAdapter(),
        "example": ExampleExecutionAdapter(),
    }
    return adapters


def get_market_data(
    config: Config,
    data_adapters: Dict[str, BaseDataAdapter],
    limit: int = 100,
) -> Dict[str, Any]:
    """Route market data request to primary/backup adapters.

    If primary health_check fails or raises, fall back to backups, then mock.
    DEMO mode: always safe; never crashes on adapter failure.
    """
    order: List[str] = [config.primary_data_source_id] + list(config.backup_data_source_ids)
    used_adapter: Optional[BaseDataAdapter] = None

    for adapter_id in order:
        adapter = data_adapters.get(adapter_id)
        if adapter is None:
            continue
        try:
            if not adapter.health_check():
                logger.warning("Data adapter %s failed health_check", adapter_id)
                continue
            candles = adapter.fetch_recent_candles(config.symbol, config.timeframe, limit)
            funding = adapter.fetch_funding(config.symbol)
            oi = adapter.fetch_open_interest(config.symbol)
            skew = adapter.fetch_skew(config.symbol)
            used_adapter = adapter
            break
        except Exception as exc:  # noqa: BLE001
            logger.warning("Data adapter %s failed: %s", adapter_id, exc)
            continue

    if used_adapter is None:
        logger.warning("Falling back to mock data adapter.")
        mock = data_adapters.get("mock") or MockDataAdapter()
        candles = mock.fetch_recent_candles(config.symbol, config.timeframe, limit)
        funding = mock.fetch_funding(config.symbol)
        oi = mock.fetch_open_interest(config.symbol)
        skew = mock.fetch_skew(config.symbol)

    return {
        "candles": candles,
        "funding": funding,
        "open_interest": oi,
        "skew": skew,
    }
