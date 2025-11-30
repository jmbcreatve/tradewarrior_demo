# AI Coding Agent Instructions for TradeWarrior Demo

## Project Overview
The TradeWarrior Demo project is a modular Python application designed to simulate trading workflows. It includes components for data routing, risk management, execution, and state management. The architecture emphasizes modularity and extensibility, making it easy to integrate new adapters or extend existing functionality.

### Key Components
- **Adapters**: Located in the `adapters/` directory, these define data and execution interfaces. Examples include `example_data_adapter.py` and `mock_execution_adapter.py`.
- **Engines**: Core logic resides in files like `engine.py`, `execution_engine.py`, `replay_engine.py`, and `risk_engine.py`.
- **State Management**: `state_memory.py` and `state.json` handle application state.
- **Utilities**: Logging and configuration utilities are in `logger_utils.py` and `config.py`.

### Data Flow
1. **Data Input**: Adapters fetch or simulate data.
2. **Processing**: Engines process data, applying business logic.
3. **Execution**: Results are routed to execution adapters.
4. **State Updates**: State is updated and persisted in `state.json`.

## Developer Workflows

### Running the Application
1. Ensure Python 3.8+ is installed.
2. Run the main script (if applicable) or interact with engines directly.

### Testing
- No explicit test framework is detected. Add tests in a `tests/` directory and use `pytest` for consistency.

### Debugging
- Use `logger_utils.py` for logging. Adjust log levels as needed.

## Project-Specific Conventions
- **Adapters**: Follow the base classes `base_data_adapter.py` and `base_execution_adapter.py`.
- **Engines**: Encapsulate logic in engine files. Avoid mixing concerns.
- **State Management**: Use `state_memory.py` for all state interactions.

## Integration Points
- **Adapters**: Extend base classes to integrate new data sources or execution targets.
- **State**: Modify `state.json` cautiously. Use `state_memory.py` to ensure consistency.

## Examples
- Adding a new data adapter:
  ```python
  from adapters.base_data_adapter import BaseDataAdapter

  class MyDataAdapter(BaseDataAdapter):
      def fetch_data(self):
          # Implement data fetching logic
          pass
  ```

- Using the risk engine:
  ```python
  from risk_engine import RiskEngine

  engine = RiskEngine()
  engine.evaluate_risk(data)
  ```

## Notes
- Ensure all new code adheres to the modular structure.
- Document any new adapters or engines thoroughly.
