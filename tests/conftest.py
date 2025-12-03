# tests/conftest.py
"""
Test configuration for TradeWarrior.

This makes sure the project root (where config.py, enums.py, etc. live)
is on sys.path so tests can do `from config import Config` without failing
with ModuleNotFoundError, regardless of how pytest is invoked.
"""

import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
