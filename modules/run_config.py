"""Persistent argparse defaults for IDE / run-button launches.

Command-line flags always win: values from `run_config.json` are only used as
argparse defaults, so `python autonomous_drone_main.py --sgc-host 1.2.3.4`
still overrides the file.

Usage:
  Create `run_config.json` next to `autonomous_drone_main.py` with any subset
  of the CLI options (dashed or underscored keys both work):

    {
      "mode": "flight",
      "drone_link": "auto",
      "sgc_host": "192.168.1.247"
    }

  A missing or malformed file is not an error - the built-in argparse
  defaults are used instead. Set the `AAD_RUN_CONFIG` environment variable to
  point at a different file, and use `save_default()` to write one option
  back without losing the others.
"""
import json
import os
from functools import lru_cache

CONFIG_ENV_VAR = "AAD_RUN_CONFIG"
DEFAULT_CONFIG_NAME = "run_config.json"
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def config_path() -> str:
    override = os.environ.get(CONFIG_ENV_VAR)
    if override:
        return override
    return os.path.join(PROJECT_ROOT, DEFAULT_CONFIG_NAME)


def _normalize(raw: dict) -> dict:
    return {
        str(key).strip().lstrip("-").replace("-", "_"): value
        for key, value in raw.items()
    }


@lru_cache(maxsize=1)
def _load() -> tuple:
    path = config_path()
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError):
        return path, {}
    if not isinstance(data, dict):
        return path, {}
    return path, _normalize(data)


def get_run_defaults() -> dict:
    """Return the persisted defaults keyed by argparse destination name."""
    return dict(_load()[1])


def config_source() -> str:
    """Return the config file that was read (may not exist)."""
    return _load()[0]


def save_default(key: str, value) -> bool:
    """Persist one option, keeping the other keys intact.

    Returns True when the file was written.
    """
    path = config_path()
    data = {}
    try:
        with open(path, "r", encoding="utf-8") as handle:
            loaded = json.load(handle)
        if isinstance(loaded, dict):
            data = loaded
    except (OSError, ValueError):
        data = {}
    data[str(key).replace("-", "_")] = value
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(data, handle, indent=2)
            handle.write("\n")
    except OSError:
        return False
    _load.cache_clear()
    return True
