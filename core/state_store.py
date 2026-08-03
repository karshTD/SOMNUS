"""Shared runtime state for agent ↔ dashboard IPC."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

STATE_FILE = Path(__file__).parent.parent / "data" / "somnus_state.json"
_lock = threading.Lock()


def write_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        STATE_FILE.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    with _lock:
        try:
            return json.loads(STATE_FILE.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return {}
