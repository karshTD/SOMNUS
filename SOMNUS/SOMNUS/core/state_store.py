"""Agent -> dashboard state publishing. Atomic writes; readers never see partials."""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

from core.control import DATA_DIR, _atomic_write

STATE_FILE = DATA_DIR / "somnus_state.json"
_lock = threading.Lock()


def write_state(state: dict[str, Any]) -> None:
    with _lock:
        _atomic_write(STATE_FILE, state)


def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return {}
    try:
        return json.loads(STATE_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}
