"""Agent <-> dashboard command channel.

The original dashboard constructed a fresh ``Simulator()`` on every Streamlit
rerun and called ``trigger_anomaly()`` on that throwaway object -- in a
different container from the agent, which held its own Simulator. The button
silently did nothing, which is the worst possible failure mode on stage: no
error, no effect.

Commands are now written to a shared file the agent polls each wake step.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

DATA_DIR = Path(os.getenv("SOMNUS_DATA_DIR", Path(__file__).parent.parent / "data"))
COMMAND_FILE = DATA_DIR / "commands.json"
_lock = threading.Lock()


def _atomic_write(path: Path, payload: Any) -> None:
    """Write via temp file + rename so a reader never sees a partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, default=str)
        os.replace(tmp, path)
    except Exception:
        Path(tmp).unlink(missing_ok=True)
        raise


def send_command(name: str, **kwargs: Any) -> None:
    with _lock:
        queue = _read_raw()
        queue.append({"command": name, "args": kwargs})
        _atomic_write(COMMAND_FILE, queue)


def _read_raw() -> list[dict[str, Any]]:
    if not COMMAND_FILE.exists():
        return []
    try:
        data = json.loads(COMMAND_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


def drain_commands() -> list[dict[str, Any]]:
    """Read and clear. Called by the agent once per wake step."""
    with _lock:
        queue = _read_raw()
        if queue:
            _atomic_write(COMMAND_FILE, [])
        return queue
