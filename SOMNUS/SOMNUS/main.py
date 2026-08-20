"""SOMNUS orchestrator - optimized."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from dataclasses import dataclass
from typing import Any, Callable, Optional

from dotenv import load_dotenv

load_dotenv()

from core.agent import SomnusAgent, WakeState
from core.simulator import Simulator
from core.state_store import write_state
from infra.config import CONFIG, OFFLINE

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("somnus.main")

_agent: SomnusAgent | None = None


@dataclass(frozen=True)
class StoreFactory:
    """Factory for creating store instances."""
    offline: bool
    
    def create(self) -> Any:
        if self.offline or not CONFIG.db_url:
            from memory.inmemory import InMemoryStore
            logger.warning("Using in-memory store (offline mode)")
            return InMemoryStore()
        from memory.cortex import CockroachStore
        return CockroachStore()


@dataclass(frozen=True)
class AWSFactory:
    """Factory for AWS client instances."""
    enabled: bool
    
    def create(self) -> Any:
        if not self.enabled or not CONFIG.s3_bucket:
            return None
        from infra.aws_client import AWSClient
        return AWSClient()


def _on_state_change(state: WakeState) -> None:
    """Handle state changes."""
    write_state({"status": "running", "wake": state.to_dict()})


def _shutdown(signum: int, frame: Any) -> None:
    """Graceful shutdown handler."""
    logger.info("Signal %s - shutting down", signum)
    if _agent:
        _agent.stop()
    write_state({"status": "stopped"})
    sys.exit(0)


def _setup_signal_handlers() -> None:
    """Register signal handlers."""
    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)


def _run_health_check() -> None:
    """Run health checks and exit."""
    from infra.aws_client import AWSClient
    from memory.cortex import CockroachStore
    
    results = {}
    
    for name, client in [("cockroachdb", CockroachStore), ("aws", AWSClient)]:
        try:
            results[name] = client().health_check()
        except Exception as e:
            results[name] = f"error: {e}"
    
    print(json.dumps(results, indent=2))


def _run_sleep_once(offline: bool, store: Any) -> None:
    """Run a single consolidation pass."""
    if offline:
        from sleep_cycle.consolidation import consolidate
        result = consolidate(store)
    else:
        from sleep_cycle.lambda_handler import run_sleep_cycle
        result = run_sleep_cycle(store=store)
    
    print(json.dumps(result.to_dict(), indent=2))


def _start_mcp_server(store: Any, agent: SomnusAgent) -> None:
    """Start MCP server in background thread."""
    from mcp_server.server import SomnusMCPServer, run_stdio
    
    server = SomnusMCPServer(store=store, agent_state_provider=agent.get_state)
    threading.Thread(
        target=run_stdio,
        args=(server,),
        daemon=True,
        name="mcp"
    ).start()


def _parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="SOMNUS agent")
    parser.add_argument("--interval", type=float, default=CONFIG.poll_interval)
    parser.add_argument("--offline", action="store_true", help="Force in-memory store + local encoder")
    parser.add_argument("--sleep-once", action="store_true", help="Run one consolidation pass and exit")
    parser.add_argument("--migrate", action="store_true", help="Apply CockroachDB schema and exit")
    parser.add_argument("--health", action="store_true", help="Check connectivity and exit")
    parser.add_argument("--mcp", action="store_true", help="Run MCP stdio server alongside loop")
    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    global _agent
    
    args = _parse_args()
    offline = args.offline or OFFLINE
    
    # Early exit commands
    if args.migrate:
        from substrate.migrate import migrate
        migrate()
        return
    
    if args.health:
        _run_health_check()
        return
    
    # Initialize stores
    store = StoreFactory(offline).create()
    
    # Single pass execution
    if args.sleep_once:
        _run_sleep_once(offline, store)
        return
    
    # Full agent execution
    aws = AWSFactory(not offline).create()
    _agent = SomnusAgent(
        store=store,
        simulator=Simulator(),
        aws=aws,
        poll_interval=args.interval,
        on_state_change=_on_state_change,
    )
    
    _setup_signal_handlers()
    
    if args.mcp:
        _start_mcp_server(store, _agent)
    
    write_state({"status": "starting", "wake": _agent.get_state()})
    logger.info("SOMNUS starting (offline=%s)", offline)
    _agent.wake_loop()


if __name__ == "__main__":
    main()
