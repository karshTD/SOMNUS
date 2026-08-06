"""SOMNUS orchestrator."""

from __future__ import annotations

import argparse
import json
import logging
import signal
import sys
import threading
from typing import Any

from dotenv import load_dotenv

load_dotenv()

from core.agent import SomnusAgent, WakeState  # noqa: E402
from core.simulator import Simulator  # noqa: E402
from core.state_store import write_state  # noqa: E402
from infra.config import CONFIG, OFFLINE  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("somnus.main")

_agent: SomnusAgent | None = None


def _build_store(offline: bool) -> Any:
    if offline or not CONFIG.db_url:
        from memory.inmemory import InMemoryStore

        logger.warning("Running with the in-memory store (no COCKROACH_DB_URL / SOMNUS_OFFLINE=1)")
        return InMemoryStore()
    from memory.cortex import CockroachStore

    return CockroachStore()


def _on_state_change(state: WakeState) -> None:
    write_state({"status": "running", "wake": state.to_dict()})


def _shutdown(signum: int, frame: Any) -> None:
    logger.info("Signal %s -- shutting down", signum)
    if _agent:
        _agent.stop()
    write_state({"status": "stopped"})
    sys.exit(0)


def main() -> None:
    global _agent

    parser = argparse.ArgumentParser(description="SOMNUS agent")
    parser.add_argument("--interval", type=float, default=CONFIG.poll_interval)
    parser.add_argument("--offline", action="store_true", help="Force in-memory store + local encoder")
    parser.add_argument("--sleep-once", action="store_true", help="Run one consolidation pass and exit")
    parser.add_argument("--migrate", action="store_true", help="Apply the CockroachDB schema and exit")
    parser.add_argument("--health", action="store_true", help="Check CRDB + AWS connectivity and exit")
    parser.add_argument("--mcp", action="store_true", help="Run the MCP stdio server alongside the loop")
    args = parser.parse_args()

    offline = args.offline or OFFLINE

    if args.migrate:
        from substrate.migrate import migrate

        migrate()
        return

    if args.health:
        from infra.aws_client import AWSClient
        from memory.cortex import CockroachStore

        report: dict[str, Any] = {}
        try:
            report["cockroachdb"] = CockroachStore().health_check()
        except Exception as exc:  # noqa: BLE001
            report["cockroachdb"] = f"error: {exc}"
        try:
            report["aws"] = AWSClient().health_check()
        except Exception as exc:  # noqa: BLE001
            report["aws"] = f"error: {exc}"
        print(json.dumps(report, indent=2))
        return

    store = _build_store(offline)

    if args.sleep_once:
        if offline:
            from sleep_cycle.consolidation import consolidate

            print(json.dumps(consolidate(store).to_dict(), indent=2))
        else:
            from sleep_cycle.lambda_handler import run_sleep_cycle

            print(json.dumps(run_sleep_cycle(store=store), indent=2))
        return

    aws = None
    if not offline and CONFIG.s3_bucket:
        from infra.aws_client import AWSClient

        aws = AWSClient()

    _agent = SomnusAgent(
        store=store,
        simulator=Simulator(),
        aws=aws,
        poll_interval=args.interval,
        on_state_change=_on_state_change,
    )

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if args.mcp:
        from mcp_server.server import SomnusMCPServer, run_stdio

        server = SomnusMCPServer(store=store, agent_state_provider=_agent.get_state)
        threading.Thread(target=run_stdio, args=(server,), daemon=True, name="mcp").start()

    write_state({"status": "starting", "wake": _agent.get_state()})
    logger.info("SOMNUS starting (offline=%s)", offline)
    _agent.wake_loop()


if __name__ == "__main__":
    main()
