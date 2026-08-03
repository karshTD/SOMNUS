"""Central orchestrator for the SOMNUS AI agent."""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import threading
from typing import Any

from dotenv import load_dotenv

from core.agent import SomnusAgent, WakeState
from core.state_store import write_state
from mcp.server import SomnusMCPServer
from sleep_cycle.lambda_handler import consolidate_episodes

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("somnus.main")

_agent: SomnusAgent | None = None
_mcp_thread: threading.Thread | None = None


def _on_state_change(state: WakeState) -> None:
    write_state(
        {
            "wake": state.to_dict(),
            "status": "running",
        }
    )


def _start_mcp_server(agent: SomnusAgent) -> threading.Thread:
    """Start MCP server in a background thread (stdio is unused in thread mode)."""

    def _serve() -> None:
        server = SomnusMCPServer(agent_state_provider=agent.get_state)
        logger.info("MCP server initialized (tools available via SomnusMCPServer)")

        import socket
        from http.server import BaseHTTPRequestHandler, HTTPServer

        class MCPHandler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", 0))
                body = self.rfile.read(length)
                import json

                request = json.loads(body.decode("utf-8"))
                response = server.handle_request(request)
                payload = json.dumps(response).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, format: str, *args: Any) -> None:
                logger.debug(format, *args)

        port = 8765
        httpd = HTTPServer(("0.0.0.0", port), MCPHandler)
        logger.info("MCP HTTP bridge listening on port %s", port)
        httpd.serve_forever()

    thread = threading.Thread(target=_serve, daemon=True, name="mcp-server")
    thread.start()
    return thread


def _shutdown(signum: int, frame: Any) -> None:
    logger.info("Received signal %s — shutting down", signum)
    if _agent:
        _agent.stop()
    write_state({"status": "stopped"})
    sys.exit(0)


def main() -> None:
    global _agent, _mcp_thread

    parser = argparse.ArgumentParser(description="SOMNUS AI Agent")
    parser.add_argument("--interval", type=float, default=2.0, help="Wake loop interval (seconds)")
    parser.add_argument("--no-mcp", action="store_true", help="Disable MCP server thread")
    parser.add_argument("--sleep-once", action="store_true", help="Run one REM consolidation cycle and exit")
    args = parser.parse_args()

    if args.sleep_once:
        result = consolidate_episodes()
        logger.info("Sleep cycle result: %s", result)
        return

    _agent = SomnusAgent(poll_interval=args.interval, on_state_change=_on_state_change)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    if not args.no_mcp:
        _mcp_thread = _start_mcp_server(_agent)

    write_state({"status": "starting", "wake": _agent.get_state()})
    logger.info("SOMNUS agent starting")
    _agent.wake_loop()


if __name__ == "__main__":
    main()
