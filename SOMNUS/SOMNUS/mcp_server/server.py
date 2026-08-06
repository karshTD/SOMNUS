"""MCP server over stdio (the transport Claude Desktop actually speaks).

The previous build had a ``run_stdio`` method with no entry point wired to it
and served raw JSON-RPC over HTTP instead, which no MCP client connects to.
``python -m mcp_server`` is now the entry point.

The package is ``mcp_server``, not ``mcp``, so it does not shadow the official
MCP SDK on sys.path.
"""

from __future__ import annotations

import json
import logging
import sys
from typing import Any, Callable

from core.actions import REMEDIATION_SCRIPTS, execute_decision
from infra.embeddings import ObservationEncoder
from mcp_server.introspection import (
    agent_state,
    beliefs_as_of,
    explain_belief,
    schema_info,
    table_stats,
)
from memory.store import MemoryStore

logger = logging.getLogger(__name__)
MCP_PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "recall_schemas",
        "description": "Semantic recall over consolidated cortical schemas, given a system state.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cpu_percent": {"type": "number"},
                "rps": {"type": "number"},
                "network_mbps": {"type": "number"},
                "limit": {"type": "integer", "default": 3},
            },
            "required": ["cpu_percent", "rps", "network_mbps"],
        },
    },
    {
        "name": "explain_belief",
        "description": "Why does the agent believe this? Traverses schema -> provenance -> episodes.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "cpu_percent": {"type": "number"},
                "rps": {"type": "number"},
                "network_mbps": {"type": "number"},
            },
            "required": ["cpu_percent", "rps", "network_mbps"],
        },
    },
    {
        "name": "beliefs_as_of",
        "description": "MVCC time travel: what did the agent believe at an earlier time? e.g. '-1h'.",
        "inputSchema": {
            "type": "object",
            "properties": {"timestamp": {"type": "string", "default": "-1h"}},
        },
    },
    {
        "name": "get_db_schema",
        "description": "Inspect the memory substrate's table structure.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_memory_stats",
        "description": "Live episode count, schema count, provenance links.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_agent_state",
        "description": "Current wake-loop state: surprise, ACh, NA, DA, active context.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_remediation_skill",
        "description": "Execute a remediation procedure.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": list(REMEDIATION_SCRIPTS.keys())},
                "context": {"type": "object"},
            },
            "required": ["action"],
        },
    },
]


class SomnusMCPServer:
    def __init__(
        self,
        store: MemoryStore | None = None,
        agent_state_provider: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self._store = store
        self._encoder: ObservationEncoder | None = None
        self._agent_state_provider = agent_state_provider

    @property
    def store(self) -> MemoryStore:
        if self._store is None:
            from memory.cortex import CockroachStore

            self._store = CockroachStore()
        return self._store

    @property
    def encoder(self) -> ObservationEncoder:
        if self._encoder is None:
            self._encoder = ObservationEncoder()
        return self._encoder

    def _observation(self, args: dict[str, Any]) -> dict[str, Any]:
        return {
            "cpu_percent": float(args.get("cpu_percent", 0)),
            "rps": float(args.get("rps", 0)),
            "network_mbps": float(args.get("network_mbps", 0)),
        }

    def _call_tool(self, name: str, args: dict[str, Any]) -> list[dict[str, Any]]:
        def text(payload: Any) -> list[dict[str, Any]]:
            return [{"type": "text", "text": json.dumps(payload, indent=2, default=str)}]

        if name == "recall_schemas":
            _, vec = self.encoder.encode(self._observation(args))
            matches = self.store.recall_schemas(vec, limit=int(args.get("limit", 3)))
            return text([{**m.schema.to_dict(), "similarity": round(m.similarity, 4)} for m in matches])

        if name == "explain_belief":
            _, vec = self.encoder.encode(self._observation(args))
            return text(explain_belief(self.store, vec))

        if name == "beliefs_as_of":
            return text(beliefs_as_of(self.store, args.get("timestamp", "-1h")))

        if name == "get_db_schema":
            return text(schema_info(self.store))

        if name == "get_memory_stats":
            return text(table_stats(self.store))

        if name == "get_agent_state":
            provider = self._agent_state_provider
            return [{"type": "text", "text": agent_state(provider() if provider else None)}]

        if name == "run_remediation_skill":
            result = execute_decision(
                {"action": args["action"], "context": args.get("context", {})}
            )
            return text(result.to_dict())

        raise ValueError(f"Unknown tool: {name}")

    def handle_request(self, request: dict[str, Any]) -> dict[str, Any]:
        req_id = request.get("id")
        method = request.get("method", "")
        params = request.get("params") or {}

        try:
            if method == "initialize":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "protocolVersion": MCP_PROTOCOL_VERSION,
                        "capabilities": {"tools": {}},
                        "serverInfo": {"name": "somnus-mcp", "version": "2.0.0"},
                    },
                }
            if method.startswith("notifications/"):
                return {}
            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}
            if method == "tools/list":
                return {"jsonrpc": "2.0", "id": req_id, "result": {"tools": TOOLS}}
            if method == "tools/call":
                content = self._call_tool(params.get("name", ""), params.get("arguments") or {})
                return {"jsonrpc": "2.0", "id": req_id, "result": {"content": content, "isError": False}}
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:  # noqa: BLE001
            logger.exception("MCP request failed")
            return {"jsonrpc": "2.0", "id": req_id, "error": {"code": -32000, "message": str(exc)}}


def run_stdio(server: SomnusMCPServer | None = None) -> None:
    """Blocking newline-delimited JSON-RPC loop over stdin/stdout.

    Deliberately synchronous: asyncio.connect_read_pipe on stdin does not work
    on Windows and get_event_loop is deprecated on 3.12+.
    """
    server = server or SomnusMCPServer()
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    logger.info("SOMNUS MCP server ready on stdio")

    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            continue
        response = server.handle_request(request)
        if response:
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()
