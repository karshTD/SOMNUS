"""MCP stdio server bridging CockroachDB cortex queries and agent skills."""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from typing import Any

from core.actions import REMEDIATION_SCRIPTS, execute_decision
from core.simulator import Simulator
from infra.aws_client import AWSClient
from mcp.introspection import get_agent_state, get_schema_info, get_table_stats
from memory.cortex import Cortex

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"

TOOLS = [
    {
        "name": "recall_similar_rules",
        "description": "Query CockroachDB cortex for semantically similar generalized rules.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Natural language query"},
                "limit": {"type": "integer", "default": 5},
            },
            "required": ["query"],
        },
    },
    {
        "name": "get_db_schema",
        "description": "Read CockroachDB semantic_memory schema for introspection.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_table_stats",
        "description": "Check semantic memory table size to assess meta-plasticity needs.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "get_agent_state",
        "description": "Return current SOMNUS agent wake-loop state.",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "run_remediation_skill",
        "description": "Execute a pre-built remediation script (agent skill).",
        "inputSchema": {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": list(REMEDIATION_SCRIPTS.keys()),
                },
                "context": {"type": "object"},
            },
            "required": ["action"],
        },
    },
]


class SomnusMCPServer:
    """Minimal MCP-compatible JSON-RPC server over stdio."""

    def __init__(self, agent_state_provider: Any = None) -> None:
        self.cortex = Cortex()
        self.aws = AWSClient()
        self.simulator = Simulator()
        self._agent_state_provider = agent_state_provider

    def _agent_state(self) -> dict[str, Any]:
        if callable(self._agent_state_provider):
            return self._agent_state_provider()
        return {}

    def _call_tool(self, name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
        if name == "recall_similar_rules":
            vector = self.aws.embed_text(arguments["query"])
            rules = self.cortex.recall_similar(vector, limit=arguments.get("limit", 5))
            text = json.dumps([r.to_dict() for r in rules], indent=2)
            return [{"type": "text", "text": text}]

        if name == "get_db_schema":
            info = get_schema_info(self.cortex)
            return [{"type": "text", "text": json.dumps(info, indent=2)}]

        if name == "get_table_stats":
            stats = get_table_stats(self.cortex)
            return [{"type": "text", "text": json.dumps(stats, indent=2)}]

        if name == "get_agent_state":
            return [{"type": "text", "text": get_agent_state(self._agent_state())}]

        if name == "run_remediation_skill":
            result = execute_decision(
                {"action": arguments["action"], "context": arguments.get("context", {})},
                simulator=self.simulator,
            )
            return [{"type": "text", "text": json.dumps(result.to_dict(), indent=2)}]

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
                        "serverInfo": {"name": "somnus-mcp", "version": "1.0.0"},
                    },
                }

            if method == "notifications/initialized":
                return {}

            if method == "tools/list":
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"tools": TOOLS},
                }

            if method == "tools/call":
                name = params.get("name", "")
                arguments = params.get("arguments") or {}
                content = self._call_tool(name, arguments)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {"content": content, "isError": False},
                }

            if method == "ping":
                return {"jsonrpc": "2.0", "id": req_id, "result": {}}

            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32601, "message": f"Method not found: {method}"},
            }
        except Exception as exc:
            logger.exception("MCP request failed")
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {"code": -32000, "message": str(exc)},
            }

    async def run_stdio(self) -> None:
        """Read JSON-RPC from stdin and write responses to stdout."""
        loop = asyncio.get_event_loop()
        reader = asyncio.StreamReader()
        protocol = asyncio.StreamReaderProtocol(reader)
        await loop.connect_read_pipe(lambda: protocol, sys.stdin)

        writer_transport, writer_protocol = await loop.connect_write_pipe(
            asyncio.streams.FlowControlMixin, sys.stdout
        )
        writer = asyncio.StreamWriter(writer_transport, writer_protocol, reader, loop)

        while True:
            line = await reader.readline()
            if not line:
                break
            line_str = line.decode("utf-8").strip()
            if not line_str:
                continue
            request = json.loads(line_str)
            response = self.handle_request(request)
            if response:
                writer.write((json.dumps(response) + "\n").encode("utf-8"))
                await writer.drain()


def run_server(agent_state_provider: Any = None) -> None:
    """Entry point for MCP server process."""
    logging.basicConfig(level=logging.INFO, stream=sys.stderr)
    server = SomnusMCPServer(agent_state_provider=agent_state_provider)
    asyncio.run(server.run_stdio())
