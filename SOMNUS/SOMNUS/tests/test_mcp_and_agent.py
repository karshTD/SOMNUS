"""MCP protocol handshake and the agent wake loop."""

import json

from core.agent import SomnusAgent
from core.control import drain_commands, send_command
from core.simulator import REGIMES, Simulator
from mcp_server.server import SomnusMCPServer
from memory.inmemory import InMemoryStore


def _server() -> SomnusMCPServer:
    return SomnusMCPServer(store=InMemoryStore(), agent_state_provider=lambda: {"cycle": 3})


def test_initialize_returns_protocol_version() -> None:
    r = _server().handle_request({"jsonrpc": "2.0", "id": 1, "method": "initialize"})
    assert r["result"]["protocolVersion"] == "2024-11-05"


def test_tools_list_is_non_empty_and_well_formed() -> None:
    r = _server().handle_request({"jsonrpc": "2.0", "id": 2, "method": "tools/list"})
    tools = r["result"]["tools"]
    assert tools and all({"name", "description", "inputSchema"} <= set(t) for t in tools)


def test_unknown_method_returns_jsonrpc_error() -> None:
    r = _server().handle_request({"jsonrpc": "2.0", "id": 3, "method": "nope"})
    assert r["error"]["code"] == -32601


def test_agent_state_tool_round_trips() -> None:
    r = _server().handle_request(
        {"jsonrpc": "2.0", "id": 4, "method": "tools/call",
         "params": {"name": "get_agent_state", "arguments": {}}}
    )
    assert json.loads(r["result"]["content"][0]["text"])["cycle"] == 3


def test_command_channel_delivers_and_clears() -> None:
    drain_commands()
    send_command("inject_anomaly", ticks=4)
    assert drain_commands()[0]["command"] == "inject_anomaly"
    assert drain_commands() == []


def test_agent_reacts_to_a_regime_command(monkeypatch) -> None:
    monkeypatch.delenv("COCKROACH_DB_URL", raising=False)
    monkeypatch.setenv("SOMNUS_OFFLINE", "1")
    drain_commands()
    store = InMemoryStore()
    agent = SomnusAgent(store=store, simulator=Simulator(seed=5))
    for _ in range(160):
        agent.wake_step()
    assert store.episode_count() > 0, "surprise-gated encoding must write something"

    send_command("set_regime", regime="surge")
    boundaries = sum(agent.wake_step().boundary for _ in range(30))
    assert boundaries > 0, "the agent must notice the world changed"


def test_simulator_drift_interpolates_between_regimes() -> None:
    sim = Simulator(seed=1)
    sim.set_drift("steady", "surge", 0.5)
    midpoint = (REGIMES["steady"].cpu + REGIMES["surge"].cpu) / 2
    assert abs(sim.regime.cpu - midpoint) < 1e-6
