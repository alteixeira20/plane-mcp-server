"""Tests for PlaneReadOnlyMiddleware and PLANE_READ_ONLY configuration.

Verifies:
- Read-only mode is disabled by default and only activates when explicitly set.
- All 82 declared read actions across the 30 resources are permitted.
- All 125 declared write actions across the 30 resources are blocked.
- All 169 legacy aliases are checked and routed to their canonical action classification.
- Single-action tools (e.g. get_pql_reference) are handled when action is omitted.
- Malformed, missing, or unknown actions/tools are rejected safely without crashes.
- Request arguments/payloads (which might carry PII or secrets) are never echoed.
- Negative tests using SpyClient prove rejected operations never reach the SDK.
- Permitted read operations pass through cleanly to the SDK.
"""

from __future__ import annotations

import pytest
from fastmcp import Client, FastMCP
from plane.models.projects import Project
from plane.models.users import UserLite
from plane.models.work_items import PaginatedWorkItemResponse, WorkItem

from plane_mcp.middleware import PlaneReadOnlyMiddleware
from plane_mcp.server import _configured, get_header_mcp, get_oauth_mcp, get_stdio_mcp, is_read_only
from plane_mcp.tools.registry import RESOURCES, alias_table
from tests.tools._spyclient import SpyClient

# --- 1. Configuration & Activation tests ---


def test_is_read_only_disabled_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PLANE_READ_ONLY", raising=False)
    monkeypatch.delenv("PLANE_MCP_READ_ONLY", raising=False)
    assert is_read_only() is False


@pytest.mark.parametrize("val", ["true", "True", "TRUE", "1", "yes", "YES", "Yes"])
def test_is_read_only_truthy_values(monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("PLANE_READ_ONLY", val)
    assert is_read_only() is True


@pytest.mark.parametrize("val", ["false", "0", "no", "", "random", "off"])
def test_is_read_only_falsy_values(monkeypatch: pytest.MonkeyPatch, val: str):
    monkeypatch.setenv("PLANE_READ_ONLY", val)
    assert is_read_only() is False


def test_is_read_only_legacy_alias_env_var(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PLANE_READ_ONLY", raising=False)
    monkeypatch.setenv("PLANE_MCP_READ_ONLY", "true")
    assert is_read_only() is True


def test_plane_read_only_precedence_over_legacy(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLANE_READ_ONLY", "false")
    monkeypatch.setenv("PLANE_MCP_READ_ONLY", "true")
    assert is_read_only() is False


def test_server_configured_without_read_only_middleware_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PLANE_READ_ONLY", raising=False)
    monkeypatch.delenv("PLANE_MCP_READ_ONLY", raising=False)

    mcp = FastMCP("test")
    _configured(mcp)
    types = [type(m) for m in mcp.middleware]
    assert PlaneReadOnlyMiddleware not in types


def test_server_configured_adds_middleware_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLANE_READ_ONLY", "true")

    mcp = FastMCP("test")
    _configured(mcp)
    types = [type(m) for m in mcp.middleware]
    assert PlaneReadOnlyMiddleware in types


def test_all_transports_receive_middleware_when_enabled(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PLANE_READ_ONLY", "true")
    monkeypatch.setenv("PLANE_API_KEY", "test-key")
    monkeypatch.setenv("PLANE_WORKSPACE_SLUG", "test-slug")
    monkeypatch.setenv("PLANE_OAUTH_PROVIDER_BASE_URL", "https://auth.plane.so")
    monkeypatch.setenv("PLANE_OAUTH_PROVIDER_CLIENT_ID", "test-id")
    monkeypatch.setenv("PLANE_OAUTH_PROVIDER_CLIENT_SECRET", "test-secret")

    for factory in (get_stdio_mcp, get_header_mcp, get_oauth_mcp):
        server = factory()
        types = [type(m) for m in server.middleware]
        assert PlaneReadOnlyMiddleware in types, f"{factory.__name__} did not register PlaneReadOnlyMiddleware"


# --- 2. Surface-wide invariant tests ---


@pytest.fixture(scope="module")
def middleware() -> PlaneReadOnlyMiddleware:
    return PlaneReadOnlyMiddleware()


def test_all_read_actions_permitted_surface_wide(middleware: PlaneReadOnlyMiddleware):
    """Every action declared with read=True must be permitted (rejection is None)."""
    verified = 0
    for mod in RESOURCES:
        for action in mod.ACTIONS:
            if action.read:
                refusal = middleware.rejection(mod.NAME, {"action": action.name})
                assert refusal is None, f"{mod.NAME}.{action.name} is marked read=True but was refused: {refusal}"
                verified += 1
    assert verified == 82, f"Expected 82 read actions across surface, verified {verified}"


def test_all_write_actions_blocked_surface_wide(middleware: PlaneReadOnlyMiddleware):
    """Every action declared with read=False must be refused."""
    blocked = 0
    for mod in RESOURCES:
        for action in mod.ACTIONS:
            if not action.read:
                refusal = middleware.rejection(mod.NAME, {"action": action.name})
                expected = f"Error: Plane MCP read-only mode blocks {mod.NAME}.{action.name}."
                assert refusal == expected, f"{mod.NAME}.{action.name} had unexpected refusal: {refusal}"
                blocked += 1
    assert blocked == 125, f"Expected 125 write actions across surface, blocked {blocked}"


def test_every_legacy_alias_correctly_classified(middleware: PlaneReadOnlyMiddleware):
    """Every one of the 169 retired aliases maps to its canonical action's read status."""
    aliases = alias_table()
    action_map = {mod.NAME: {a.name: a.read for a in mod.ACTIONS} for mod in RESOURCES}

    assert len(aliases) == 169
    for alias, (resource, action) in aliases.items():
        is_read = action_map[resource][action]
        refusal = middleware.rejection(alias, {})
        if is_read:
            assert refusal is None, f"Read alias {alias} -> {resource}.{action} was refused: {refusal}"
        else:
            expected = f"Error: Plane MCP read-only mode blocks {resource}.{action}."
            assert refusal == expected, f"Write alias {alias} -> {resource}.{action} expected {expected}, got {refusal}"


def test_single_action_tool_without_action_parameter(middleware: PlaneReadOnlyMiddleware):
    """Single-action tools take no action parameter; read-only must resolve their only action."""
    assert middleware.rejection("get_pql_reference", {}) is None
    assert middleware.rejection("get_pql_reference", None) is None


def test_unrecognised_tool_is_safely_blocked(middleware: PlaneReadOnlyMiddleware):
    refusal = middleware.rejection("nonexistent_unknown_tool", {})
    assert refusal == "Error: Plane MCP read-only mode blocks an unrecognised tool 'nonexistent_unknown_tool'."


def test_malformed_and_missing_arguments_safely_handled(middleware: PlaneReadOnlyMiddleware):
    """Ensure no exceptions occur when arguments are None, not a dict, or missing action."""
    assert middleware.rejection("workitem", None) == "Error: Plane MCP read-only mode blocks workitem.unknown."
    assert middleware.rejection("workitem", {}) == "Error: Plane MCP read-only mode blocks workitem.unknown."
    assert (
        middleware.rejection("workitem", {"action": None}) == "Error: Plane MCP read-only mode blocks workitem.unknown."
    )
    assert (
        middleware.rejection("workitem", {"action": 123}) == "Error: Plane MCP read-only mode blocks workitem.unknown."
    )
    assert (
        middleware.rejection("workitem", {"action": "not_an_action"})
        == "Error: Plane MCP read-only mode blocks workitem.not_an_action."
    )


def test_argument_payload_never_echoed_in_refusal(middleware: PlaneReadOnlyMiddleware):
    """Refusals must never echo arguments, which might contain private tokens or user data."""
    secret = "sk-super-secret-key-that-must-not-leak"
    refusal = middleware.rejection("workitem", {"action": "create", "secret_content": secret})
    assert refusal is not None
    assert secret not in refusal


# --- 3. End-to-end integration & SDK isolation tests ---


class _AllFeaturesOn:
    def model_dump(self):
        return {"initiatives": True, "customers": True, "releases": True, "epics": True}


@pytest.fixture
def spy(monkeypatch: pytest.MonkeyPatch):
    """Inject a recording SpyClient across all resource modules."""
    client = SpyClient()
    client.returns["workspaces.get_features"] = _AllFeaturesOn()
    # Populate returns with valid Pydantic models so FastMCP serializer succeeds
    client.returns["work_items.retrieve"] = WorkItem(id="w1", name="Item 1", project="p1")
    client.returns["work_items.list"] = PaginatedWorkItemResponse(
        results=[],
        total_count=0,
        next_cursor="",
        prev_cursor="",
        next_page_results=False,
        prev_page_results=False,
        count=0,
        total_pages=0,
        total_results=0,
    )
    client.returns["work_items.create"] = WorkItem(id="w1", name="Created Item", project="p1")
    client.returns["projects.retrieve"] = Project(id="p1", name="Project 1", identifier="PROJ")
    client.returns["users.get_me"] = UserLite(id="u1", email="me@example.com")

    for mod in RESOURCES:
        if hasattr(mod, "get_plane_client_context"):
            monkeypatch.setattr(mod, "get_plane_client_context", lambda: (client, "test-ws"))
    return client


@pytest.mark.anyio
async def test_read_only_mode_blocks_writes_before_sdk(monkeypatch: pytest.MonkeyPatch, spy: SpyClient):
    """Negative test: Prove that refused write operations NEVER reach the SDK."""
    monkeypatch.setenv("PLANE_READ_ONLY", "true")
    monkeypatch.setenv("PLANE_API_KEY", "test-key")
    monkeypatch.setenv("PLANE_WORKSPACE_SLUG", "test-ws")

    server = get_stdio_mcp()

    write_calls = [
        ("workitem", {"action": "create", "project_id": "p1", "name": "Test Item"}),
        ("workitem", {"action": "delete", "project_id": "p1", "workitem_id": "w1"}),
        ("project", {"action": "update", "project_id": "p1", "name": "Updated"}),
        ("cycle", {"action": "create", "project_id": "p1", "name": "Sprint 1", "owned_by": "u1"}),
        ("label", {"action": "delete", "project_id": "p1", "label_id": "l1"}),
        ("create_issue", {"project_id": "p1", "name": "Legacy Issue"}),
        ("delete_issue", {"project_id": "p1", "work_item_id": "w1"}),
    ]

    async with Client(server) as client:
        for tool_name, arguments in write_calls:
            result = await client.call_tool(tool_name, arguments)
            text = result.content[0].text if result.content else ""
            assert "Error: Plane MCP read-only mode blocks" in text, f"Expected read-only refusal for {tool_name}"

    # Assert that zero calls reached the PlaneClient / SpyClient!
    assert len(spy.recorder.calls) == 0, f"Expected 0 SDK calls, but {len(spy.recorder.calls)} were recorded"


@pytest.mark.anyio
async def test_read_only_mode_permits_reads_to_sdk(monkeypatch: pytest.MonkeyPatch, spy: SpyClient):
    """Positive test: Prove that permitted read operations pass through and reach the SDK."""
    monkeypatch.setenv("PLANE_READ_ONLY", "true")
    monkeypatch.setenv("PLANE_API_KEY", "test-key")
    monkeypatch.setenv("PLANE_WORKSPACE_SLUG", "test-ws")

    server = get_stdio_mcp()

    read_calls = [
        ("get_pql_reference", {}),
        ("project", {"action": "retrieve", "project_id": "p1"}),
        ("workitem", {"action": "retrieve", "project_id": "p1", "workitem_id": "w1"}),
        ("member", {"action": "me"}),
        ("retrieve_work_item", {"project_id": "p1", "work_item_id": "w1"}),
    ]

    async with Client(server) as client:
        for tool_name, arguments in read_calls:
            result = await client.call_tool(tool_name, arguments)
            text = result.content[0].text if result.content else ""
            assert "Error: Plane MCP read-only mode blocks" not in text

    # Assert that read calls reached the PlaneClient / SpyClient
    assert len(spy.recorder.calls) > 0, "Expected SDK calls for permitted read operations"


@pytest.mark.anyio
async def test_disabled_by_default_permits_writes(monkeypatch: pytest.MonkeyPatch, spy: SpyClient):
    """When disabled, write operations pass through the middleware stack to the SDK."""
    monkeypatch.delenv("PLANE_READ_ONLY", raising=False)
    monkeypatch.delenv("PLANE_MCP_READ_ONLY", raising=False)
    monkeypatch.setenv("PLANE_API_KEY", "test-key")
    monkeypatch.setenv("PLANE_WORKSPACE_SLUG", "test-ws")

    server = get_stdio_mcp()

    async with Client(server) as client:
        result = await client.call_tool("workitem", {"action": "create", "project_id": "p1", "name": "Item"})
        text = result.content[0].text if result.content else ""
        assert "Error: Plane MCP read-only mode blocks" not in text

    # Write operation reached the SDK
    assert len(spy.recorder.calls) == 1
    assert spy.recorder.calls[0].method == "work_items.create"


@pytest.mark.anyio
async def test_middleware_order_preserves_validate_action_arguments(monkeypatch: pytest.MonkeyPatch):
    """When action is omitted on a multi-action tool, ValidateActionArguments reports the actions."""
    monkeypatch.setenv("PLANE_READ_ONLY", "true")
    monkeypatch.setenv("PLANE_API_KEY", "test-key")
    monkeypatch.setenv("PLANE_WORKSPACE_SLUG", "test-ws")

    server = get_stdio_mcp()

    async with Client(server) as client:
        result = await client.call_tool("workitem", {"project_id": "p1"})
        text = result.content[0].text if result.content else ""
        assert "Error: workitem requires an action. It takes:" in text
