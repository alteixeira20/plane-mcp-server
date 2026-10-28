"""Deterministic tests for PQL fail-closed behavior on self-hosted Plane CE and error conditions.

Covers:
- Scenario A: Backend returns HTTP 400 with an invalid PQL message.
- Scenario B: Backend returns HTTP 200 while ignoring a valid PQL predicate.
- Scenario C: Backend correctly applies the requested PQL filter.
- Scenario D: Backend returns a valid empty filtered result.
- Scenario E: Backend returns paginated results with filters correctly applied.
- Scenario F: Backend claims success on cycle/module with PQL on CE.
- Scenario G: Backend supports unrelated query parameters but ignores PQL.
- Adversarial tests: probe mechanics, auth failure preservation, unfiltered regression,
  raw MCP isError=true contract, and schema stability.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest
from fastmcp import Client, FastMCP
from plane.errors.errors import HttpError
from plane.models.work_items import PaginatedWorkItemResponse, WorkItem

from plane_mcp.toolkit.paging import (
    reset_pql_capability_cache,
)
from plane_mcp.tools.cycle import register as register_cycle
from plane_mcp.tools.module import register as register_module
from plane_mcp.tools.workitem import register as register_workitem


def _make_http_400(response_body: dict) -> HttpError:
    return HttpError("Bad Request", status_code=400, response=response_body)


def _make_work_item(
    item_id: str,
    name: str = "Test item",
    priority: str = "urgent",
    state: str = "state-1",
    assignees: list[str] | None = None,
) -> WorkItem:
    return WorkItem.model_validate(
        {
            "id": item_id,
            "name": name,
            "priority": priority,
            "state": state,
            "assignees": assignees or ["user-1"],
            "project": "00000000-0000-0000-0000-000000000001",
            "workspace": "test-ws",
            "sequence_id": 1,
            "sort_order": 1000.0,
            "is_draft": False,
        }
    )


def _make_page(
    items: list[WorkItem],
    total_count: int | None = None,
    next_cursor: str = "",
    prev_cursor: str = "",
) -> PaginatedWorkItemResponse:
    count_val = total_count if total_count is not None else len(items)
    return PaginatedWorkItemResponse.model_validate(
        {
            "results": items,
            "count": len(items),
            "total_count": count_val,
            "total_results": count_val,
            "next_cursor": next_cursor,
            "prev_cursor": prev_cursor,
            "next_page_results": bool(next_cursor),
            "prev_page_results": bool(prev_cursor),
            "total_pages": 1 if not next_cursor else 2,
            "extra_stats": None,
        }
    )


@pytest.fixture(autouse=True)
def clean_pql_cache():
    """Ensure capability cache is fresh for every test."""
    reset_pql_capability_cache()
    yield
    reset_pql_capability_cache()


@pytest.fixture
def test_server():
    server = FastMCP("test-server")
    register_workitem(server)
    register_cycle(server)
    register_module(server)
    return server


# ==============================================================================
# Scenario A: Backend returns HTTP 400 with an invalid PQL message
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_a_invalid_pql_raises_tool_error_with_is_error_true(test_server: FastMCP):
    """When the backend returns HTTP 400 for PQL, tool raises ToolError and MCP sets isError: true."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"
    mock_client.work_items.list.side_effect = _make_http_400({"pql": "Unknown field 'stat'"})

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'stat = "done"',
                },
            )
            assert res.isError is True
            assert len(res.content) > 0
            err_text = res.content[0].text
            assert "Unknown field 'stat'" in err_text
            assert "Failed PQL:" in err_text
            assert 'stat = "done"' in err_text


@pytest.mark.anyio
async def test_scenario_a_ce_explicit_400_refusal_propagates_as_tool_error(test_server: FastMCP):
    """When Plane CE v1.4.1+ returns 400 unsupported PQL, ToolError carries the refusal loudly."""
    ce_body = {
        "pql": (
            "PQL and structured filters are not supported on this Plane edition. "
            "Remove the pql/filters parameter and filter results client-side."
        ),
        "unsupported_parameters": ["pql"],
    }
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    mock_client.work_items.list.side_effect = _make_http_400(ce_body)

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is True
            err_text = res.content[0].text
            assert (
                "not supported on this Plane edition" in err_text
                or "not supported on this Plane installation" in err_text
            )
            assert 'priority = "urgent"' in err_text


# ==============================================================================
# Scenario B: Backend returns HTTP 200 while ignoring a valid PQL predicate (CE silent ignore)
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_b_silent_http_200_ignored_pql_fails_closed(test_server: FastMCP):
    """On self-hosted CE <= 1.3.1, broken probe PQL returns 200, proving silent ignore. Tool must fail closed."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    # Both probe and user query would return HTTP 200 with unfiltered data on Plane CE <= 1.3.1
    unfiltered_page = _make_page(
        [
            _make_work_item("item-1", priority="low"),
            _make_work_item("item-2", priority="medium"),
            _make_work_item("item-3", priority="urgent"),
        ],
        total_count=3,
    )
    mock_client.work_items.list.return_value = unfiltered_page

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            # The tool must fail closed with isError: true and NOT return the unfiltered 3 items!
            assert res.isError is True
            err_text = res.content[0].text
            assert "PQL filtering is not supported on this Plane installation" in err_text
            assert "No filtered result was returned" in err_text
            assert 'priority = "urgent"' in err_text
            assert "item-1" not in err_text
            assert "item-2" not in err_text


# ==============================================================================
# Scenario C: Backend correctly applies the requested PQL filter
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_c_supported_pql_returns_filtered_results(test_server: FastMCP):
    """When connected to Plane Cloud or supported backend, PQL filter is dispatched and results returned."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"
    urgent_items = [
        _make_work_item("item-1", priority="urgent"),
        _make_work_item("item-2", priority="urgent"),
    ]
    mock_client.work_items.list.return_value = _make_page(urgent_items, total_count=2)

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is False
            # Verify SDK list was called with the PQL filter
            assert mock_client.work_items.list.called
            call_kwargs = mock_client.work_items.list.call_args.kwargs
            assert call_kwargs["params"].pql == 'priority = "urgent"'


# ==============================================================================
# Scenario D: Backend returns a valid empty filtered result
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_d_valid_empty_filtered_result(test_server: FastMCP):
    """When a filter matches zero items on a supported backend, empty results are returned with isError: false."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"
    mock_client.work_items.list.return_value = _make_page([], total_count=0)

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is False
            data = json.loads(res.content[0].text)
            assert data["results"] == []
            assert data["total_count"] == 0


# ==============================================================================
# Scenario E: Backend returns paginated results with filters correctly applied
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_e_paginated_filtered_results_preserve_envelope(test_server: FastMCP):
    """Multi-page filtered results preserve complete cursor and pagination contracts."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"
    page1_items = [_make_work_item(f"item-{i}", priority="urgent") for i in range(10)]
    mock_client.work_items.list.return_value = _make_page(page1_items, total_count=25, next_cursor="cursor_page_2")

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                    "per_page": 10,
                },
            )
            assert res.isError is False
            data = json.loads(res.content[0].text)
            assert data["next_cursor"] == "cursor_page_2"
            assert data["next_page_results"] is True
            assert data["total_count"] == 25
            assert data["count"] == 10


# ==============================================================================
# Scenario F: Backend claims success on cycle/module with PQL on Plane CE
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_f_cycle_workitems_pql_fails_closed_on_ce(test_server: FastMCP):
    """cycle.list_workitems with PQL fails closed with ToolError on Plane CE."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    # Probe returns 200 (silent ignore CE <= 1.3.1)
    mock_client.work_items.list.return_value = _make_page([])

    with patch("plane_mcp.tools.cycle.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "cycle",
                {
                    "action": "list_workitems",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "cycle_id": "00000000-0000-0000-0000-000000000002",
                    "pql": 'priority = "high"',
                },
            )
            assert res.isError is True
            err_text = res.content[0].text
            assert "PQL filtering is not supported on cycle.list_workitems in Plane CE" in err_text
            assert 'priority = "high"' in err_text
            # Client cycle list_work_items must not be called
            assert not mock_client.cycles.list_work_items.called


@pytest.mark.anyio
async def test_scenario_f_module_workitems_pql_fails_closed_on_ce(test_server: FastMCP):
    """module.list_workitems with PQL fails closed with ToolError on Plane CE."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    # Probe returns 200 (silent ignore CE <= 1.3.1)
    mock_client.work_items.list.return_value = _make_page([])

    with patch("plane_mcp.tools.module.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "module",
                {
                    "action": "list_workitems",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "module_id": "00000000-0000-0000-0000-000000000003",
                    "pql": 'priority = "high"',
                },
            )
            assert res.isError is True
            err_text = res.content[0].text
            assert "PQL filtering is not supported on module.list_workitems in Plane CE" in err_text
            assert 'priority = "high"' in err_text
            assert not mock_client.modules.list_work_items.called


# ==============================================================================
# Scenario G: Backend supports unrelated query parameters but ignores PQL
# ==============================================================================


@pytest.mark.anyio
async def test_scenario_g_unrelated_query_params_with_ignored_pql_fails_closed(test_server: FastMCP):
    """When order_by and per_page are provided alongside an ignored PQL, tool fails closed."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    # Probe returns 200
    mock_client.work_items.list.return_value = _make_page([_make_work_item("item-1")])

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'state__group = "started"',
                    "order_by": "-created_at",
                    "per_page": 20,
                },
            )
            assert res.isError is True
            err_text = res.content[0].text
            assert "PQL filtering is not supported on this Plane installation" in err_text


# ==============================================================================
# Adversarial & Regression Tests
# ==============================================================================


@pytest.mark.anyio
async def test_unfiltered_list_preserves_behavior(test_server: FastMCP):
    """Unfiltered queries (no pql) bypass capability checks and succeed as normal."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    items = [_make_work_item("item-1"), _make_work_item("item-2")]
    mock_client.work_items.list.return_value = _make_page(items, total_count=2)

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                },
            )
            assert res.isError is False
            data = json.loads(res.content[0].text)
            assert data["total_count"] == 2
            call_kwargs = mock_client.work_items.list.call_args.kwargs
            assert call_kwargs["params"].pql is None


@pytest.mark.anyio
async def test_env_override_plane_supports_pql_false(test_server: FastMCP, monkeypatch):
    """Explicit PLANE_SUPPORTS_PQL=false immediately fails closed without probing."""
    monkeypatch.setenv("PLANE_SUPPORTS_PQL", "false")
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"  # Even on plane.so, env takes precedence

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is True
            assert "PQL filtering is not supported on this Plane installation" in res.content[0].text
            assert not mock_client.work_items.list.called


@pytest.mark.anyio
async def test_env_override_plane_supports_pql_true(test_server: FastMCP, monkeypatch):
    """Explicit PLANE_SUPPORTS_PQL=true confirms support without probing self-hosted."""
    monkeypatch.setenv("PLANE_SUPPORTS_PQL", "true")
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted-enterprise.corp"
    mock_client.work_items.list.return_value = _make_page([_make_work_item("item-1")], total_count=1)

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is False
            # Dispatched directly once with the user PQL (no probe call)
            assert mock_client.work_items.list.call_count == 1
            call_kwargs = mock_client.work_items.list.call_args.kwargs
            assert call_kwargs["params"].pql == 'priority = "urgent"'


@pytest.mark.anyio
async def test_self_hosted_enterprise_confirmed_by_syntax_rejection(test_server: FastMCP):
    """Self-hosted Enterprise parses PQL, so probe gets HTTP 400 syntax error, confirming support."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted-enterprise.corp"

    # First call is the probe with invalid syntax -> returns 400 syntax error
    # Second call is the actual user query -> returns filtered page
    def _side_effect(*args, **kwargs):
        params = kwargs.get("params")
        if params and "pql_probe_syntax_check" in (params.pql or ""):
            raise _make_http_400({"pql": "Invalid PQL syntax at position 1"})
        return _make_page([_make_work_item("item-1", priority="urgent")], total_count=1)

    mock_client.work_items.list.side_effect = _side_effect

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is False
            data = json.loads(res.content[0].text)
            assert data["total_count"] == 1
            assert mock_client.work_items.list.call_count == 2


@pytest.mark.anyio
async def test_cycle_and_module_workitems_pql_succeed_on_supported_backend(test_server: FastMCP):
    """cycle and module list_workitems with PQL succeed when connected to Plane Cloud."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"
    mock_client.cycles.list_work_items.return_value = _make_page([_make_work_item("c-1")])
    mock_client.modules.list_work_items.return_value = _make_page([_make_work_item("m-1")])

    with (
        patch("plane_mcp.tools.cycle.get_plane_client_context", return_value=(mock_client, "test-ws")),
        patch("plane_mcp.tools.module.get_plane_client_context", return_value=(mock_client, "test-ws")),
    ):
        async with Client(test_server) as client:
            res_c = await client.call_tool_mcp(
                "cycle",
                {
                    "action": "list_workitems",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "cycle_id": "00000000-0000-0000-0000-000000000002",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res_c.isError is False
            assert mock_client.cycles.list_work_items.called

            res_m = await client.call_tool_mcp(
                "module",
                {
                    "action": "list_workitems",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "module_id": "00000000-0000-0000-0000-000000000003",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res_m.isError is False
            assert mock_client.modules.list_work_items.called


@pytest.mark.anyio
async def test_count_with_invalid_pql_raises_tool_error(test_server: FastMCP):
    """workitem count with invalid PQL raises ToolError with isError: true."""
    mock_client = MagicMock()
    mock_client.work_items.count_workspace.side_effect = _make_http_400({"pql": "Unknown field 'stat'"})

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "count",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'stat = "done"',
                },
            )
            assert res.isError is True
            assert "Unknown field 'stat'" in res.content[0].text
            assert "Failed PQL:" in res.content[0].text


@pytest.mark.anyio
async def test_backend_auth_failure_propagates_unmasked(test_server: FastMCP):
    """HTTP 401 or 403 authorization failures are propagated directly rather than converted to PQL errors."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://self-hosted.local"
    mock_client.work_items.list.side_effect = HttpError("Forbidden", status_code=403, response={"detail": "Forbidden"})

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is True
            assert "403" in res.content[0].text or "Forbidden" in res.content[0].text


@pytest.mark.anyio
async def test_backend_server_error_propagates_unmasked(test_server: FastMCP):
    """HTTP 500 server errors are propagated directly rather than swallowed."""
    mock_client = MagicMock()
    mock_client.config.api_root = "https://api.plane.so"
    mock_client.work_items.list.side_effect = HttpError(
        "Internal Error", status_code=500, response={"detail": "DB down"}
    )

    with patch("plane_mcp.tools.workitem.get_plane_client_context", return_value=(mock_client, "test-ws")):
        async with Client(test_server) as client:
            res = await client.call_tool_mcp(
                "workitem",
                {
                    "action": "list",
                    "project_id": "00000000-0000-0000-0000-000000000001",
                    "pql": 'priority = "urgent"',
                },
            )
            assert res.isError is True
            assert "500" in res.content[0].text or "DB down" in res.content[0].text


@pytest.mark.anyio
async def test_tool_schema_stability(test_server: FastMCP):
    """Tool schema parameters and descriptions remain stable without unexpected drift."""
    async with Client(test_server) as client:
        tools = await client.list_tools()
        tool_names = {t.name for t in tools}
        assert {"workitem", "cycle", "module"}.issubset(tool_names)

        workitem_tool = next(t for t in tools if t.name == "workitem")
        props = workitem_tool.inputSchema.get("properties", {})
        assert "pql" in props
        assert "action" in props
        assert "project_id" in props
        assert props["pql"].get("type") == "string"
