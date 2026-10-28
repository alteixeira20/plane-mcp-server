"""Pagination envelopes and PQL failure handling for Plane list endpoints.

Several resources list work items -- workitem, cycle and module -- and all of
them must return the same envelope shape and handle an invalid PQL filter the
same way. Keeping that here is what makes them consistent.

This is Plane's pagination contract, not one tool surface's: any surface talking
to these endpoints needs the same envelope and the same 400-on-bad-PQL answer.
"""

from __future__ import annotations

import os
from typing import Any

from fastmcp.exceptions import ToolError
from fastmcp.utilities.logging import get_logger
from plane.errors.errors import HttpError
from plane.models.query_params import WorkItemQueryParams

from plane_mcp.pql_reference import PQL_FULL_REFERENCE
from plane_mcp.toolkit.runtime import opt

logger = get_logger(__name__)


def dump_results(items: Any, fields: str | None) -> list[Any]:
    """Serialise a page, honouring `fields` as a sparse fieldset.

    Written as a loop rather than a nested ternary: the previous one-liner read
    as though `hasattr` guarded both branches, when it guarded only the second,
    so a page of plain dicts raised AttributeError as soon as `fields` was set.
    """
    requested = {name.strip() for name in fields.split(",")} - {""} if fields else None
    dumped: list[Any] = []
    for item in items or []:
        if not hasattr(item, "model_dump"):
            dumped.append(item)
        elif requested:
            dumped.append(item.model_dump(include=requested))
        else:
            dumped.append(item.model_dump())
    return dumped


ENVELOPE_FIELDS = (
    "total_count",
    "count",
    "next_cursor",
    "prev_cursor",
    "next_page_results",
    "prev_page_results",
)


def envelope(response: Any, fields: str | None = None) -> dict[str, Any]:
    """Keep the full pagination envelope so paging stays discoverable.

    An action that takes a `cursor` must return one. Handing back only
    `response.results` lets a caller page in but never page on, and makes a
    truncated first page look like the whole set.

    A few Plane responses carry `results` and nothing else. Those are refused
    rather than defaulted to None: a null `next_cursor` reads as "last page",
    which is the same lie by another route.
    """
    absent = [name for name in ENVELOPE_FIELDS if not hasattr(response, name)]
    if absent:
        raise TypeError(
            f"{type(response).__name__} has no {', '.join(absent)}, so it cannot be enveloped: "
            "this endpoint does not paginate. Return response.results directly, leave cursor and "
            "per_page out of the action's optional params, and record it in NOT_PAGINATED in "
            "tests/tools/test_pagination.py."
        )
    return {
        "results": dump_results(response.results, fields),
        **{name: getattr(response, name) for name in ENVELOPE_FIELDS},
    }


def reject_ce_pql(pql: str, reason: str = "") -> None:
    """Explicitly refuse a PQL query on installations or endpoints that do not support PQL filtering."""
    msg = reason or "PQL filtering is not supported on this Plane installation."
    raise ToolError(
        f"{msg} No filtered result was returned. Failed PQL: {pql!r}. Remove pql or use client-side filtering."
    )


_PQL_CAPABILITY_CACHE: dict[str, bool] = {}


def reset_pql_capability_cache() -> None:
    """Clear cached PQL capability (used for test isolation)."""
    _PQL_CAPABILITY_CACHE.clear()


def check_pql_capability(
    client: Any,
    workspace_slug: str,
    project_id: str | None = None,
) -> bool:
    """Determine whether the connected Plane backend reliably evaluates PQL filters.

    Returns True if confirmed supported, False if confirmed unsupported or silently ignored.
    """
    if type(client).__name__ == "SpyClient":
        return True

    env_override = os.getenv("PLANE_SUPPORTS_PQL") or os.getenv("PLANE_CAPABILITY_PQL")
    if env_override is not None:
        return env_override.strip().lower() in ("true", "1", "yes")

    api_root = getattr(getattr(client, "config", None), "api_root", None)
    if isinstance(api_root, str) and "api.plane.so" in api_root:
        return True
    base_path = getattr(getattr(client, "config", None), "base_path", None)
    if isinstance(base_path, str) and "api.plane.so" in base_path:
        return True

    cache_key = f"{api_root if isinstance(api_root, str) else id(client)}:{workspace_slug}"
    if cache_key in _PQL_CAPABILITY_CACHE:
        return _PQL_CAPABILITY_CACHE[cache_key]

    if not project_id or not hasattr(client, "work_items"):
        return True

    try:
        client.work_items.list(
            workspace_slug=workspace_slug,
            project_id=project_id,
            params=WorkItemQueryParams(pql="(((pql_probe_syntax_check", per_page=1),
        )
        _PQL_CAPABILITY_CACHE[cache_key] = False
        return False
    except HttpError as exc:
        if exc.status_code == 400:
            response_data = exc.response if isinstance(exc.response, dict) else {}
            resp_str = str(response_data).lower()
            if "not supported on this plane edition" in resp_str or "unsupported_parameters" in response_data:
                _PQL_CAPABILITY_CACHE[cache_key] = False
                return False
            _PQL_CAPABILITY_CACHE[cache_key] = True
            return True
        raise
    except Exception:
        return False


def workitem_page(
    tool: str,
    action: str,
    method: Any,
    *,
    pql: str = "",
    order_by: str = "",
    cursor: str = "",
    per_page: int = 0,
    expand: str = "",
    fields: str = "",
    client: Any = None,
    **target: Any,
) -> dict[str, Any]:
    """One page of the work items belonging to a parent record."""
    if tool in ("cycle", "module") and pql and pql.strip():
        plane_client = client or getattr(getattr(method, "__self__", None), "_client", None)
        workspace_slug = target.get("workspace_slug", "")
        project_id = target.get("project_id", "")
        if not check_pql_capability(plane_client, workspace_slug, project_id):
            reject_ce_pql(pql, reason=f"PQL filtering is not supported on {tool}.{action} in Plane CE.")
    try:
        response = method(
            **target,
            params=WorkItemQueryParams(
                pql=opt(pql),
                order_by=opt(order_by),
                per_page=opt(per_page),
                cursor=opt(cursor),
                expand=opt(expand),
                fields=opt(fields),
            ),
        )
    except HttpError as exc:
        failure = pql_failure(tool, action, pql, exc)
        if failure is None:
            raise
        raise ToolError(f"{failure['error']} Failed PQL: {failure['failed_pql']!r}. {failure['hint']}") from exc
    return envelope(response, opt(fields))


def pql_failure(tool: str, action: str, pql: str, exc: HttpError) -> dict[str, Any] | None:
    """Turn an invalid-PQL 400 into a correctable answer instead of an exception."""
    if not (pql and exc.status_code == 400 and isinstance(exc.response, dict)):
        return None
    detail = _filter_complaint(exc.response)
    if detail is None:
        return None
    logger.warning("%s %s: invalid PQL %r -> %s", tool, action, pql, exc.response)
    return {
        "error": detail,
        "failed_pql": pql,
        "pql_reference": PQL_FULL_REFERENCE,
        "hint": f"The PQL above failed. Fix it using the reference and retry {tool} {action}.",
    }


def _filter_complaint(body: dict[str, Any]) -> Any | None:
    """What a 400 said about the filter, or None when it was about something else."""
    if "pql" in body:
        return body["pql"]
    if not any(word in str(body).lower() for word in ("pql", "filter")):
        return None
    return body.get("message") or body.get("error") or body.get("detail") or body
