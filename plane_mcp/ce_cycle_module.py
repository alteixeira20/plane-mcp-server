"""Read-only fallback for standard Plane Community Edition cycle and module list endpoints.

Some self-hosted Plane Community Edition versions (prior to v1.4.0) lack the
dedicated lightweight endpoints (/cycles-lite/, /modules-lite/) and return
HTTP 404. This module provides bounded, truthful fallbacks to the standard
endpoints (/cycles/, /modules/) with honest client-side pagination envelopes
and field trimming conforming to the *Lite models.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from fastmcp.exceptions import ToolError
from plane.models.cycles import CycleLite, PaginatedCycleLiteResponse
from plane.models.modules import ModuleLite, PaginatedModuleLiteResponse

_MAX_ITEMS = 1_000
_MAX_PAGES = 10
_BACKEND_PAGE_SIZE = 100
_MAX_CURSOR_OFFSET_DIGITS = 10

_CYCLE_LITE_FIELDS = set(CycleLite.model_fields.keys())
_MODULE_LITE_FIELDS = set(ModuleLite.model_fields.keys())


def _get_field(obj: Any, field: str, default: Any = None) -> Any:
    """Safely extract a field value from a model or dict without raising."""
    if isinstance(obj, dict):
        val = obj.get(field)
    else:
        val = getattr(obj, field, default)
    return default if val is None else val


def _ce_page(items: list[Any], *, cursor: str, per_page: int, model: type):
    """Create an honest lite-style envelope with CE-specific offset cursors."""
    if not 1 <= per_page <= 1000:
        raise ToolError(f"CE per_page must be in the range 1..1000, got {per_page}.")
    cursor_str = (cursor or "").strip()
    if not cursor_str:
        offset = 0
    elif cursor_str.startswith("ce:") and cursor_str[3:].isascii() and cursor_str[3:].isdigit():
        digits = cursor_str[3:]
        if len(digits) > _MAX_CURSOR_OFFSET_DIGITS:
            safe_cursor = f"'{cursor_str[:32]}...'" if len(cursor_str) > 32 else f"'{cursor_str}'"
            raise ToolError(
                f"Invalid CE pagination cursor: {safe_cursor}. "
                f"Offset exceeds maximum supported digits ({_MAX_CURSOR_OFFSET_DIGITS})."
            )
        offset = int(digits)
    else:
        safe_cursor = f"'{cursor_str[:32]}...'" if len(cursor_str) > 32 else (f"'{cursor_str}'" if cursor_str else "''")
        raise ToolError(f"Invalid CE pagination cursor: {safe_cursor}. Expected empty string or 'ce:<offset>'.")

    total = len(items)
    if offset > total:
        raise ToolError(f"CE pagination cursor offset {offset} is out of range for total {total} items.")
    result = items[offset : offset + per_page]
    has_next = offset + len(result) < total
    has_prev = offset > 0
    payload = {
        "results": result,
        "count": len(result),
        "total_count": total,
        "total_results": total,
        "total_pages": ceil(total / per_page) if per_page else 1,
        "next_page_results": has_next,
        "prev_page_results": has_prev,
        "next_cursor": f"ce:{offset + len(result)}" if has_next else "",
        "prev_cursor": f"ce:{max(0, offset - per_page)}" if has_prev else "",
    }
    return model.model_validate(payload)


def _ce_collect_standard(
    method: Any,
    workspace_slug: str,
    project_id: str,
    *,
    resource_name: str,
) -> list[Any]:
    """Read every standard API page, detecting pagination stalls, duplicate IDs, and bounds."""
    items: list[Any] = []
    seen_ids: set[str] = set()
    seen_cursors: set[str] = set()
    backend_cursor: str | None = None

    for _ in range(_MAX_PAGES):
        params: dict[str, Any] = {"per_page": _BACKEND_PAGE_SIZE}
        if backend_cursor:
            params["cursor"] = backend_cursor

        page = method(
            workspace_slug=workspace_slug,
            project_id=project_id,
            params=params,
        )

        # Handle bare list response (e.g. non-paginated endpoints)
        if isinstance(page, list):
            for item in page:
                item_id = _get_field(item, "id")
                if not item_id:
                    raise ToolError(f"CE {resource_name} record missing required 'id' attribute.")
                item_id_str = str(item_id)
                if item_id_str in seen_ids:
                    raise ToolError(f"CE {resource_name} listing returned duplicate record ID '{item_id_str}'.")
                seen_ids.add(item_id_str)
                items.append(item)
                if len(items) > _MAX_ITEMS:
                    raise ToolError(f"CE {resource_name} listing exceeds maximum safety limit of {_MAX_ITEMS} items.")
            return items

        # Handle paginated response
        results = getattr(page, "results", None)
        if results is None or not isinstance(results, (list, tuple)):
            raise ToolError(f"CE standard {resource_name} list did not return a valid results list.")

        for item in results:
            item_id = _get_field(item, "id")
            if not item_id:
                raise ToolError(f"CE {resource_name} record missing required 'id' attribute.")
            item_id_str = str(item_id)
            if item_id_str in seen_ids:
                raise ToolError(f"CE {resource_name} listing returned duplicate record ID '{item_id_str}'.")
            seen_ids.add(item_id_str)
            items.append(item)
            if len(items) > _MAX_ITEMS:
                raise ToolError(f"CE {resource_name} listing exceeds maximum safety limit of {_MAX_ITEMS} items.")

        if not getattr(page, "next_page_results", False):
            return items

        following = getattr(page, "next_cursor", None)
        if not following or following in seen_cursors or not results:
            raise ToolError(f"CE {resource_name} pagination stalled during scan.")
        seen_cursors.add(following)
        backend_cursor = following
    else:
        raise ToolError(f"CE {resource_name} listing exceeded maximum limit of {_MAX_PAGES} pages.")


def list_cycles_ce(
    client: Any,
    workspace_slug: str,
    project_id: str,
    *,
    cursor: str = "",
    per_page: int = 100,
    status: str = "",
    order_by: str = "",
) -> PaginatedCycleLiteResponse:
    """Use standard cycles API, exclude archived records, and paginate honestly.

    Fails closed if unsupported status or order_by filters are requested.
    """
    if not 1 <= per_page <= 1000:
        raise ToolError(f"CE per_page must be in the range 1..1000, got {per_page}.")
    if status:
        raise ToolError(
            f"Filtering by status='{status}' is not supported on this Plane CE cycle fallback endpoint. "
            "Please retry without status."
        )
    if order_by:
        raise ToolError(
            f"Sorting by '{order_by}' is not supported on this Plane CE cycle fallback endpoint. "
            "Please retry without order_by."
        )

    cycles = _ce_collect_standard(
        client.cycles.list,
        workspace_slug=workspace_slug,
        project_id=project_id,
        resource_name="cycle",
    )

    active_cycles = [c for c in cycles if _get_field(c, "archived_at") is None]
    rows: list[dict[str, Any]] = []
    for c in active_cycles:
        row = (
            c.model_dump(include=_CYCLE_LITE_FIELDS)
            if hasattr(c, "model_dump")
            else {k: v for k, v in c.items() if k in _CYCLE_LITE_FIELDS}
        )
        if not row.get("workspace"):
            row["workspace"] = workspace_slug
        if not row.get("project"):
            row["project"] = project_id
        if row.get("owned_by") is None:
            row["owned_by"] = ""
        rows.append(row)
    return _ce_page(
        rows,
        cursor=cursor,
        per_page=per_page,
        model=PaginatedCycleLiteResponse,
    )


def list_modules_ce(
    client: Any,
    workspace_slug: str,
    project_id: str,
    *,
    cursor: str = "",
    per_page: int = 100,
    order_by: str = "",
) -> PaginatedModuleLiteResponse:
    """Use standard modules API, exclude archived records, and paginate honestly.

    Fails closed if unsupported order_by filter is requested.
    """
    if not 1 <= per_page <= 1000:
        raise ToolError(f"CE per_page must be in the range 1..1000, got {per_page}.")
    if order_by:
        raise ToolError(
            f"Sorting by '{order_by}' is not supported on this Plane CE module fallback endpoint. "
            "Please retry without order_by."
        )

    modules = _ce_collect_standard(
        client.modules.list,
        workspace_slug=workspace_slug,
        project_id=project_id,
        resource_name="module",
    )

    active_modules = [m for m in modules if _get_field(m, "archived_at") is None]
    rows: list[dict[str, Any]] = []
    for m in active_modules:
        row = (
            m.model_dump(include=_MODULE_LITE_FIELDS)
            if hasattr(m, "model_dump")
            else {k: v for k, v in m.items() if k in _MODULE_LITE_FIELDS}
        )
        if not row.get("workspace"):
            row["workspace"] = workspace_slug
        if not row.get("project"):
            row["project"] = project_id
        rows.append(row)
    return _ce_page(
        rows,
        cursor=cursor,
        per_page=per_page,
        model=PaginatedModuleLiteResponse,
    )
