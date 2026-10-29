"""Read-only fallback for standard Plane Community Edition list endpoints.

Some self-hosted Plane Community Edition versions (prior to v1.4.0) lack the
dedicated lightweight endpoints (/projects-lite/, /members-lite/) and return
HTTP 404. This module provides bounded, truthful fallbacks to the standard
endpoints with honest client-side filtering and pagination envelopes.
"""

from __future__ import annotations

from math import ceil
from typing import Any

from fastmcp.exceptions import ToolError
from plane.models.projects import PaginatedProjectLiteResponse, ProjectLite
from plane.models.query_params import PaginatedQueryParams
from plane.models.workspaces import PaginatedWorkspaceMemberResponse

_MAX_PROJECTS = 10_000
_MAX_MEMBERS = 10_000
_MAX_PAGES = 100
_BACKEND_PAGE_SIZE = 100
_PROJECT_LITE_FIELDS = set(ProjectLite.model_fields.keys())


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
        offset = int(cursor_str[3:])
    else:
        raise ToolError(f"Invalid CE pagination cursor: '{cursor}'. Expected empty string or 'ce:<offset>'.")

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


def list_projects_ce(
    client: Any,
    workspace_slug: str,
    *,
    cursor: str = "",
    per_page: int = 100,
    order_by: str = "",
) -> PaginatedProjectLiteResponse:
    """Read standard CE project pages, exclude archived records, then paginate.

    Refuses excessively large workspaces instead of returning partial or truncated results.
    Detects pagination stalls and repeated IDs during scan.
    """
    projects: list[Any] = []
    seen_ids: set[str] = set()
    seen_cursors: set[str] = set()
    backend_cursor: str | None = None

    for _ in range(_MAX_PAGES):
        page = client.projects.list(
            workspace_slug=workspace_slug,
            params=PaginatedQueryParams(
                cursor=backend_cursor,
                per_page=_BACKEND_PAGE_SIZE,
                order_by=order_by or None,
            ),
        )
        results = getattr(page, "results", None)
        if results is None or not isinstance(results, (list, tuple)):
            raise ToolError("CE project listing did not return a valid paginated response.")
        for project in results:
            pid = _get_field(project, "id")
            if not pid or pid in seen_ids:
                raise ToolError("CE project pagination returned missing or duplicate project IDs.")
            seen_ids.add(pid)
            projects.append(project)

        if len(projects) > _MAX_PROJECTS:
            raise ToolError(f"CE project listing exceeds maximum scan limit of {_MAX_PROJECTS} projects.")

        if not getattr(page, "next_page_results", False):
            break

        following = getattr(page, "next_cursor", None)
        if not following or following in seen_cursors or not results:
            raise ToolError("CE project pagination stalled during scan.")
        seen_cursors.add(following)
        backend_cursor = following
    else:
        raise ToolError(f"CE project listing exceeded maximum limit of {_MAX_PAGES} pages.")

    active_projects = [
        project.model_dump(include=_PROJECT_LITE_FIELDS)
        if hasattr(project, "model_dump")
        else {k: v for k, v in project.items() if k in _PROJECT_LITE_FIELDS}
        for project in projects
        if _get_field(project, "archived_at") is None
    ]
    return _ce_page(
        active_projects,
        cursor=cursor,
        per_page=per_page,
        model=PaginatedProjectLiteResponse,
    )


def list_workspace_members_ce(
    client: Any,
    workspace_slug: str,
    *,
    cursor: str = "",
    per_page: int = 100,
    order_by: str = "",
    first_name: str = "",
    last_name: str = "",
    email: str = "",
    display_name: str = "",
    role_slug: str = "",
    is_active: bool | None = None,
    is_bot: bool | None = None,
) -> PaginatedWorkspaceMemberResponse:
    """Use the standard CE members endpoint with truthful client-side filters and pagination."""
    members = client.workspaces.get_members(workspace_slug=workspace_slug)
    if not isinstance(members, (list, tuple)):
        raise ToolError("CE workspace member listing did not return a valid list.")
    if len(members) > _MAX_MEMBERS:
        raise ToolError(f"CE workspace member listing exceeds maximum limit of {_MAX_MEMBERS} members.")

    text_filters = {
        "first_name": first_name,
        "last_name": last_name,
        "email": email,
        "display_name": display_name,
    }
    for field, value in text_filters.items():
        if value:
            needle = value.casefold()
            members = [m for m in members if needle in str(_get_field(m, field, "")).casefold()]

    exact_filters = (
        ("role_slug", role_slug),
        ("is_active", is_active),
        ("is_bot", is_bot),
    )
    for field, value in exact_filters:
        if value is None or value == "":
            continue
        if any(_get_field(m, field, None) is None for m in members):
            raise ToolError(f"CE cannot reliably filter by {field}: API field unavailable on records.")
        if field == "role_slug":
            members = [m for m in members if _get_field(m, "role_slug") == value]
        else:
            members = [m for m in members if _get_field(m, field) is value]

    if order_by:
        reverse = order_by.startswith("-")
        field = order_by[1:] if reverse else order_by
        allowed_sort_fields = {
            "first_name",
            "last_name",
            "email",
            "display_name",
            "id",
        }
        if field not in allowed_sort_fields:
            raise ToolError(
                f"Unsupported CE workspace member order_by: '{order_by}'. "
                f"Allowed fields: {', '.join(sorted(allowed_sort_fields))}."
            )
        members.sort(
            key=lambda m: (
                _get_field(m, field, None) is None,
                str(_get_field(m, field, "")).casefold(),
                str(_get_field(m, "id", "")),
            ),
            reverse=reverse,
        )

    rows = [m.model_dump() if hasattr(m, "model_dump") else m for m in members]
    return _ce_page(
        rows,
        cursor=cursor,
        per_page=per_page,
        model=PaginatedWorkspaceMemberResponse,
    )
