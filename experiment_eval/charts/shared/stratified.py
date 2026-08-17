"""Shared ordering helpers for stratified interpretive charts."""

from __future__ import annotations

from typing import Any

from ...schema import group_sort_key


def _run_order(
    rows: list[dict[str, Any]], entity_field: str
) -> list[tuple[str, str, str]]:
    return sorted(
        {
            (str(row[entity_field]), str(row["outer_run_id"]), str(row["stable_id"]))
            for row in rows
        },
        key=lambda value: (group_sort_key(value[0]), group_sort_key(value[1])),
    )
