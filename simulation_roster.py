"""Side-effect-free simulation roster and asset-root resolution."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


DEFAULT_VILLAGE_PERSONAS = (
    "卡布达",
    "金龟次郎",
    "田德莉娜",
    "呱呱蛙",
    "蜻蜓队长",
    "蟑螂恶霸",
)


def resolve_run_context(
    checkpoints_folder: str | Path,
    assets_root_override: str | None = None,
) -> tuple[str, list[str]]:
    """Resolve assets and roster using the historical compress precedence."""

    checkpoints_path = Path(checkpoints_folder)
    assets_root = assets_root_override
    roster: list[str] | None = None
    try:
        snapshots = sorted(checkpoints_path.glob("simulate-*.json"))
    except OSError:
        snapshots = []

    if snapshots:
        try:
            snapshot = json.loads(snapshots[0].read_text(encoding="utf-8"))
        except (OSError, ValueError):
            snapshot = {}
        if assets_root is None:
            maze_field = snapshot.get("maze")
            maze_path = ""
            if isinstance(maze_field, dict):
                maze_path = str(maze_field.get("path", "") or "")
            assets_root = "counsel_room" if "counsel_room" in maze_path else "village"

        agents = snapshot.get("agents")
        if isinstance(agents, dict) and agents:
            roster = list(agents)

    return assets_root or "village", roster or list(DEFAULT_VILLAGE_PERSONAS)


def replay_roster(payload: Any) -> list[str]:
    """Resolve the roster embedded in compressed movement data."""

    if isinstance(payload, dict):
        initial_positions = payload.get("persona_init_pos")
        if isinstance(initial_positions, dict) and initial_positions:
            return list(initial_positions)
    return list(DEFAULT_VILLAGE_PERSONAS)
