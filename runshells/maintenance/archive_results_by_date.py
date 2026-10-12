#!/usr/bin/env python3
"""按日期将 results 中尚未归档的实验内容移入单独目录。

修改下方“运行配置”后，直接运行：
    python3 runshells/maintenance/archive_results_by_date.py
"""

import re
from pathlib import Path


# ========================= 运行配置 =========================
# 实验日期，使用文件/目录名中的四位 MMDD；例如 "0928"。
EXPERIMENT_DATE = "0928"

# results 下的目标目录名。允许目录已经存在（例如里面已有绘图），
# 但遇到同名目标文件或目录时会在移动前报错，不会覆盖。
ARCHIVE_NAME = "0928-g1-2-5-9"

# True：只打印迁移清单；False：实际移动。首次使用可先改为 True 核对。
DRY_RUN = False

# results 目录的位置；通常无需修改。
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results"
# ============================================================


SHARED_DIRS = (
    "checkpoints",
    "compressed",
    "experiment_data",
    "backup-normal",
    "recovery_backups",
    "recovery_locks",
)
EXPERIMENT_DATA_SUBDIRS = ("api_cost", "batch_state", "repeat_scale_eval", "reports")


def collect_moves(results: Path, date: str, archive_name: str) -> list[tuple[Path, Path]]:
    """选出日期对应的完整条目，不进入已有的其他归档目录。"""
    matches_date = re.compile(rf"(?<!\d){re.escape(date)}(?!\d)").search
    destination = results / archive_name
    moves = []

    for entry in sorted(results.iterdir()):
        if entry.is_file() and matches_date(entry.name):
            moves.append((entry, destination / entry.name))
        elif entry.is_dir() and (entry.name == f"log-{date}" or entry.name == f"{date}画图"):
            moves.append((entry, destination / entry.name))

    for shared_name in SHARED_DIRS:
        shared = results / shared_name
        if not shared.is_dir():
            continue
        for entry in sorted(shared.iterdir()):
            if matches_date(entry.name):
                moves.append((entry, destination / shared_name / entry.name))
            elif shared_name == "experiment_data" and entry.name in EXPERIMENT_DATA_SUBDIRS:
                for nested in sorted(entry.iterdir()):
                    if matches_date(nested.name):
                        moves.append((nested, destination / shared_name / entry.name / nested.name))

    return moves


def main() -> None:
    if not re.fullmatch(r"\d{4}", EXPERIMENT_DATE):
        raise SystemExit("EXPERIMENT_DATE 必须是四位 MMDD")
    archive_name = ARCHIVE_NAME or EXPERIMENT_DATE
    if Path(archive_name).name != archive_name or archive_name in (".", ".."):
        raise SystemExit("ARCHIVE_NAME 必须是单个目录名")
    if not archive_name.startswith(EXPERIMENT_DATE):
        raise SystemExit("ARCHIVE_NAME 必须以实验日期开头")

    results = RESULTS_DIR.resolve()
    if not results.is_dir():
        raise SystemExit(f"results 目录不存在：{results}")
    destination = results / archive_name
    if destination.exists() and not destination.is_dir():
        raise SystemExit(f"归档目标不是目录：{destination}")

    moves = collect_moves(results, EXPERIMENT_DATE, archive_name)
    conflicts = [target for _, target in moves if target.exists()]
    if conflicts:
        raise SystemExit("目标路径已存在，未移动任何内容：\n" + "\n".join(map(str, conflicts)))

    for source, target in moves:
        print(f"{source.relative_to(results)} -> {target.relative_to(results)}")
        if not DRY_RUN:
            target.parent.mkdir(parents=True, exist_ok=True)
            source.rename(target)
    print(f"{'计划移动' if DRY_RUN else '已移动'} {len(moves)} 项")


if __name__ == "__main__":
    main()
