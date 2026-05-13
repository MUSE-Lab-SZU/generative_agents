#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


# ===== 可直接修改的默认参数 =====
# 直接执行 `python scripts/intervention_prompt_txt_sync.py` 时，会使用下面这些默认值。
# 如果同时传入命令行参数，则命令行参数优先。
DEFAULT_MODE = "export"  # 可选: "export" 或 "import"
DEFAULT_JSON_PATH = Path("./data/prompts/intervention_prompts.json")  # 要读写的 JSON 文件

# export 模式：从 JSON 导出多个 txt 文件。
DEFAULT_EXPORT_GROUP = "CBT"  # 要导出的一级键，例如 CBT / PST
DEFAULT_OUTPUT_DIR = Path(f"./prompt_txt/{DEFAULT_EXPORT_GROUP}")  # txt 导出目录
DEFAULT_EXPORT_OVERWRITE = True  # 若同名 txt 已存在，是否覆盖

# import 模式：把 txt 文件夹内容写回 JSON。
DEFAULT_IMPORT_GROUP = "CBT"  # 要写入 JSON 的一级键
DEFAULT_INPUT_DIR = DEFAULT_OUTPUT_DIR  # 要读取的 txt 文件夹
DEFAULT_IMPORT_OVERWRITE_GROUP = True  # 若一级键已存在，是否整体覆盖
# ===== 默认参数结束 =====


def load_json(json_path: Path) -> dict[str, Any]:
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    data = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("JSON 根节点必须是对象")
    return data


def dump_json(json_path: Path, data: dict[str, Any]) -> None:
    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )


def sanitize_filename(name: str) -> str:
    invalid_chars = '<>:"/\\|?*'
    result = "".join("_" if ch in invalid_chars else ch for ch in name).strip()
    if not result:
        raise ValueError("键名无法转换为有效文件名")
    return result


def ensure_prompt_group(data: dict[str, Any], group: str) -> dict[str, str]:
    group_value = data.get(group)
    if group_value is None:
        raise KeyError(f"未找到一级字段: {group}")
    if not isinstance(group_value, dict):
        raise TypeError(f"一级字段 '{group}' 的值不是对象")

    for key, value in group_value.items():
        if not isinstance(value, str):
            raise TypeError(f"字段 '{group}.{key}' 的值不是字符串")

    return group_value


def export_group(json_path: Path, group: str, output_dir: Path, overwrite: bool) -> None:
    data = load_json(json_path)
    prompts = ensure_prompt_group(data, group)

    output_dir.mkdir(parents=True, exist_ok=True)

    used_names: dict[str, str] = {}
    exported = 0
    for key, text in prompts.items():
        filename = sanitize_filename(key)
        if filename in used_names and used_names[filename] != key:
            raise ValueError(
                f"键名 '{key}' 与 '{used_names[filename]}' 会生成同名文件 '{filename}.txt'"
            )
        used_names[filename] = key

        txt_path = output_dir / f"{filename}.txt"
        if txt_path.exists() and not overwrite:
            raise FileExistsError(
                f"目标文件已存在: {txt_path}。如需覆盖，请传入 --overwrite"
            )
        txt_path.write_text(text, encoding="utf-8")
        exported += 1

    print(f"已导出 {exported} 个 prompt 到: {output_dir}")


def import_group(json_path: Path, group: str, input_dir: Path, overwrite_group: bool) -> None:
    if not input_dir.exists():
        raise FileNotFoundError(f"输入文件夹不存在: {input_dir}")
    if not input_dir.is_dir():
        raise NotADirectoryError(f"输入路径不是文件夹: {input_dir}")

    txt_files = sorted(
        path for path in input_dir.iterdir() if path.is_file() and path.suffix.lower() == ".txt"
    )
    if not txt_files:
        raise FileNotFoundError(f"文件夹中未找到 txt 文件: {input_dir}")

    imported_prompts: dict[str, str] = {}
    for txt_file in txt_files:
        key = txt_file.stem
        if key in imported_prompts:
            raise ValueError(f"检测到重复键名: {key}")
        imported_prompts[key] = txt_file.read_text(encoding="utf-8")

    data = load_json(json_path)
    if group in data and not overwrite_group:
        existing = data[group]
        if not isinstance(existing, dict):
            raise TypeError(f"一级字段 '{group}' 的值不是对象，无法覆盖")
        raise ValueError(
            f"一级字段 '{group}' 已存在。若要整体替换，请传入 --overwrite-group"
        )

    data[group] = imported_prompts
    dump_json(json_path, data)
    print(f"已从 {input_dir} 导入 {len(imported_prompts)} 个 prompt 到: {json_path} -> {group}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="在 intervention_prompts.json 与一组 txt 文件之间双向转换"
    )
    subparsers = parser.add_subparsers(dest="mode")

    export_parser = subparsers.add_parser(
        "export",
        help="将 JSON 某个治疗类型下的所有 prompt 导出为 txt 文件",
    )
    export_parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON_PATH),
        help="JSON 文件路径",
    )
    export_parser.add_argument(
        "--group",
        default=DEFAULT_EXPORT_GROUP,
        help="要导出的一级字段名，例如 CBT",
    )
    export_parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="导出目录",
    )
    export_parser.add_argument(
        "--overwrite",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_EXPORT_OVERWRITE,
        help="是否覆盖已存在的 txt 文件",
    )

    import_parser = subparsers.add_parser(
        "import",
        help="将文件夹中的 txt 文件导入 JSON 的某个一级字段",
    )
    import_parser.add_argument(
        "--json",
        default=str(DEFAULT_JSON_PATH),
        help="JSON 文件路径",
    )
    import_parser.add_argument(
        "--group",
        default=DEFAULT_IMPORT_GROUP,
        help="要写入的一级字段名，例如 CBT 或 CBT_NEW",
    )
    import_parser.add_argument(
        "--input-dir",
        default=str(DEFAULT_INPUT_DIR),
        help="包含 txt 文件的文件夹路径",
    )
    import_parser.add_argument(
        "--overwrite-group",
        action=argparse.BooleanOptionalAction,
        default=DEFAULT_IMPORT_OVERWRITE_GROUP,
        help="一级字段已存在时是否整体替换",
    )

    return parser


def main() -> None:
    if DEFAULT_MODE not in {"export", "import"}:
        raise ValueError("DEFAULT_MODE 只能是 'export' 或 'import'")

    parser = build_parser()
    argv = sys.argv[1:] or [DEFAULT_MODE]
    args = parser.parse_args(argv)

    json_path = Path(args.json)

    if args.mode == "export":
        export_group(
            json_path=json_path,
            group=args.group,
            output_dir=Path(args.output_dir),
            overwrite=args.overwrite,
        )
        return

    if args.mode == "import":
        import_group(
            json_path=json_path,
            group=args.group,
            input_dir=Path(args.input_dir),
            overwrite_group=args.overwrite_group,
        )
        return

    parser.error("请指定模式 export 或 import")


if __name__ == "__main__":
    main()
