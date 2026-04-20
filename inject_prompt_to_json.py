#!/usr/bin/env python3
"""
将 txt 文本注入到 JSON 的指定字段。

示例：
python inject_prompt_to_json.py \
  --txt ./xxx.txt \
  --json ./data/prompts/intervention_prompts.json \
  --key CBT \
  --key session001

以上命令会把 xxx.txt 的内容写入 intervention_prompts.json 的 ["CBT"]["session001"]。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

DEFAULT_TXT_PATH = "./CBT_session.txt"
DEFAULT_JSON_PATH = "./data/prompts/intervention_prompts.json"
DEFAULT_KEY_PATH = ["CBT", "session4.3"]


def set_nested_value(data: dict[str, Any], keys: list[str], value: str) -> None:
    """按 keys 路径设置嵌套值，不存在的中间层会自动创建为 dict。"""
    if not keys:
        raise ValueError("keys 不能为空")

    node: dict[str, Any] = data
    for key in keys[:-1]:
        cur = node.get(key)
        if cur is None:
            node[key] = {}
        elif not isinstance(cur, dict):
            raise TypeError(f"路径冲突：键 '{key}' 对应的值不是对象，无法继续向下写入")
        node = node[key]

    node[keys[-1]] = value


def main() -> None:
    parser = argparse.ArgumentParser(description="将 txt 内容写入 JSON 的指定字段")
    parser.add_argument(
        "--txt",
        default=DEFAULT_TXT_PATH,
        help=f"源 txt 文件路径（默认: {DEFAULT_TXT_PATH}）。用于读取待写入 JSON 的完整文本。",
    )
    parser.add_argument(
        "--json",
        default=DEFAULT_JSON_PATH,
        help=f"目标 JSON 文件路径（默认: {DEFAULT_JSON_PATH}）。用于定位要更新的 JSON 文件。",
    )
    parser.add_argument(
        "--key",
        action="append",
        default=None,
        help=(
            f"目标字段路径，可重复传入；不传时默认等效为："
            f"--key {DEFAULT_KEY_PATH[0]} --key {DEFAULT_KEY_PATH[1]}。"
            "用于按层级写入 JSON。"
        ),
    )

    args = parser.parse_args()

    txt_path = Path(args.txt)
    json_path = Path(args.json)
    keys: list[str] = args.key if args.key else DEFAULT_KEY_PATH.copy()

    if not txt_path.exists():
        raise FileNotFoundError(f"txt 文件不存在: {txt_path}")

    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    txt_content = txt_path.read_text(encoding="utf-8")
    data = json.loads(json_path.read_text(encoding="utf-8"))

    if not isinstance(data, dict):
        raise TypeError("JSON 根节点必须是对象")

    set_nested_value(data, keys, txt_content)

    json_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=4) + "\n",
        encoding="utf-8",
    )

    print(f"已写入: {json_path} -> {keys}")


if __name__ == "__main__":
    main()
