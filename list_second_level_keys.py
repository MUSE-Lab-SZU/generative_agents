#!/usr/bin/env python3
"""
将 JSON 指定一级字段下的二级字段名提取为列表并输出。

示例：
python list_second_level_keys.py --json ./data/prompts/intervention_prompts.json --top CBT

输出示例：
["session001", "session002"]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="提取指定一级字段下的二级字段名列表")
    parser.add_argument(
        "--json",
        default="./data/prompts/intervention_prompts.json",
        help="JSON 文件路径（默认: ./data/prompts/intervention_prompts.json）",
    )
    parser.add_argument(
        "--top",
        default="CBT",
        help="一级字段名（默认: CBT）",
    )
    args = parser.parse_args()

    json_path = Path(args.json)
    if not json_path.exists():
        raise FileNotFoundError(f"JSON 文件不存在: {json_path}")

    data: Any = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TypeError("JSON 根节点必须是对象")

    top_value = data.get(args.top)
    if top_value is None:
        raise KeyError(f"未找到一级字段: {args.top}")
    if not isinstance(top_value, dict):
        raise TypeError(f"一级字段 '{args.top}' 的值不是对象")

    second_level_keys = list(top_value.keys())
    print(json.dumps(second_level_keys, ensure_ascii=False))


if __name__ == "__main__":
    main()
