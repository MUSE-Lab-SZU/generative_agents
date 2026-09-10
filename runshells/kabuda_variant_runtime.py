#!/usr/bin/env python3
"""Prepare normalized runtime files for replaceable patient personas."""

from __future__ import annotations

import argparse
import copy
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any


RUNTIME_AGENT_NAME = "卡布达"
KABUDA_VARIANTS = ["kbd1", "kbd2", "kbd3", "kbd4", "kbd5", "kbd6", "kbd7", "kbd8", "kbd9"]
TOWN_PERSONA_VARIANTS = ["lrn", "gc", "cy", "tw", "zyh", "sql", "zmy", "xfh"]
VARIANTS = [*KABUDA_VARIANTS, *TOWN_PERSONA_VARIANTS]
VARIANT_SOURCE_AGENT_NAMES = {
    "kbd1": "卡布达",
    "kbd2": "卡布达2",
    "kbd3": "卡布达3",
    "kbd4": "卡布达4",
    "kbd5": "卡布达5",
    "kbd6": "卡布达6",
    "kbd7": "卡布达7",
    "kbd8": "卡布达8",
    "kbd9": "卡布达9",
    "lrn": "林若宁",
    "gc": "顾晨",
    "cy": "陈屿",
    "tw": "唐婉",
    "zyh": "周远航",
    "sql": "苏晴岚",
    "zmy": "赵明远",
    "xfh": "许芳华",
}
VARIANT_SHORT_NAMES = {
    "kbd1": "KBD1",
    "kbd2": "KBD2",
    "kbd3": "KBD3",
    "kbd4": "KBD4",
    "kbd5": "KBD5",
    "kbd6": "KBD6",
    "kbd7": "KBD7",
    "kbd8": "KBD8",
    "kbd9": "KBD9",
    "lrn": "LRN",
    "gc": "GC",
    "cy": "CY",
    "tw": "TW",
    "zyh": "ZYH",
    "sql": "SQL",
    "zmy": "ZMY",
    "xfh": "XFH",
}
VARIANT_SELECTOR_ALIASES = {
    "KBD1": "kbd1",
    "KABUDA": "kbd1",
    "KABUDA1": "kbd1",
    "ORIGINAL": "kbd1",
    "BASE": "kbd1",
    "KBD2": "kbd2",
    "KABUDA2": "kbd2",
    "KBD3": "kbd3",
    "KABUDA3": "kbd3",
    "KBD4": "kbd4",
    "KABUDA4": "kbd4",
    "KBD5": "kbd5",
    "KABUDA5": "kbd5",
    "KBD6": "kbd6",
    "KABUDA6": "kbd6",
    "KBD7": "kbd7",
    "KABUDA7": "kbd7",
    "KBD8": "kbd8",
    "KABUDA8": "kbd8",
    "KBD9": "kbd9",
    "KABUDA9": "kbd9",
    "LRN": "lrn",
    "GC": "gc",
    "CY": "cy",
    "TW": "tw",
    "ZYH": "zyh",
    "SQL": "sql",
    "ZMY": "zmy",
    "XFH": "xfh",
}
MEMORY_INJECTION_CONFIG_PATHS = {
    "kbd1": "data/intervention/memory_injections.json",
    "kbd2": "data/intervention/memory_injections_kabuda2.json",
    "kbd3": "data/intervention/memory_injections_kabuda3.json",
    "kbd4": "data/intervention/memory_injections_kabuda4.json",
    "kbd5": "data/intervention/memory_injections_kabuda5.json",
    "kbd6": "data/intervention/memory_injections_kabuda6.json",
    "kbd7": "data/intervention/memory_injections_kabuda7.json",
    "kbd8": "data/intervention/memory_injections_kabuda8.json",
    "kbd9": "data/intervention/memory_injections_kabuda9.json",
    "lrn": "data/intervention/memory_injections_lrn.json",
    "gc": "data/intervention/memory_injections_gc.json",
    "cy": "data/intervention/memory_injections_cy.json",
    "tw": "data/intervention/memory_injections_tw.json",
    "zyh": "data/intervention/memory_injections_zyh.json",
    "sql": "data/intervention/memory_injections_sql.json",
    "zmy": "data/intervention/memory_injections_zmy.json",
    "xfh": "data/intervention/memory_injections_xfh.json",
}
SEVERITY_CONFIG_NAMES = {
    "mild": "depression_config_mild.json",
    "moderate": "depression_config_moderate.json",
    "severe": "depression_config_severe.json",
}
VARIANT_SUPPORTED_SEVERITIES = {
    variant: set(SEVERITY_CONFIG_NAMES) for variant in KABUDA_VARIANTS
}
VARIANT_SUPPORTED_SEVERITIES.update(
    {variant: {"moderate"} for variant in TOWN_PERSONA_VARIANTS}
)


@dataclass(frozen=True)
class KabudaVariantRuntime:
    variant: str
    variant_short_name: str
    source_agent_name: str
    runtime_agent_name: str
    source_agent_dir: Path
    agent_config_path: Path
    depression_config_path: Path
    memory_injection_config_path: str


def resolve_variant(value: str) -> str:
    normalized = str(value or "").strip()
    if normalized in VARIANTS:
        return normalized
    upper = normalized.upper()
    if upper in VARIANT_SELECTOR_ALIASES:
        return VARIANT_SELECTOR_ALIASES[upper]
    raise ValueError(f"unknown patient persona variant: {value}")


def runtime_agent_name_for_variant(value: str) -> str:
    resolved = resolve_variant(value)
    if resolved in TOWN_PERSONA_VARIANTS:
        return VARIANT_SOURCE_AGENT_NAMES[resolved]
    return RUNTIME_AGENT_NAME


def load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON payload must be an object: {path}")
    return payload


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
        f.write("\n")
    os.replace(tmp_path, path)


RESOURCE_PATH_KEYS = {"portrait", "texture"}


def normalize_runtime_agent_name(payload: Any, *, source_name: str, runtime_name: str) -> Any:
    if isinstance(payload, dict):
        normalized = {}
        for key, value in payload.items():
            if key in RESOURCE_PATH_KEYS:
                normalized[key] = value
            else:
                normalized[key] = normalize_runtime_agent_name(
                    value,
                    source_name=source_name,
                    runtime_name=runtime_name,
                )
        return normalized
    if isinstance(payload, list):
        return [
            normalize_runtime_agent_name(item, source_name=source_name, runtime_name=runtime_name)
            for item in payload
        ]
    if isinstance(payload, str) and source_name != runtime_name:
        return payload.replace(source_name, runtime_name)
    return payload


def prepare_kabuda_variant_runtime(
    *,
    base_dir: Path,
    variant: str,
    severity: str,
    output_dir: Path,
    dry_run: bool = False,
    runtime_agent_name: str | None = None,
    agent_source_subdir: str = "village",
) -> KabudaVariantRuntime:
    resolved_variant = resolve_variant(variant)
    if severity not in SEVERITY_CONFIG_NAMES:
        raise ValueError(f"unknown severity: {severity}")
    if severity not in VARIANT_SUPPORTED_SEVERITIES[resolved_variant]:
        raise ValueError(
            f"persona {VARIANT_SHORT_NAMES[resolved_variant]} only supports severity: moderate"
        )
    if agent_source_subdir not in {"village", "counsel_room"}:
        raise ValueError(f"unknown agent source subdir: {agent_source_subdir}")
    source_agent_name = VARIANT_SOURCE_AGENT_NAMES[resolved_variant]
    runtime_agent_name = (
        str(runtime_agent_name or "").strip()
        or runtime_agent_name_for_variant(resolved_variant)
    )
    source_memory_path = Path(base_dir) / MEMORY_INJECTION_CONFIG_PATHS[resolved_variant]
    village_agent_dir = (
        Path(base_dir)
        / "frontend"
        / "static"
        / "assets"
        / "village"
        / "agents"
        / source_agent_name
    )
    source_depression_path = village_agent_dir / SEVERITY_CONFIG_NAMES[severity]

    # 咨询室的 kbd1 使用专用角色配置；其余卡布达变体和新人设复用村庄人设，
    # 再叠加卡布达的咨询室坐标、空间树与当前场景。严重度配置始终来自村庄变体目录。
    counsel_spatial_template: Path | None = None
    if agent_source_subdir == "counsel_room":
        counsel_agent_path = (
            Path(base_dir)
            / "frontend"
            / "static"
            / "assets"
            / "counsel_room"
            / "agents"
            / source_agent_name
            / "agent.json"
        )
        if counsel_agent_path.is_file():
            source_agent_path = counsel_agent_path
            source_agent_dir = counsel_agent_path.parent
        else:
            source_agent_path = village_agent_dir / "agent.json"
            source_agent_dir = village_agent_dir
            counsel_spatial_template = (
                Path(base_dir)
                / "frontend"
                / "static"
                / "assets"
                / "counsel_room"
                / "agents"
                / RUNTIME_AGENT_NAME
                / "agent.json"
            )
    else:
        source_agent_path = village_agent_dir / "agent.json"
        source_agent_dir = village_agent_dir
    if not source_agent_path.is_file():
        raise FileNotFoundError(source_agent_path)
    if not source_depression_path.is_file():
        raise FileNotFoundError(source_depression_path)
    if not source_memory_path.is_file():
        raise FileNotFoundError(source_memory_path)

    agent_payload = normalize_runtime_agent_name(
        load_json_file(source_agent_path),
        source_name=source_agent_name,
        runtime_name=runtime_agent_name,
    )
    if counsel_spatial_template is not None:
        if not counsel_spatial_template.is_file():
            raise FileNotFoundError(counsel_spatial_template)
        template = load_json_file(counsel_spatial_template)
        for field in ("spatial", "coord", "currently"):
            if field in template:
                agent_payload[field] = copy.deepcopy(template[field])
    depression_payload = normalize_runtime_agent_name(
        load_json_file(source_depression_path),
        source_name=source_agent_name,
        runtime_name=runtime_agent_name,
    )
    memory_injection_payload = normalize_runtime_agent_name(
        load_json_file(source_memory_path),
        source_name=source_agent_name,
        runtime_name=runtime_agent_name,
    )
    agent_payload["name"] = runtime_agent_name
    depression_payload["name"] = runtime_agent_name

    output_dir = Path(output_dir)
    agent_output_path = output_dir / "agent.json"
    depression_output_path = output_dir / SEVERITY_CONFIG_NAMES[severity]
    memory_injection_output_path = output_dir / "memory_injections.json"
    if not dry_run:
        atomic_write_json(agent_output_path, agent_payload)
        atomic_write_json(depression_output_path, depression_payload)
        atomic_write_json(memory_injection_output_path, memory_injection_payload)

    return KabudaVariantRuntime(
        variant=resolved_variant,
        variant_short_name=VARIANT_SHORT_NAMES[resolved_variant],
        source_agent_name=source_agent_name,
        runtime_agent_name=runtime_agent_name,
        source_agent_dir=source_agent_dir,
        agent_config_path=agent_output_path,
        depression_config_path=depression_output_path,
        memory_injection_config_path=str(memory_injection_output_path),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare normalized runtime files for a patient persona")
    parser.add_argument(
        "--variant",
        required=True,
        help="KBD1..KBD9 or LRN/GC/CY/TW/ZYH/SQL/ZMY/XFH",
    )
    parser.add_argument("--severity", required=True, choices=sorted(SEVERITY_CONFIG_NAMES), help="mild/moderate/severe")
    parser.add_argument("--output-dir", required=True, help="Directory to write normalized runtime files")
    parser.add_argument("--base-dir", default=str(Path(__file__).resolve().parents[1]), help="Repository root")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print paths without writing files")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    runtime = prepare_kabuda_variant_runtime(
        base_dir=Path(args.base_dir),
        variant=args.variant,
        severity=args.severity,
        output_dir=Path(args.output_dir),
        dry_run=bool(args.dry_run),
    )
    print(json.dumps(
        {
            "variant": runtime.variant,
            "variant_short_name": runtime.variant_short_name,
            "source_agent_name": runtime.source_agent_name,
            "source_agent_dir": str(runtime.source_agent_dir),
            "agent_config_path": str(runtime.agent_config_path),
            "depression_config_path": str(runtime.depression_config_path),
            "memory_injection_config_path": runtime.memory_injection_config_path,
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
