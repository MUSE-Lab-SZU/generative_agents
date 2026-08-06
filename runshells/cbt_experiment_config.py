#!/usr/bin/env python3
"""CBT experiment-entry configuration, identity, manifest, and resume helpers."""

from __future__ import annotations

import argparse
import copy
import json
import re
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from artifact_digest import canonical_json_sha256


CONTROLLERS = ("legacy", "minimal", "progressive")
PROGRESSIVE_STAGES = ("D",)
LEGACY_PROGRESSIVE_STAGE_FLAGS = ("A", "B", "C", "D")
PROGRESSIVE_D_CONTROLLER_VERSION = (
    "progressive_d_v3_1_calibrated_batch_control"
)
MANIFEST_SCHEMA_VERSION = 2
MANIFEST_FILENAME = "cbt_condition_manifest.json"
PROGRESSIVE_D_CAPABILITY_KEYS = (
    "batch_control",
    "progressive_d_tracker",
    "stage_adapter",
)

STAGE_FLAG_PATHS = {
    "A": ("intervention", "subgoal_shadow_enabled"),
    "B": ("intervention", "subgoal_progress_for_judge", "enabled"),
    "C": ("intervention", "limited_subgoal_transitions", "enabled"),
    "D": ("intervention", "legacy_stage_transition_adapter", "enabled"),
}
PROGRESSIVE_D_ENABLED_PATH = ("intervention", "progressive_d", "enabled")
PROGRESSIVE_D_VERSION_PATH = (
    "intervention",
    "progressive_d",
    "controller_version",
)


class CBTEntrypointConfigError(ValueError):
    """Raised when an experiment controller condition is unsafe or ambiguous."""


def deep_merge_dict(base: Mapping[str, Any], overlay: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in (overlay or {}).items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), Mapping):
            merged[key] = deep_merge_dict(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def normalize_controller_request(
    controller: str | None,
    progressive_stage: str | None,
) -> tuple[str, str | None]:
    normalized_controller = str(controller or "legacy").strip().lower()
    normalized_stage = str(progressive_stage or "").strip().upper() or None
    if normalized_controller not in CONTROLLERS:
        raise CBTEntrypointConfigError(
            "cbt controller 必须是 legacy、minimal 或 progressive"
        )
    if normalized_controller == "progressive":
        if normalized_stage is None:
            normalized_stage = "D"
        elif normalized_stage not in PROGRESSIVE_STAGES:
            raise CBTEntrypointConfigError(
                "progressive controller 只支持 D"
            )
    if normalized_controller != "progressive" and normalized_stage is not None:
        raise CBTEntrypointConfigError(
            f"{normalized_controller} controller 不能搭配 --progressive-stage"
        )
    return normalized_controller, normalized_stage


def controller_identity_label(controller: str, progressive_stage: str | None) -> str:
    controller, progressive_stage = normalize_controller_request(
        controller,
        progressive_stage,
    )
    if controller == "progressive":
        return f"progressive-{progressive_stage.lower()}"
    return controller


def slugify(value: str, *, field_name: str = "value") -> str:
    raw = str(value or "").strip()
    if not raw:
        return ""
    if "/" in raw or "\\" in raw or raw in {".", ".."}:
        raise CBTEntrypointConfigError(f"{field_name} 不能包含路径分隔符")
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", raw).strip("._-").lower()
    if not slug:
        raise CBTEntrypointConfigError(f"{field_name} 不能生成有效名称")
    return slug


def build_condition_key(
    condition_name: str,
    controller: str,
    progressive_stage: str | None,
    output_tag: str = "",
) -> str:
    identity = controller_identity_label(controller, progressive_stage)
    parts = [str(condition_name).strip(), f"cbt-{identity}"]
    tag = slugify(output_tag, field_name="output tag")
    if tag:
        parts.append(tag)
    return "--".join(parts)


def controller_identity_from_config(config: Mapping[str, Any]) -> dict[str, Any]:
    intervention = config.get("intervention", {})
    if not isinstance(intervention, Mapping):
        intervention = {}
    mode = str(intervention.get("cbt_controller_mode", "") or "").strip().lower()
    version = str(intervention.get("cbt_controller_version", "") or "").strip()
    flags = {
        stage: bool(_get_path(config, path))
        for stage, path in STAGE_FLAG_PATHS.items()
    }
    progressive_d_raw = intervention.get("progressive_d", {}) or {}
    native_d_enabled = bool(
        mode == "minimal"
        and isinstance(progressive_d_raw, Mapping)
        and progressive_d_raw.get("enabled") is True
    )
    progressive_stage = (
        "D"
        if native_d_enabled
        else (
            _progressive_stage_from_flags(flags)
            if mode == "minimal"
            else None
        )
    )
    kind = "progressive" if progressive_stage else mode
    label = (
        f"progressive-{progressive_stage.lower()}"
        if progressive_stage
        else mode
    )
    return {
        "controller_identity": label,
        "kind": kind,
        "mode": mode,
        "controller_version": version,
        "progressive_stage": progressive_stage,
        "capabilities": _progressive_d_capabilities(
            progressive_stage == "D"
        ),
        "stage_flags": flags,
    }


def _progressive_d_capabilities(enabled: bool) -> dict[str, bool]:
    return {
        key: bool(enabled)
        for key in PROGRESSIVE_D_CAPABILITY_KEYS
    }


def _progressive_stage_from_flags(flags: Mapping[str, bool]) -> str | None:
    enabled = [
        stage
        for stage in LEGACY_PROGRESSIVE_STAGE_FLAGS
        if bool(flags.get(stage))
    ]
    if not enabled:
        return None
    highest = enabled[-1]
    expected = list(
        LEGACY_PROGRESSIVE_STAGE_FLAGS[
            : LEGACY_PROGRESSIVE_STAGE_FLAGS.index(highest) + 1
        ]
    )
    if enabled != expected:
        return "INVALID"
    return highest


def _get_path(payload: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    current: Any = payload
    for key in path:
        if not isinstance(current, Mapping) or key not in current:
            return None
        current = current[key]
    return current


def _set_existing_path(
    payload: dict[str, Any],
    path: tuple[str, ...],
    value: Any,
) -> None:
    current: Any = payload
    for key in path[:-1]:
        if not isinstance(current, dict) or key not in current:
            raise CBTEntrypointConfigError(
                f"base/group config 缺少当前 CBT schema 字段: {'.'.join(path)}"
            )
        current = current[key]
    if not isinstance(current, dict) or path[-1] not in current:
        raise CBTEntrypointConfigError(
            f"base/group config 缺少当前 CBT schema 字段: {'.'.join(path)}"
        )
    current[path[-1]] = value


def apply_controller_override(
    merged_config: Mapping[str, Any],
    controller: str,
    progressive_stage: str | None,
) -> dict[str, Any]:
    controller, progressive_stage = normalize_controller_request(
        controller,
        progressive_stage,
    )
    resolved = copy.deepcopy(dict(merged_config))
    intervention = resolved.get("intervention")
    if not isinstance(intervention, dict):
        raise CBTEntrypointConfigError("config.intervention 必须是 JSON object")

    _set_existing_path(
        resolved,
        ("intervention", "cbt_controller_mode"),
        "legacy" if controller == "legacy" else "minimal",
    )
    _set_existing_path(
        resolved,
        PROGRESSIVE_D_ENABLED_PATH,
        controller == "progressive" and progressive_stage == "D",
    )
    for removed_key in (
        "subgoal_shadow_enabled",
        "subgoal_shadow",
        "subgoal_progress_for_judge",
        "limited_subgoal_transitions",
        "legacy_stage_transition_adapter",
    ):
        intervention.pop(removed_key, None)

    if controller in {"minimal", "progressive"}:
        _set_existing_path(
            resolved,
            ("intervention", "session_eval", "enabled"),
            not (
                controller == "progressive"
                and progressive_stage == "D"
            ),
        )
    if controller == "progressive" and progressive_stage == "D":
        _set_existing_path(
            resolved,
            ("intervention", "cbt_controller_version"),
            PROGRESSIVE_D_CONTROLLER_VERSION,
        )
        _set_existing_path(
            resolved,
            PROGRESSIVE_D_VERSION_PATH,
            PROGRESSIVE_D_CONTROLLER_VERSION,
        )
    return resolved


def resolve_controller_config(
    base_config: Mapping[str, Any],
    group_overlay: Mapping[str, Any],
    controller: str,
    progressive_stage: str | None,
    *,
    project_root: Path,
    available_agents: set[str] | None = None,
) -> dict[str, Any]:
    merged = deep_merge_dict(base_config, group_overlay)
    resolved = apply_controller_override(merged, controller, progressive_stage)
    validate_resolved_controller_config(
        resolved,
        controller,
        progressive_stage,
        project_root=project_root,
        available_agents=available_agents,
    )
    return resolved


def _require_enabled(
    config: Mapping[str, Any],
    path: tuple[str, ...],
    *,
    reason: str,
) -> None:
    if _get_path(config, path) is not True:
        raise CBTEntrypointConfigError(
            f"{reason}: {'.'.join(path)} 必须为 true"
        )


def _resolve_resource(project_root: Path, raw_path: Any, field: str) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        raise CBTEntrypointConfigError(f"资源路径为空: {field}")
    path = Path(text).expanduser()
    if not path.is_absolute():
        path = project_root / path
    path = path.resolve()
    if not path.is_file():
        raise CBTEntrypointConfigError(f"Prompt/map 文件不存在: {field}={path}")
    return path


def _validate_resource_paths(
    config: Mapping[str, Any],
    controller: str,
    progressive_stage: str | None,
    project_root: Path,
) -> None:
    native_progressive_d = bool(
        controller == "progressive"
        and progressive_stage == "D"
        and _get_path(config, PROGRESSIVE_D_ENABLED_PATH) is True
    )
    common_paths = [
        (
            "intervention.session_prompt_injection.prompt_file",
            ("intervention", "session_prompt_injection", "prompt_file"),
        ),
        (
            (
                "intervention.dialog_judge.legacy_prompt_file"
                if controller == "legacy"
                else (
                    "intervention.progressive_d.judge.prompt_file"
                    if native_progressive_d
                    else "intervention.dialog_judge.prompt_file"
                )
            ),
            (
                ("intervention", "dialog_judge", "legacy_prompt_file")
                if controller == "legacy"
                else (
                        (
                            "intervention",
                            "progressive_d",
                            "judge",
                            "prompt_file",
                        )
                    if native_progressive_d
                    else ("intervention", "dialog_judge", "prompt_file")
                )
            ),
        ),
    ]
    if not (controller == "progressive" and progressive_stage == "D"):
        common_paths.append(
            (
                (
                    "intervention.session_eval.legacy_prompt_file"
                    if controller == "legacy"
                    else "intervention.session_eval.prompt_file"
                ),
                (
                    ("intervention", "session_eval", "legacy_prompt_file")
                    if controller == "legacy"
                    else ("intervention", "session_eval", "prompt_file")
                ),
            )
        )
    if controller in {"minimal", "progressive"}:
        common_paths.extend(
            [
                (
                    (
                        "intervention.progressive_d.state_tracker.prompt_file"
                        if native_progressive_d
                        else "intervention.state_tracker.prompt_file"
                    ),
                    (
                            (
                                "intervention",
                                "progressive_d",
                                "state_tracker",
                                "prompt_file",
                            )
                        if native_progressive_d
                        else ("intervention", "state_tracker", "prompt_file")
                    ),
                ),
                (
                    "intervention.strategy_router.strategy_map_file",
                    ("intervention", "strategy_router", "strategy_map_file"),
                ),
                (
                    "intervention.strategy_router.term_glossary_file",
                    ("intervention", "strategy_router", "term_glossary_file"),
                ),
            ]
        )
    if native_progressive_d:
        common_paths.extend(
            [
                (
                    "intervention.progressive_d.control_eval.prompt_file",
                    (
                        "intervention",
                        "progressive_d",
                        "control_eval",
                        "prompt_file",
                    ),
                ),
                (
                    "intervention.progressive_d.control_eval.stage_subgoals_file",
                    (
                        "intervention",
                        "progressive_d",
                        "control_eval",
                        "stage_subgoals_file",
                    ),
                ),
            ]
        )
    elif controller == "progressive":
        common_paths.extend(
            [
                (
                    "intervention.subgoal_shadow.progressive_d_prompt_file",
                    (
                        "intervention",
                        "subgoal_shadow",
                        "progressive_d_prompt_file",
                    ),
                ),
                (
                    "intervention.subgoal_shadow.stage_subgoals_file",
                    ("intervention", "subgoal_shadow", "stage_subgoals_file"),
                ),
            ]
        )
    for field, path in common_paths:
        raw_path = _get_path(config, path)
        if (
            field == "intervention.strategy_router.term_glossary_file"
            and not str(raw_path or "").strip()
        ):
            raw_path = "data/intervention/cbt_term_glossary.json"
        resolved_path = _resolve_resource(project_root, raw_path, field)
        if resolved_path.suffix.lower() == ".json":
            try:
                with resolved_path.open("r", encoding="utf-8") as handle:
                    json.load(handle)
            except (OSError, json.JSONDecodeError) as exc:
                raise CBTEntrypointConfigError(
                    f"Prompt/map JSON 无效: {field}={resolved_path}: {exc}"
                ) from exc


def _validate_doctor_patient_condition(
    config: Mapping[str, Any],
    available_agents: set[str] | None,
) -> None:
    intervention = config["intervention"]
    doctor = str(intervention.get("doctor", "") or "").strip()
    patients_raw = intervention.get("patients", [])
    patients = [
        str(item or "").strip()
        for item in patients_raw
        if str(item or "").strip()
    ] if isinstance(patients_raw, list) else []
    if not doctor or not patients:
        raise CBTEntrypointConfigError(
            "CBT controller 需要非空 intervention.doctor 和 intervention.patients"
        )
    if doctor in patients:
        raise CBTEntrypointConfigError("intervention.doctor 不能同时作为 patient")
    if available_agents is not None:
        missing = sorted(({doctor, *patients}) - set(available_agents))
        if missing:
            raise CBTEntrypointConfigError(
                "doctor/patient 不在当前 condition agent roster: " + ", ".join(missing)
            )

    rules = intervention.get("meeting_rules", [])
    valid_rule = False
    if isinstance(rules, list):
        for rule in rules:
            if not isinstance(rule, Mapping) or rule.get("enabled") is not True:
                continue
            rule_doctor = str(rule.get("doctor", doctor) or "").strip()
            rule_patient = str(
                rule.get("patient", rule.get("target_patient", "")) or ""
            ).strip()
            if rule_doctor == doctor and rule_patient in patients:
                valid_rule = True
                break
    if not valid_rule:
        raise CBTEntrypointConfigError(
            "没有找到 doctor/patient 匹配且 enabled=true 的 intervention.meeting_rules"
        )


def validate_resolved_controller_config(
    config: Mapping[str, Any],
    controller: str,
    progressive_stage: str | None,
    *,
    project_root: Path,
    available_agents: set[str] | None = None,
) -> dict[str, Any]:
    controller, progressive_stage = normalize_controller_request(
        controller,
        progressive_stage,
    )
    intervention = config.get("intervention")
    if not isinstance(intervention, Mapping):
        raise CBTEntrypointConfigError("config.intervention 必须是 JSON object")
    identity = controller_identity_from_config(config)
    expected_mode = "legacy" if controller == "legacy" else "minimal"
    if identity["mode"] != expected_mode:
        raise CBTEntrypointConfigError(
            f"controller mode 不匹配: expected={expected_mode}, actual={identity['mode']}"
        )

    native_d_enabled = bool(
        controller == "progressive"
        and progressive_stage == "D"
        and _get_path(config, PROGRESSIVE_D_ENABLED_PATH) is True
    )
    expected_flags = {
        stage: bool(
            controller == "progressive"
            and progressive_stage == "D"
            and not native_d_enabled
        )
        for stage in LEGACY_PROGRESSIVE_STAGE_FLAGS
    }
    if identity["stage_flags"] != expected_flags:
        raise CBTEntrypointConfigError(
            "resolved stage flags 不匹配: "
            f"expected={expected_flags}, actual={identity['stage_flags']}"
        )
    for later_index, later_stage in enumerate(LEGACY_PROGRESSIVE_STAGE_FLAGS):
        if not identity["stage_flags"][later_stage]:
            continue
        missing = [
            stage
            for stage in LEGACY_PROGRESSIVE_STAGE_FLAGS[:later_index]
            if not identity["stage_flags"][stage]
        ]
        if missing:
            raise CBTEntrypointConfigError(
                f"Stage {later_stage} 缺少前置 Stage: {', '.join(missing)}"
            )

    version = identity["controller_version"]
    if not version:
        raise CBTEntrypointConfigError("intervention.cbt_controller_version 不能为空")

    if controller != "legacy":
        _require_enabled(
            config,
            ("intervention", "enabled"),
            reason="minimal/Progressive controller 不可用",
        )
        for path, label in (
            (("intervention", "session_prompt_injection", "enabled"), "Session Prompt"),
            (("intervention", "dialog_judge", "enabled"), "Judge"),
            (("intervention", "state_tracker", "enabled"), "Tracker"),
        ):
            _require_enabled(config, path, reason=f"{label} 不可用")
        if not (
            controller == "progressive"
            and progressive_stage == "D"
        ):
            _require_enabled(
                config,
                ("intervention", "session_eval", "enabled"),
                reason="Session Eval 不可用",
            )
        _validate_doctor_patient_condition(config, available_agents)

    if controller == "progressive" and progressive_stage == "D":
        if native_d_enabled:
            _require_enabled(
                config,
                PROGRESSIVE_D_ENABLED_PATH,
                reason="Progressive D 不可用",
            )
        else:
            for stage in LEGACY_PROGRESSIVE_STAGE_FLAGS:
                _require_enabled(
                    config,
                    STAGE_FLAG_PATHS[stage],
                    reason=f"旧 Progressive D 缺少 Stage {stage}",
                )
        policy_version = str(
            _get_path(
                config,
                (
                    PROGRESSIVE_D_VERSION_PATH
                    if native_d_enabled
                    else (
                        "intervention",
                        "legacy_stage_transition_adapter",
                        "controller_version",
                    )
                ),
            )
            or ""
        ).strip()
        if policy_version != version:
            raise CBTEntrypointConfigError(
                "Stage D controller version 不一致: "
                f"intervention.cbt_controller_version={version}, "
                f"policy.controller_version={policy_version or '(empty)'}"
            )
        if version != PROGRESSIVE_D_CONTROLLER_VERSION:
            raise CBTEntrypointConfigError(
                "Stage D 必须使用批量控制器版本: "
                f"expected={PROGRESSIVE_D_CONTROLLER_VERSION}, actual={version}"
            )

    _validate_resource_paths(
        config,
        controller,
        progressive_stage,
        project_root,
    )

    if controller == "progressive" and identity["progressive_stage"] != progressive_stage:
        raise CBTEntrypointConfigError(
            "表面 Progressive、实际非目标 Progressive 配置，拒绝启动"
        )
    return identity


def json_sha256(payload: Any) -> str:
    return canonical_json_sha256(payload)


def git_state(project_root: Path) -> dict[str, Any]:
    def run(*args: str) -> str:
        result = subprocess.run(
            ["git", *args],
            cwd=project_root,
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()

    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain")
    except (OSError, subprocess.CalledProcessError):
        return {"commit": "", "dirty": None}
    return {"commit": commit, "dirty": bool(status)}


def build_condition_manifest(
    *,
    project_root: Path,
    condition_name: str,
    condition_key: str,
    controller: str,
    progressive_stage: str | None,
    output_tag: str,
    base_config_path: Path,
    base_config: Mapping[str, Any],
    group_overlay_path: Path,
    group_overlay: Mapping[str, Any],
    resolved_config_path: Path,
    resolved_config: Mapping[str, Any],
    run_name: str,
    resume_requested: bool,
    resume_checkpoint_exists: bool,
    resume_snapshot: str,
) -> dict[str, Any]:
    identity = validate_resolved_controller_config(
        resolved_config,
        controller,
        progressive_stage,
        project_root=project_root,
        available_agents=(
            set(resolved_config.get("agents", {}))
            if isinstance(resolved_config.get("agents"), Mapping)
            and resolved_config.get("agents")
            else None
        ),
    )
    manifest_identity_payload = {
        key: copy.deepcopy(identity[key])
        for key in (
            "controller_identity",
            "kind",
            "mode",
            "controller_version",
            "progressive_stage",
            "capabilities",
        )
    }
    return {
        "schema_version": MANIFEST_SCHEMA_VERSION,
        "artifact_kind": "cbt_experiment_condition",
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        **manifest_identity_payload,
        "condition_name": condition_name,
        "condition_key": condition_key,
        "output_tag": str(output_tag or "").strip(),
        "run_name": run_name,
        "git": git_state(project_root),
        "base_config": {
            "path": _display_path(base_config_path, project_root),
            "sha256": json_sha256(base_config),
        },
        "group_overlay": {
            "path": _display_path(group_overlay_path, project_root),
            "sha256": json_sha256(group_overlay),
            "config": copy.deepcopy(dict(group_overlay)),
        },
        "resolved_config": {
            "path": _display_path(resolved_config_path, project_root),
            "sha256": json_sha256(resolved_config),
            "config": copy.deepcopy(dict(resolved_config)),
        },
        "resume": {
            "requested": bool(resume_requested),
            "checkpoint_exists": bool(resume_checkpoint_exists),
            "snapshot": str(resume_snapshot or ""),
            "validated": bool(resume_requested and resume_checkpoint_exists),
        },
    }


def _display_path(path: Path, project_root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(project_root.resolve()))
    except ValueError:
        return str(resolved)


def manifest_identity(manifest: Mapping[str, Any]) -> dict[str, Any]:
    controller_identity = str(
        manifest.get("controller_identity", "") or ""
    )
    mode = str(manifest.get("mode", "") or "")
    controller_version = str(
        manifest.get("controller_version", "") or ""
    )
    progressive_stage = manifest.get("progressive_stage")
    schema_raw = manifest.get("schema_version")
    try:
        schema_version = int(schema_raw) if schema_raw is not None else None
    except (TypeError, ValueError) as exc:
        raise CBTEntrypointConfigError(
            f"controller manifest schema_version 无效: {schema_raw!r}"
        ) from exc

    if schema_version not in {None, 1, MANIFEST_SCHEMA_VERSION}:
        raise CBTEntrypointConfigError(
            f"不支持的 controller manifest schema_version: {schema_version}"
        )

    capabilities_raw = manifest.get("capabilities")
    if schema_version == MANIFEST_SCHEMA_VERSION or isinstance(
        capabilities_raw,
        Mapping,
    ):
        capabilities = {
            key: bool((capabilities_raw or {}).get(key))
            for key in PROGRESSIVE_D_CAPABILITY_KEYS
        }
    else:
        flags_raw = manifest.get("stage_flags", {}) or {}
        if not isinstance(flags_raw, Mapping):
            flags_raw = {}
        flags = {
            stage: bool(flags_raw.get(stage))
            for stage in LEGACY_PROGRESSIVE_STAGE_FLAGS
        }
        is_progressive_identity = bool(
            controller_identity.startswith("progressive-")
            or progressive_stage is not None
        )
        if is_progressive_identity:
            expected_d_flags = {
                stage: True for stage in LEGACY_PROGRESSIVE_STAGE_FLAGS
            }
            if (
                controller_identity != "progressive-d"
                or progressive_stage != "D"
                or flags != expected_d_flags
            ):
                raise CBTEntrypointConfigError(
                    "旧 controller manifest 仅支持读取完整的 Progressive D 身份；"
                    "Progressive A/B/C 已移除"
                )
            capabilities = _progressive_d_capabilities(True)
        else:
            if any(flags.values()):
                raise CBTEntrypointConfigError(
                    "非 Progressive controller manifest 不得启用旧 stage_flags"
                )
            capabilities = _progressive_d_capabilities(False)

    is_progressive_d = controller_identity == "progressive-d"
    if is_progressive_d != (progressive_stage == "D"):
        raise CBTEntrypointConfigError(
            "Progressive D controller identity 与 progressive_stage 不一致"
        )
    expected_capabilities = _progressive_d_capabilities(is_progressive_d)
    if capabilities != expected_capabilities:
        raise CBTEntrypointConfigError(
            "Progressive D controller capabilities 不完整或与身份冲突"
        )
    if controller_identity.startswith("progressive-") and not is_progressive_d:
        raise CBTEntrypointConfigError("Progressive A/B/C 已移除")

    return {
        "controller_identity": controller_identity,
        "mode": mode,
        "controller_version": controller_version,
        "progressive_stage": progressive_stage,
        "capabilities": capabilities,
    }


def assert_identity_matches(
    expected: Mapping[str, Any],
    actual: Mapping[str, Any],
    *,
    context: str,
) -> None:
    expected_identity = manifest_identity(expected)
    actual_identity = manifest_identity(actual)
    mismatches = {
        key: {"expected": expected_identity[key], "actual": actual_identity[key]}
        for key in expected_identity
        if expected_identity[key] != actual_identity[key]
    }
    if mismatches:
        raise CBTEntrypointConfigError(
            f"{context} controller identity 不匹配: "
            + json.dumps(mismatches, ensure_ascii=False, sort_keys=True)
        )


def assert_checkpoint_identity(
    expected_manifest: Mapping[str, Any],
    snapshot_config: Mapping[str, Any],
    *,
    context: str,
) -> None:
    actual = controller_identity_from_config(snapshot_config)
    assert_identity_matches(expected_manifest, actual, context=context)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="CBT experiment-entry helper")
    subparsers = parser.add_subparsers(dest="command", required=True)
    key_parser = subparsers.add_parser("condition-key")
    key_parser.add_argument("--condition", required=True)
    key_parser.add_argument("--cbt-controller", default="legacy", choices=CONTROLLERS)
    key_parser.add_argument("--progressive-stage", choices=PROGRESSIVE_STAGES)
    key_parser.add_argument("--output-tag", default="")
    args = parser.parse_args()
    if args.command == "condition-key":
        try:
            print(
                build_condition_key(
                    args.condition,
                    args.cbt_controller,
                    args.progressive_stage,
                    args.output_tag,
                )
            )
        except CBTEntrypointConfigError as exc:
            parser.error(str(exc))


if __name__ == "__main__":
    _cli()
