"""外置记忆检索调试脚本：按存档与角色/Agent 直连 retrieve，并输出 Markdown 报告。

使用方式：
1) 直接编辑本文件顶部“配置区”的参数；
2) 直接运行：python3 test/live_external_memory_retrieve_dump.py；
3) 如需临时覆盖顶部配置，也可以继续传命令行参数。
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

CURRENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = CURRENT_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient

CHECKPOINTS_ROOT = REPO_ROOT / "results" / "checkpoints"
GLOBAL_CONFIG_PATH = REPO_ROOT / "data" / "config.json"


# =========================
# 顶部配置区（可直接编辑）
# =========================
SAVE_NAME = "sim-test-0515"
# 必填。目标存档名，例如：sim-test-0513

SELECTOR = "卡布达"
# 必填。支持以下形式：
# - 直接 Agent 名：卡布达 / 蜻蜓队长
# - agent:<name>
# - doctor / doctor:<name>
# - patient / patient:<name>

QUERY = "最近跟心理医生的对话对你有帮助吗？上周你们聊了什么？"
# 必填。直接传给外置记忆服务 retrieve 接口的 query 文本。

BASE_URL = "http://localhost:8031"
# 外置记忆服务地址。

TIMEOUT_SECONDS = 15.0
# HTTP 请求超时时间（秒）。

OUTPUT_PATH = "test/external_memory_retrieve_report/0515_test_1.md"
# 可选。Markdown 输出路径。
# 留空时，自动写到 results/checkpoints/<save_name>/ 目录下。
# 也支持相对路径，例如：results/tmp/my_report.md

SNAPSHOT_PATH = ""
# 可选。手动指定某个 snapshot JSON 路径。
# 留空时，自动选取该 save 目录下最新的 simulate-*.json。

ACTIVE_PROJECT = ""
# 可选。透传给 retrieve 接口的 active_project 字段。
# 如果不需要，保持空字符串即可。
# =========================


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="按存档与角色/Agent 调用外置记忆 retrieve，并导出 Markdown 调试报告。"
    )
    parser.add_argument(
        "--save-name",
        default=None,
        help="存档名；不传时使用脚本顶部 SAVE_NAME",
    )
    parser.add_argument(
        "--selector",
        default=None,
        help="Agent 名或别名；不传时使用脚本顶部 SELECTOR",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="retrieve 接口 query；不传时使用脚本顶部 QUERY",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="外置记忆服务 base URL；不传时使用脚本顶部 BASE_URL",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="HTTP 超时秒数；不传时使用脚本顶部 TIMEOUT_SECONDS",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Markdown 输出路径；不传时使用脚本顶部 OUTPUT_PATH",
    )
    parser.add_argument(
        "--snapshot-path",
        default=None,
        help="手动指定 snapshot JSON 路径；不传时使用脚本顶部 SNAPSHOT_PATH",
    )
    parser.add_argument(
        "--active-project",
        default=None,
        help="透传给 retrieve 的 active_project；不传时使用脚本顶部 ACTIVE_PROJECT",
    )
    return parser.parse_args()


def resolve_runtime_options(args: argparse.Namespace) -> Dict[str, Any]:
    save_name = str(args.save_name if args.save_name is not None else SAVE_NAME).strip()
    selector = str(args.selector if args.selector is not None else SELECTOR).strip()
    query = str(args.query if args.query is not None else QUERY).strip()
    base_url = str(args.base_url if args.base_url is not None else BASE_URL).strip()
    timeout = float(args.timeout if args.timeout is not None else TIMEOUT_SECONDS)
    output = str(args.output if args.output is not None else OUTPUT_PATH).strip()
    snapshot_path = str(
        args.snapshot_path if args.snapshot_path is not None else SNAPSHOT_PATH
    ).strip()
    active_project = str(
        args.active_project if args.active_project is not None else ACTIVE_PROJECT
    ).strip()

    if not save_name:
        raise ValueError("save_name 为空；请修改脚本顶部 SAVE_NAME 或通过 --save-name 传入")
    if not selector:
        raise ValueError("selector 为空；请修改脚本顶部 SELECTOR 或通过 --selector 传入")
    if not query:
        raise ValueError("query 为空；请修改脚本顶部 QUERY 或通过 --query 传入")
    if not base_url:
        raise ValueError("base_url 为空；请修改脚本顶部 BASE_URL 或通过 --base-url 传入")
    if timeout <= 0:
        raise ValueError("timeout 必须大于 0")

    return {
        "save_name": save_name,
        "selector": selector,
        "query": query,
        "base_url": base_url,
        "timeout": timeout,
        "output": output,
        "snapshot_path": snapshot_path,
        "active_project": active_project,
    }


def load_json_file(path: Path) -> Dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    errors: List[str] = []
    for encoding in ("utf-8", "utf-8-sig"):
        try:
            payload = json.loads(path.read_text(encoding=encoding))
            if isinstance(payload, dict):
                return payload
            return {"_payload": payload}
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{encoding}: {exc}")
    raise ValueError(f"Failed to parse JSON: {path}\n" + "\n".join(errors))


def resolve_save_dir(save_name: str) -> Path:
    save_dir = CHECKPOINTS_ROOT / str(save_name).strip()
    if not save_dir.is_dir():
        raise FileNotFoundError(f"Save directory not found: {save_dir}")
    return save_dir


def resolve_snapshot_path(save_dir: Path, snapshot_override: str) -> Path:
    if str(snapshot_override or "").strip():
        path = Path(str(snapshot_override).strip())
        if not path.is_absolute():
            path = REPO_ROOT / path
        if not path.is_file():
            raise FileNotFoundError(f"Snapshot file not found: {path}")
        return path
    snapshots = sorted(save_dir.glob("simulate-*.json"))
    if not snapshots:
        raise FileNotFoundError(f"No simulate-*.json found under: {save_dir}")
    return snapshots[-1]


def resolve_agent_config_path(raw_path: str, agent_name: str) -> Optional[Path]:
    path_text = str(raw_path or "").strip()
    candidates: List[Path] = []
    if path_text:
        raw = Path(path_text)
        if raw.is_absolute():
            candidates.append(raw)
        else:
            candidates.append(REPO_ROOT / raw)
            candidates.append(REPO_ROOT / "frontend" / "static" / raw)
    candidates.append(
        REPO_ROOT / "frontend" / "static" / "assets" / "village" / agent_name / "agent.json"
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def read_global_config() -> Dict[str, Any]:
    return load_json_file(GLOBAL_CONFIG_PATH)


def get_snapshot_agents(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    agents = snapshot.get("agents", {})
    if not isinstance(agents, dict):
        raise ValueError("snapshot['agents'] is not a dict")
    return agents


def resolve_selector(
    selector: str,
    snapshot: Dict[str, Any],
    global_config: Dict[str, Any],
) -> Tuple[str, str, List[str]]:
    agents = get_snapshot_agents(snapshot)
    selector_text = str(selector or "").strip()
    selector_lower = selector_text.lower()
    notes: List[str] = []

    intervention_cfg = global_config.get("intervention", {})
    if not isinstance(intervention_cfg, dict):
        intervention_cfg = {}
    configured_doctor = str(intervention_cfg.get("doctor", "") or "").strip()
    configured_patients = [
        str(item or "").strip()
        for item in (intervention_cfg.get("patients", []) or [])
        if str(item or "").strip()
    ]

    def ensure_in_snapshot(name: str) -> str:
        if name not in agents:
            raise ValueError(f"Resolved agent not found in snapshot: {name}")
        return name

    if selector_text in agents:
        notes.append("selector matched snapshot agent name directly")
        return selector_text, "agent", notes

    if selector_lower.startswith("agent:"):
        name = selector_text.split(":", 1)[1].strip()
        if not name:
            raise ValueError("selector 'agent:' is missing an agent name")
        ensure_in_snapshot(name)
        notes.append("selector used explicit agent:<name> form")
        return name, "agent", notes

    if selector_lower == "doctor":
        if not configured_doctor:
            raise ValueError("data/config.json does not define intervention.doctor")
        ensure_in_snapshot(configured_doctor)
        notes.append("selector 'doctor' resolved via data/config.json intervention.doctor")
        return configured_doctor, "doctor", notes

    if selector_lower.startswith("doctor:"):
        name = selector_text.split(":", 1)[1].strip()
        if not name:
            raise ValueError("selector 'doctor:' is missing a doctor name")
        if configured_doctor and name != configured_doctor:
            raise ValueError(
                f"selector doctor:{name} does not match configured doctor {configured_doctor}"
            )
        ensure_in_snapshot(name)
        notes.append("selector used explicit doctor:<name> form")
        return name, "doctor", notes

    if selector_lower == "patient":
        if not configured_patients:
            raise ValueError("data/config.json does not define intervention.patients")
        if len(configured_patients) != 1:
            raise ValueError(
                "selector 'patient' is ambiguous because multiple patients are configured; "
                "please use patient:<name> or a direct agent name"
            )
        ensure_in_snapshot(configured_patients[0])
        notes.append("selector 'patient' resolved via single configured patient")
        return configured_patients[0], "patient", notes

    if selector_lower.startswith("patient:"):
        name = selector_text.split(":", 1)[1].strip()
        if not name:
            raise ValueError("selector 'patient:' is missing a patient name")
        if configured_patients and name not in configured_patients:
            raise ValueError(
                f"selector patient:{name} is not listed in configured patients {configured_patients}"
            )
        ensure_in_snapshot(name)
        notes.append("selector used explicit patient:<name> form")
        return name, "patient", notes

    raise ValueError(
        "Unable to resolve selector. Use an agent name, agent:<name>, doctor, doctor:<name>, patient, or patient:<name>."
    )


def resolve_user_id(
    snapshot: Dict[str, Any],
    agent_name: str,
    global_config: Dict[str, Any],
) -> Tuple[str, str, Optional[Path], List[str]]:
    agents = get_snapshot_agents(snapshot)
    agent_payload = agents.get(agent_name, {})
    if not isinstance(agent_payload, dict):
        agent_payload = {}

    notes: List[str] = []
    raw_config_path = str(agent_payload.get("config_path", "") or "")
    config_path = resolve_agent_config_path(raw_config_path, agent_name)
    if config_path is not None:
        notes.append(f"agent config path resolved to {config_path}")
        try:
            agent_cfg = load_json_file(config_path)
            ext_cfg = agent_cfg.get("external_memory", {})
            if isinstance(ext_cfg, dict):
                uid = str(ext_cfg.get("user_id", "") or "").strip()
                if uid:
                    notes.append("raw user_id loaded from agent config external_memory.user_id")
                    return uid, "agent_config", config_path, notes
        except Exception as exc:  # noqa: BLE001
            notes.append(f"failed to read agent config user_id: {exc}")
    else:
        notes.append("agent config path could not be resolved")

    agent_global_cfg = global_config.get("agent", {})
    if not isinstance(agent_global_cfg, dict):
        agent_global_cfg = {}
    global_ext_cfg = agent_global_cfg.get("external_memory", {})
    if not isinstance(global_ext_cfg, dict):
        global_ext_cfg = {}
    global_uid = str(global_ext_cfg.get("user_id", "") or "").strip()
    if global_uid:
        notes.append("raw user_id fell back to data/config.json agent.external_memory.user_id")
        return global_uid, "global_config", config_path, notes

    raise ValueError(f"No external_memory.user_id found for agent: {agent_name}")


def build_scoped_user_id(raw_user_id: str, save_name: str) -> str:
    base = str(raw_user_id or "").strip()
    save = str(save_name or "").strip()
    if not base:
        return ""
    if not save:
        return base
    return f"{base}+{save}"


def safe_filename(name: str) -> str:
    return re.sub(r'[\\/:*?"<>|]+', "_", str(name or "").strip()) or "unknown"


def choose_first_present(item: Dict[str, Any], keys: List[str]) -> Any:
    for key in keys:
        if key in item and item.get(key) not in (None, ""):
            return item.get(key)
    return None


def preview_text(value: Any, max_len: int = 200) -> str:
    if value is None:
        return ""
    if isinstance(value, (dict, list)):
        text = json.dumps(value, ensure_ascii=False)
    else:
        text = str(value)
    compact = re.sub(r"\s+", " ", text).strip()
    if len(compact) <= max_len:
        return compact
    return compact[: max_len - 3] + "..."


def summarize_ranked_memories(details: Dict[str, Any]) -> List[str]:
    ranked = details.get("ranked_memories", [])
    if not isinstance(ranked, list) or not ranked:
        return ["- (empty)"]

    lines: List[str] = []
    for idx, item in enumerate(ranked, start=1):
        if not isinstance(item, dict):
            lines.append(f"- {idx}. {preview_text(item)}")
            continue
        remote_id = choose_first_present(item, ["remote_id", "id", "memory_id"])
        score = choose_first_present(item, ["score", "similarity", "rerank_score", "final_score"])
        level = choose_first_present(item, ["level", "memory_level"])
        locked = choose_first_present(item, ["locked", "is_locked"])
        milestone = choose_first_present(item, ["milestone", "is_milestone"])
        create_time = choose_first_present(item, ["create_time", "created_at", "timestamp", "time"])
        content = choose_first_present(item, ["content", "describe", "text", "memory_text", "summary"])

        meta_bits: List[str] = []
        if remote_id is not None:
            meta_bits.append(f"remote_id={remote_id}")
        if score is not None:
            meta_bits.append(f"score={score}")
        if level is not None:
            meta_bits.append(f"level={level}")
        if locked is not None:
            meta_bits.append(f"locked={locked}")
        if milestone is not None:
            meta_bits.append(f"milestone={milestone}")
        if create_time is not None:
            meta_bits.append(f"time={create_time}")

        header = f"- {idx}. " + (", ".join(meta_bits) if meta_bits else "(no metadata)")
        lines.append(header)
        lines.append(f"  - preview: {preview_text(content or item, max_len=240)}")
    return lines


def build_details_overview(details: Dict[str, Any]) -> Dict[str, Any]:
    profile_data = details.get("profile_data", {})
    if not isinstance(profile_data, dict):
        profile_data = {}

    def list_count(key: str) -> int:
        value = details.get(key, [])
        if isinstance(value, list):
            return len(value)
        return 0

    return {
        "l0_history_count": list_count("l0_history"),
        "ranked_memories_count": list_count("ranked_memories"),
        "profile_data_key_count": len(profile_data),
        "recent_emotion_count": list_count("recent_emotion"),
        "milestones_count": list_count("milestones"),
        "short_term_recent_count": list_count("short_term_recent"),
    }


def to_pretty_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def render_markdown(
    metadata: Dict[str, Any],
    resolution_notes: List[str],
    retrieve_ok: bool,
    elapsed_ms: int,
    response: Dict[str, Any],
    error_text: str,
) -> str:
    formatted_prompt = str(response.get("formatted_prompt", "") or "") if isinstance(response, dict) else ""
    details = response.get("details", {}) if isinstance(response, dict) else {}
    if not isinstance(details, dict):
        details = {"_raw": details}
    overview = build_details_overview(details)

    lines: List[str] = [
        "# External Memory Retrieval Debug Report",
        "",
        "## Request metadata",
        "",
    ]
    for key in [
        "generated_at",
        "save_name",
        "save_dir",
        "snapshot_path",
        "selector",
        "resolved_agent_name",
        "resolved_role",
        "config_path",
        "raw_user_id",
        "scoped_user_id",
        "user_id_source",
        "base_url",
        "timeout",
        "active_project",
        "query",
    ]:
        lines.append(f"- {key}: `{metadata.get(key, '')}`")

    lines.extend(
        [
            "",
            "## Retrieval status",
            "",
            f"- success: `{retrieve_ok}`",
            f"- elapsed_ms: `{elapsed_ms}`",
        ]
    )
    if error_text:
        lines.append(f"- error: `{error_text}`")

    lines.extend(
        [
            "",
            "## Resolution details",
            "",
        ]
    )
    if resolution_notes:
        for note in resolution_notes:
            lines.append(f"- {note}")
    else:
        lines.append("- (none)")

    lines.extend(
        [
            "",
            "## formatted_prompt",
            "```text",
            formatted_prompt,
            "```",
            "",
            "## ranked memories summary",
            "",
        ]
    )
    lines.extend(summarize_ranked_memories(details))

    lines.extend(
        [
            "",
            "## details overview",
            "",
        ]
    )
    for key, value in overview.items():
        lines.append(f"- {key}: `{value}`")

    lines.extend(
        [
            "",
            "## raw details JSON",
            "```json",
            to_pretty_json(details),
            "```",
            "",
            "## raw full response JSON",
            "```json",
            to_pretty_json(response if isinstance(response, dict) else {"_raw": response}),
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def default_output_path(save_dir: Path, agent_name: str) -> Path:
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return save_dir / f"external_memory_retrieve_{safe_filename(agent_name)}_{timestamp}.md"


def main() -> int:
    cli_args = parse_args()

    try:
        options = resolve_runtime_options(cli_args)
        save_dir = resolve_save_dir(options["save_name"])
        snapshot_path = resolve_snapshot_path(save_dir, options["snapshot_path"])
        snapshot = load_json_file(snapshot_path)
        global_config = read_global_config()

        agent_name, resolved_role, selector_notes = resolve_selector(
            options["selector"],
            snapshot,
            global_config,
        )
        raw_user_id, user_id_source, config_path, user_id_notes = resolve_user_id(
            snapshot,
            agent_name,
            global_config,
        )
        scoped_user_id = build_scoped_user_id(raw_user_id, options["save_name"])
        if not scoped_user_id:
            raise ValueError("Resolved scoped_user_id is empty")

        output_path = (
            Path(options["output"]).expanduser()
            if str(options["output"] or "").strip()
            else default_output_path(save_dir, agent_name)
        )
        if not output_path.is_absolute():
            output_path = REPO_ROOT / output_path
        if not output_path.parent.exists():
            raise FileNotFoundError(
                f"Output parent directory does not exist: {output_path.parent}"
            )

        metadata = {
            "generated_at": datetime.now().isoformat(timespec="seconds"),
            "save_name": options["save_name"],
            "save_dir": str(save_dir),
            "snapshot_path": str(snapshot_path),
            "selector": options["selector"],
            "resolved_agent_name": agent_name,
            "resolved_role": resolved_role,
            "config_path": str(config_path) if config_path else "",
            "raw_user_id": raw_user_id,
            "scoped_user_id": scoped_user_id,
            "user_id_source": user_id_source,
            "base_url": options["base_url"],
            "timeout": options["timeout"],
            "active_project": options["active_project"],
            "query": options["query"],
        }
        resolution_notes = selector_notes + user_id_notes

        response: Dict[str, Any] = {}
        error_text = ""
        retrieve_ok = False
        start_ts = time.perf_counter()
        try:
            client = ECDollMemoryServiceClient(
                base_url=options["base_url"],
                timeout=options["timeout"],
            )
            response = client.retrieve_memory_context(
                query=options["query"],
                user_id=scoped_user_id,
                active_project=(options["active_project"] or None),
            )
            if not isinstance(response, dict):
                response = {"_raw": response}
            retrieve_ok = True
        except Exception as exc:  # noqa: BLE001
            error_text = f"{type(exc).__name__}: {exc}"
            response = {"error": error_text}
        elapsed_ms = int((time.perf_counter() - start_ts) * 1000)

        markdown = render_markdown(
            metadata=metadata,
            resolution_notes=resolution_notes,
            retrieve_ok=retrieve_ok,
            elapsed_ms=elapsed_ms,
            response=response,
            error_text=error_text,
        )
        output_path.write_text(markdown, encoding="utf-8")

        print(f"Report written to: {output_path}")
        print(f"Resolved agent     : {agent_name}")
        print(f"Scoped user_id     : {scoped_user_id}")
        print(f"Retrieve success   : {retrieve_ok}")
        if error_text:
            print(f"Retrieve error     : {error_text}")
            return 1
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"[ERROR] {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
