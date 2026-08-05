"""隔离式专家（治疗师）与 checkpoint Agent 对话页面。

该页面仅读取 results/checkpoints。每个网页会话会复制对应存档的 associate
存储到 customization/expert_checkpoint_chat/sessions，所有可变状态均在该目录中
运行；可下载的对话记录则以 JSONL 保存在 records 目录。
"""

from __future__ import annotations

import copy
import json
import os
import shutil
import sys
import tempfile
import threading
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import gradio as gr


BASE_DIR = Path(__file__).resolve().parents[2]
if str(BASE_DIR) not in sys.path:
    sys.path.append(str(BASE_DIR))

from modules import utils  # noqa: E402
from modules.game import create_game  # noqa: E402
from modules.memory import Event  # noqa: E402


CHECKPOINTS_ROOT = BASE_DIR / "results" / "checkpoints"
STATIC_ROOT = BASE_DIR / "frontend" / "static"
APP_ROOT = Path(__file__).resolve().parent
SESSIONS_ROOT = APP_ROOT / "sessions"
RECORDS_ROOT = APP_ROOT / "records"
THERAPIST_NAME = "治疗师"
# 项目将 GAME 和 TIMER 保存在全局容器中；串行化并在每轮前重新激活会话，
# 避免不同浏览器会话交叉使用彼此的模拟时钟。
GLOBAL_RUNTIME_LOCK = threading.RLock()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _safe_child(parent: Path, name: str) -> Path:
    """仅允许选择 parent 的直接子项，避免 Dropdown 值被篡改为任意路径。"""
    if not isinstance(name, str) or not name or Path(name).name != name:
        raise ValueError("无效的存档或快照名称。")
    candidate = (parent / name).resolve()
    if candidate.parent != parent.resolve():
        raise ValueError("选择的路径不在允许范围内。")
    return candidate


def list_archives() -> List[str]:
    if not CHECKPOINTS_ROOT.is_dir():
        return []
    return sorted(path.name for path in CHECKPOINTS_ROOT.iterdir() if path.is_dir())


def list_checkpoints(archive_name: str) -> List[str]:
    try:
        archive_dir = _safe_child(CHECKPOINTS_ROOT, archive_name)
    except ValueError:
        return []
    if not archive_dir.is_dir():
        return []
    return sorted(
        path.name
        for path in archive_dir.glob("simulate-*.json")
        if path.is_file()
    )


def load_checkpoint(archive_name: str, checkpoint_name: str) -> Dict[str, Any]:
    archive_dir = _safe_child(CHECKPOINTS_ROOT, archive_name)
    checkpoint_path = _safe_child(archive_dir, checkpoint_name)
    if not checkpoint_path.is_file() or not checkpoint_path.name.startswith("simulate-"):
        raise ValueError("找不到所选 checkpoint。")
    with checkpoint_path.open("r", encoding="utf-8") as file:
        config = json.load(file)
    if not isinstance(config, dict) or not isinstance(config.get("agents"), dict):
        raise ValueError("checkpoint 格式无效：缺少 agents。")
    return config


def normalize_checkpoint_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """深拷贝并将配置路径重新锚定到当前项目，绝不信任存档中的旧绝对路径。"""
    normalized = copy.deepcopy(config)
    if isinstance(normalized.get("time"), str):
        normalized["time"] = {"start": normalized["time"]}

    agents = normalized.get("agents", {})
    for agent_name, agent_config in agents.items():
        if not isinstance(agent_config, dict):
            continue
        agent_dir = STATIC_ROOT / "assets" / "village" / "agents" / agent_name.replace(" ", "_")
        agent_config["config_path"] = str(
            Path("assets") / "village" / "agents" / agent_name.replace(" ", "_") / "agent.json"
        )
        depression_config = agent_dir / "depression_config.json"
        # 快照里的绝对路径可能指向另一台机器或旧版代码；只使用当前工作区配置。
        if depression_config.is_file():
            agent_config["depression_config_path"] = str(depression_config)
        else:
            agent_config.pop("depression_config_path", None)
    return normalized


def list_agents(archive_name: str, checkpoint_name: str) -> List[str]:
    if not archive_name or not checkpoint_name:
        return []
    try:
        config = load_checkpoint(archive_name, checkpoint_name)
    except (OSError, ValueError, json.JSONDecodeError):
        return []
    return sorted(str(name) for name in config.get("agents", {}).keys())


@dataclass
class TherapistStub:
    """将人工专家明确建模为治疗师，供原对话 prompt 与抑郁模块共同使用。"""

    tile: object
    name: str = THERAPIST_NAME

    def __post_init__(self) -> None:
        self._event = Event(
            self.name,
            "正在",
            "进行专业访谈",
            describe=f"{self.name} 正在进行专业访谈",
            address=self.tile.get_address(),
        )

    def get_event(self):
        return self._event

    def get_tile(self):
        return self.tile


class ExpertCheckpointSession:
    """单个网页会话的隔离运行时与 JSONL 记录器。"""

    def __init__(self, archive_name: str, checkpoint_name: str, agent_name: str):
        self.archive_name = archive_name
        self.checkpoint_name = checkpoint_name
        self.agent_name = agent_name
        source_config = load_checkpoint(archive_name, checkpoint_name)
        if agent_name not in source_config.get("agents", {}):
            raise ValueError(f"checkpoint 中没有角色：{agent_name}")

        self.session_id = uuid.uuid4().hex
        SESSIONS_ROOT.mkdir(parents=True, exist_ok=True)
        RECORDS_ROOT.mkdir(parents=True, exist_ok=True)
        self.session_dir = Path(
            tempfile.mkdtemp(prefix="session-", dir=str(SESSIONS_ROOT))
        )
        self.storage_root = self.session_dir / "storage"
        self._copy_storage_snapshot()

        self.config = normalize_checkpoint_config(source_config)
        # 原始历史对话只读地作为游戏上下文载入；本页不会将新对话写回它。
        self.conversation = self._load_historical_conversation()
        self.logger = utils.create_io_logger("info")
        self.game = create_game(
            f"expert-session-{self.session_id}",
            str(STATIC_ROOT),
            self.config,
            self.conversation,
            logger=self.logger,
            storage_root=str(self.storage_root),
        )
        self.agent = self.game.get_agent(agent_name)
        # 保留 reset()：它会初始化 LLM，并按需求补全动态抑郁主诉图窗口。
        self.agent.reset()
        self.chats: List[Tuple[str, str]] = [
            (str(item[0]), str(item[1]))
            for item in (self.agent.chats or [])
            if isinstance(item, (list, tuple)) and len(item) >= 2
        ]
        self.turn = 0
        # 仅以 UUID 命名文件，避免存档中意外的角色名成为输出路径的一部分。
        self.record_path = RECORDS_ROOT / f"expert-dialogue-{self.session_id}.jsonl"
        self._append_record(
            {
                "type": "session_start",
                "timestamp": _utc_now(),
                "session_id": self.session_id,
                "archive": archive_name,
                "checkpoint": checkpoint_name,
                "agent": agent_name,
                "expert_name": THERAPIST_NAME,
                "expert_role": "治疗师",
                "loaded_history_turns": len(self.chats),
                "storage_mode": "isolated_copy",
            }
        )

    def activate(self) -> None:
        """在当前请求中恢复本会话对应的全局游戏与模拟时钟。"""
        utils.set_timer(**self.config.get("time", {}))
        utils.GenerativeAgentsMap.set(utils.GenerativeAgentsKey.GAME, self.game)

    def _archive_dir(self) -> Path:
        return _safe_child(CHECKPOINTS_ROOT, self.archive_name)

    def _copy_storage_snapshot(self) -> None:
        source_storage = self._archive_dir() / "storage"
        if source_storage.is_dir():
            shutil.copytree(source_storage, self.storage_root)
        else:
            self.storage_root.mkdir(parents=True, exist_ok=True)

    def _load_historical_conversation(self) -> Dict[str, Any]:
        source = self._archive_dir() / "conversation.json"
        if not source.is_file():
            return {}
        with source.open("r", encoding="utf-8") as file:
            data = json.load(file)
        return data if isinstance(data, dict) else {}

    def _append_record(self, record: Dict[str, Any]) -> None:
        with self.record_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    def chat(self, therapist_text: str) -> str:
        therapist_text = str(therapist_text or "").strip()
        if not therapist_text:
            raise ValueError("请输入访谈内容。")

        therapist = TherapistStub(self.agent.get_tile())
        relation = self.agent.completion("summarize_relation", self.agent, THERAPIST_NAME)
        chats = list(self.chats)
        chats.append((THERAPIST_NAME, therapist_text))
        reply = self.agent.completion(
            "generate_chat", self.agent, therapist, relation, chats
        )
        reply = str(reply or "")
        chats.append((self.agent.name, reply))
        self.chats = chats
        self.turn += 1

        # 只把记忆写入 session 的 storage 副本，便于同一场访谈的后续轮次保持连贯。
        summary = self.agent.completion("summarize_chats", chats)
        event = Event(
            self.agent.name,
            "对话",
            THERAPIST_NAME,
            describe=str(summary or ""),
            address=self.agent.get_tile().get_address(),
        )
        self.agent._add_concept("chat", event)

        base_record = {
            "type": "message",
            "timestamp": _utc_now(),
            "session_id": self.session_id,
            "turn": self.turn,
            "archive": self.archive_name,
            "checkpoint": self.checkpoint_name,
            "agent": self.agent.name,
        }
        self._append_record({**base_record, "role": "therapist", "name": THERAPIST_NAME, "content": therapist_text})
        self._append_record({**base_record, "role": "agent", "name": self.agent.name, "content": reply})
        return reply


def _default_choice(values: List[str]) -> Optional[str]:
    return values[0] if values else None


def on_refresh_archives():
    archives = list_archives()
    return gr.update(choices=archives, value=_default_choice(archives))


def on_archive_change(archive_name: str):
    checkpoints = list_checkpoints(archive_name) if archive_name else []
    checkpoint = _default_choice(checkpoints)
    agents = list_agents(archive_name, checkpoint) if checkpoint else []
    return (
        gr.update(choices=checkpoints, value=checkpoint),
        gr.update(choices=agents, value=_default_choice(agents)),
        None,
        [],
        [],
        gr.update(value=None, interactive=False),
    )


def on_checkpoint_change(archive_name: str, checkpoint_name: str):
    agents = list_agents(archive_name, checkpoint_name)
    return (
        gr.update(choices=agents, value=_default_choice(agents)),
        None,
        [],
        [],
        gr.update(value=None, interactive=False),
    )


def on_agent_change():
    return None, [], [], gr.update(value=None, interactive=False)


def on_send(
    message: str,
    messages: Optional[List[Dict[str, str]]],
    archive_name: str,
    checkpoint_name: str,
    agent_name: str,
    session: Optional[ExpertCheckpointSession],
):
    messages = list(messages or [])
    try:
        with GLOBAL_RUNTIME_LOCK:
            if not archive_name or not checkpoint_name or not agent_name:
                raise ValueError("请先选择存档、checkpoint 和角色。")
            if not str(message or "").strip():
                raise ValueError("请输入访谈内容。")
            if (
                session is None
                or session.archive_name != archive_name
                or session.checkpoint_name != checkpoint_name
                or session.agent_name != agent_name
            ):
                session = ExpertCheckpointSession(archive_name, checkpoint_name, agent_name)
            session.activate()
            reply = session.chat(message)
    except Exception as exc:
        messages.append({"role": "assistant", "content": f"对话初始化或生成失败：{exc}"})
        return "", messages, messages, session, gr.update(value=None, interactive=False)

    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": reply})
    return (
        "",
        messages,
        messages,
        session,
        gr.update(value=str(session.record_path), interactive=True),
    )


def on_clear(session: Optional[ExpertCheckpointSession]):
    # 清空仅影响网页显示；JSONL 已经是审计记录，隔离中的 agent 会话也继续保留。
    return [], [], "", session


def build_ui():
    archives = list_archives()
    default_archive = _default_choice(archives)
    checkpoints = list_checkpoints(default_archive) if default_archive else []
    default_checkpoint = _default_choice(checkpoints)
    agents = list_agents(default_archive, default_checkpoint) if default_checkpoint else []

    with gr.Blocks(title="专家治疗师访谈 - Checkpoint 沙箱") as demo:
        gr.Markdown(
            "# 专家治疗师访谈（Checkpoint 沙箱）\n"
            "选择精确存档点后开始访谈。所有可变记忆都在 customization 的隔离目录中运行，"
            "不会写入 `results`；可随时下载本次 JSONL 记录。"
        )
        with gr.Row():
            archive_dropdown = gr.Dropdown(
                label="模拟存档", choices=archives, value=default_archive
            )
            refresh_button = gr.Button("刷新存档列表")
        with gr.Row():
            checkpoint_dropdown = gr.Dropdown(
                label="Checkpoint", choices=checkpoints, value=default_checkpoint
            )
            agent_dropdown = gr.Dropdown(
                label="访谈对象", choices=agents, value=_default_choice(agents)
            )
        gr.Markdown("专家身份固定为：**治疗师**")
        # Gradio 6 默认使用 MessageDict 格式，不再接受旧版的 `type` 参数。
        chatbot = gr.Chatbot(label="访谈记录", height=460)
        with gr.Row():
            message = gr.Textbox(label="治疗师输入", placeholder="请输入访谈问题或回应…", lines=2)
            send_button = gr.Button("发送", variant="primary")
        with gr.Row():
            clear_button = gr.Button("清空页面显示")
            download_button = gr.DownloadButton(
                "下载本次 JSONL 记录", value=None, interactive=False
            )

        session_state = gr.State(None)
        messages_state = gr.State([])
        refresh_button.click(on_refresh_archives, outputs=[archive_dropdown])
        archive_dropdown.change(
            on_archive_change,
            inputs=[archive_dropdown],
            outputs=[
                checkpoint_dropdown,
                agent_dropdown,
                session_state,
                messages_state,
                chatbot,
                download_button,
            ],
        )
        checkpoint_dropdown.change(
            on_checkpoint_change,
            inputs=[archive_dropdown, checkpoint_dropdown],
            outputs=[agent_dropdown, session_state, messages_state, chatbot, download_button],
        )
        agent_dropdown.change(
            on_agent_change,
            outputs=[session_state, messages_state, chatbot, download_button],
        )
        send_inputs = [
            message,
            messages_state,
            archive_dropdown,
            checkpoint_dropdown,
            agent_dropdown,
            session_state,
        ]
        send_outputs = [message, messages_state, chatbot, session_state, download_button]
        send_button.click(on_send, inputs=send_inputs, outputs=send_outputs)
        message.submit(on_send, inputs=send_inputs, outputs=send_outputs)
        clear_button.click(
            on_clear,
            inputs=[session_state],
            outputs=[messages_state, chatbot, message, session_state],
        )
    return demo


if __name__ == "__main__":
    build_ui().launch()
