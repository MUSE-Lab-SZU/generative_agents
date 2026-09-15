"""从 checkpoint 创建临时量表访谈运行时，不写回模拟状态。"""

import copy
import shutil
import tempfile
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path

from modules import utils
from modules.game import create_game
from modules.memory import Event

from customization.checkpoint_depression_scale.assessment import BASE_DIR, load_checkpoint, safe_child

RUNTIME_LOCK = threading.RLock()
STATIC_ROOT = BASE_DIR / "frontend" / "static"


@dataclass
class Therapist:
    tile: object
    name: str = "治疗师"

    def get_tile(self):
        return self.tile

    def get_event(self):
        return Event(self.name, "正在", "进行总体抑郁水平及干扰程度量表测试",
                     address=self.tile.get_address())


class AssessmentSession:
    def __init__(self, agent, metadata):
        self.agent = agent
        self.metadata = metadata
        self.chats = []
        self.therapist = Therapist(agent.get_tile())
        self.relation = agent.completion("summarize_relation", agent, self.therapist.name)

    def chat(self, prompt):
        self.chats.append((self.therapist.name, prompt))
        reply = self.agent.completion("generate_chat", self.agent, self.therapist, self.relation, list(self.chats))
        self.chats.append((self.agent.name, str(reply or "")))
        return reply


@contextmanager
def checkpoint_session(archive, checkpoint, agent_name, *, root):
    # GAME、TIMER 和 embedding Settings 是进程级状态，整场量表串行运行。
    with RUNTIME_LOCK, tempfile.TemporaryDirectory(prefix="checkpoint-scale-") as temporary:
        config = copy.deepcopy(load_checkpoint(archive, checkpoint, root))
        archive_dir = safe_child(root, archive)
        storage = Path(temporary) / "storage"
        source = safe_child(archive_dir / "storage", agent_name)
        agent_config = config["agents"][agent_name]
        references = agent_config.get("associate", {}).get("memory", {})
        if source.is_dir():
            shutil.copytree(source, storage / agent_name)
        elif any(references.values()):
            raise ValueError("存档包含记忆引用但缺少该 agent 的 storage，无法完整恢复。")
        else:
            storage.mkdir()
        if isinstance(config.get("time"), str):
            config["time"] = {"start": config["time"]}
        # 只构建被测角色；其历史记忆仍来自该存档对应的共享存储副本。
        config["agents"] = {agent_name: agent_config}
        agent_dir = STATIC_ROOT / "assets" / "village" / "agents" / agent_name.replace(" ", "_")
        agent_config["config_path"] = str(agent_dir.relative_to(STATIC_ROOT) / "agent.json")
        depression_config = agent_dir / "depression_config.json"
        if depression_config.is_file():
            agent_config["depression_config_path"] = str(depression_config)
        else:
            agent_config.pop("depression_config_path", None)
        keys = (utils.GenerativeAgentsKey.GAME, utils.GenerativeAgentsKey.TIMER)
        previous = {key: utils.GenerativeAgentsMap.get(key) for key in keys}
        try:
            # conversation.json 汇总了整个模拟，可能包含当前 checkpoint 之后的对话。
            game = create_game("checkpoint-scale", str(STATIC_ROOT), config, {},
                               logger=utils.create_io_logger("info"), storage_root=str(storage))
            agent = game.get_agent(agent_name)
            # Associate 初始化会按模拟时钟移除未来/过期节点；检查快照引用是否齐全。
            missing = [node for nodes in references.values() for node in nodes
                       if not agent.associate.index.has_node(node)]
            metadata = {"missing_memory_references": missing,
                        "conversation_source": "checkpoint_memory_only",
                        "reset_before_assessment": True,
                        "dynamic_state_mode": "evolves_within_assessment"}
            agent.reset()
            if not agent.llm_available():
                raise RuntimeError("Agent 的 LLM 不可用，请检查存档对应的模型配置和服务。")
            yield AssessmentSession(agent, metadata)
        finally:
            for key, value in previous.items():
                if value is None:
                    utils.GenerativeAgentsMap.delete(key)
                else:
                    utils.GenerativeAgentsMap.set(key, value)
