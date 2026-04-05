import os
import sys
import json
from dataclasses import dataclass

import gradio as gr


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.game import create_game  # noqa: E402
from modules.memory import Event  # noqa: E402
from modules import utils  # noqa: E402
from customization.depression_scale_agent.ExpertLLM import ExpertLLM  # type: ignore


CHECKPOINTS_ROOT = os.path.join(BASE_DIR, "results", "checkpoints")
STATIC_ROOT = os.path.join(BASE_DIR, "frontend", "static")
DEFAULT_USER_NAME = "用户"
OUTPUT_ROOT = os.path.join(BASE_DIR, "customization", "depression_scale_agent", "questions")


@dataclass
class UserStub:
    name: str
    tile: object

    def __post_init__(self):
        address = self.tile.get_address()
        self._event = Event(
            self.name,
            "正在",
            "聊天",
            describe=f"{self.name} 正在聊天",
            address=address,
        )

    def get_event(self):
        return self._event

    def get_tile(self):
        return self.tile


class ChatSession:
    def __init__(self, sim_name):
        self.sim_name = sim_name
        self.config = load_latest_config(sim_name)
        if self.config is None:
            raise ValueError(f"找不到存档: {sim_name}")
        self.conversation = load_conversation(sim_name)
        self.logger = utils.create_io_logger("info")
        self.game = create_game(
            sim_name,
            STATIC_ROOT,
            self.config,
            self.conversation,
            logger=self.logger,
        )
        self.game.reset_game()
        self.agent_name = ""
        self.agent = None
        self.chats = []

    def set_agent(self, agent_name):
        self.agent_name = agent_name
        self.agent = self.game.get_agent(agent_name)
        self.agent.reset()
        self.chats = []

    def clear_history(self):
        self.chats = []

    def chat(self, user_text, user_name=DEFAULT_USER_NAME):
        if not self.agent:
            return "请先选择一个角色。"
        if not user_text.strip():
            return "请先输入内容。"

        user = UserStub(user_name, self.agent.get_tile())
        relation = self.agent.completion("summarize_relation", self.agent, user_name)

        chats = list(self.chats)
        chats.append((user_name, user_text))
        reply = self.agent.completion("generate_chat", self.agent, user, relation, chats)
        chats.append((self.agent.name, reply))
        self.chats = chats

        summary = self.agent.completion("summarize_chats", chats)
        event = Event(
            self.agent.name,
            "对话",
            user_name,
            describe=summary,
            address=self.agent.get_tile().get_address(),
        )
        self.agent._add_concept("chat", event)
        self._append_conversation(chats, user_name)
        return reply

    def answer_without_memory(self, user_text, user_name=DEFAULT_USER_NAME):
        if not self.agent:
            return "请先选择一个角色。"
        if not user_text.strip():
            return "请先输入内容。"

        user = UserStub(user_name, self.agent.get_tile())
        relation = self.agent.completion("summarize_relation", self.agent, user_name)
        chats = [(user_name, user_text)]
        reply = self.agent.completion("generate_chat", self.agent, user, relation, chats)
        return reply

    def _append_conversation(self, chats, user_name):
        key = utils.get_timer().get_date("%Y%m%d-%H:%M")
        address = "，".join(self.agent.get_event().address)
        entry = {f"{user_name} -> {self.agent.name} @ {address}": chats}
        self.conversation.setdefault(key, []).append(entry)
        conversation_path = os.path.join(
            CHECKPOINTS_ROOT, self.sim_name, "conversation.json"
        )
        with open(conversation_path, "w", encoding="utf-8") as f:
            json.dump(self.conversation, f, indent=2, ensure_ascii=False)


def list_simulations():
    if not os.path.isdir(CHECKPOINTS_ROOT):
        return []
    return sorted(
        d
        for d in os.listdir(CHECKPOINTS_ROOT)
        if os.path.isdir(os.path.join(CHECKPOINTS_ROOT, d))
    )


def load_latest_config(sim_name):
    folder = os.path.join(CHECKPOINTS_ROOT, sim_name)
    if not os.path.isdir(folder):
        return None
    files = sorted(
        f
        for f in os.listdir(folder)
        if f.endswith(".json") and f != "conversation.json"
    )
    if not files:
        return None
    latest = os.path.join(folder, files[-1])
    with open(latest, "r", encoding="utf-8") as f:
        config = json.load(f)
    if isinstance(config.get("time"), str):
        config["time"] = {"start": config["time"]}
    assets_root = os.path.join("assets", "village")
    for agent_name in config.get("agents", {}):
        config["agents"][agent_name]["config_path"] = os.path.join(
            assets_root, "agents", agent_name.replace(" ", "_"), "agent.json"
        )
    return config


def load_conversation(sim_name):
    path = os.path.join(CHECKPOINTS_ROOT, sim_name, "conversation.json")
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_agents(sim_name):
    config = load_latest_config(sim_name)
    if not config:
        return []
    return sorted(config.get("agents", {}).keys())


def ensure_session(sim_name, session):
    if session and session.sim_name == sim_name:
        return session
    return ChatSession(sim_name)


def iter_jsonl(path):
    with open(path, "r", encoding="utf-8") as f:
        for raw_line in f:
            line = raw_line.strip()
            if not line:
                continue
            yield json.loads(line)


def resolve_question(item):
    if isinstance(item, str):
        return item
    if not isinstance(item, dict):
        raise ValueError(f"Unsupported jsonl item: {item}")
    for key in ("question", "prompt", "text", "content"):
        if key in item and str(item[key]).strip():
            return str(item[key]).strip()
    if len(item) == 1:
        return str(next(iter(item.values()))).strip()
    raise ValueError(f"Missing question text in item: {item}")


def default_output_path(input_path):
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    filename = os.path.basename(input_path)
    base, ext = os.path.splitext(filename)
    suffix = "_answered"
    if ext.lower() == ".jsonl":
        output_name = f"{base}{suffix}.jsonl"
    else:
        output_name = f"{filename}{suffix}.jsonl"
    return os.path.join(OUTPUT_ROOT, output_name)


def write_jsonl_line(handle, data):
    handle.write(json.dumps(data, ensure_ascii=False) + "\n")


def on_refresh():
    sims = list_simulations()
    return gr.Dropdown(choices=sims, value=(sims[0] if sims else None))


def on_sim_change(sim_name):
    if not sim_name:
        return gr.Dropdown(choices=[], value=None), None
    session = ChatSession(sim_name)
    agents = list_agents(sim_name)
    return (
        gr.Dropdown(choices=agents, value=(agents[0] if agents else None)),
        session,
    )


def on_agent_change(agent_name, session):
    if not session or not agent_name:
        return session
    session.set_agent(agent_name)
    return session


def on_jsonl_change(jsonl_file):
    if not jsonl_file:
        return "", None, [], ""
    output_path = default_output_path(jsonl_file)
    return output_path, None, [], ""


def on_batch_run(jsonl_file, sim_name, agent_name, session):
    if not jsonl_file:
        yield [], "请先选择 jsonl 文件。", None, session
        return
    if not sim_name or not agent_name:
        yield [], "请先选择存档和角色。", None, session
        return

    session = ensure_session(sim_name, session)
    if session.agent_name != agent_name:
        session.set_agent(agent_name)

    output_path = default_output_path(jsonl_file)
    rows = []
    try:
        with open(output_path, "w", encoding="utf-8") as f:
            for idx, item in enumerate(iter_jsonl(jsonl_file)):
                question = resolve_question(item)
                answer = session.answer_without_memory(question)
                result = dict(item) if isinstance(item, dict) else {"question": question}
                result["answer"] = answer
                write_jsonl_line(f, result)
                rows.append([idx + 1, question, answer])
                yield rows, f"已完成 {idx + 1} 条", output_path, session
    except Exception as exc:
        yield rows, f"执行失败: {exc}", None, session
        return

    yield rows, f"完成，共 {len(rows)} 条", output_path, session


def on_expert_review(answered_file, system_prompt, model_name):
    if not answered_file:
        return "请先生成 answered 的 jsonl 文件。"
    if not os.path.exists(answered_file):
        return f"文件不存在: {answered_file}"
    try:
        with open(answered_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
    except Exception as exc:
        return f"读取文件失败: {exc}"
    if not content:
        return "answered 文件内容为空。"

    try:
        llm = ExpertLLM(model=model_name)
        reply = llm.generate(content, system_prompt=system_prompt or None)
        return reply or ""
    except Exception as exc:
        return f"专家模型调用失败: {exc}"


def build_ui():
    sims = list_simulations()
    default_sim = sims[0] if sims else None
    agents = list_agents(default_sim) if default_sim else []
    default_agent = agents[0] if agents else None

    with gr.Blocks(title="AI 小镇 - 角色对话") as demo:
        gr.Markdown("# AI 小镇 - 角色对话")
        with gr.Row():
            sim_dropdown = gr.Dropdown(
                label="选择模拟存档",
                choices=sims,
                value=default_sim,
                interactive=True,
            )
            refresh_button = gr.Button("刷新存档")
        agent_dropdown = gr.Dropdown(
            label="选择角色",
            choices=agents,
            value=default_agent,
            interactive=True,
        )

        gr.Markdown("## 批量问答（jsonl）")
        with gr.Row():
            jsonl_file = gr.File(
                label="选择 jsonl 文件",
                file_types=[".jsonl"],
                type="filepath",
            )
            output_path = gr.Textbox(
                label="输出文件路径",
                interactive=False,
            )
        batch_button = gr.Button("开始批量问答")
        batch_status = gr.Textbox(label="批量状态", interactive=False)
        batch_table = gr.Dataframe(
            headers=["序号", "问题", "回复"],
            datatype=["number", "str", "str"],
            row_count=0,
            col_count=(3, "fixed"),
            interactive=False,
        )
        batch_output = gr.File(label="输出文件", interactive=False)
        gr.Markdown("## 专家模型总结")
        with gr.Row():
            expert_system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="输入专家模型的系统提示词（可选）",
                lines=2,
            )
            expert_model = gr.Textbox(
                label="模型名称",
                value="deepseek-chat",
                interactive=True,
            )
        expert_button = gr.Button("调用专家模型")
        expert_result = gr.Textbox(label="专家模型输出", lines=8, interactive=False)

        session_state = gr.State(None)

        refresh_button.click(
            on_refresh,
            inputs=[],
            outputs=[sim_dropdown],
        )
        sim_dropdown.change(
            on_sim_change,
            inputs=[sim_dropdown],
            outputs=[agent_dropdown, session_state],
        )
        agent_dropdown.change(
            on_agent_change,
            inputs=[agent_dropdown, session_state],
            outputs=[session_state],
        )
        jsonl_file.change(
            on_jsonl_change,
            inputs=[jsonl_file],
            outputs=[output_path, batch_output, batch_table, batch_status],
        )
        batch_button.click(
            on_batch_run,
            inputs=[jsonl_file, sim_dropdown, agent_dropdown, session_state],
            outputs=[batch_table, batch_status, batch_output, session_state],
        )
        expert_button.click(
            on_expert_review,
            inputs=[batch_output, expert_system_prompt, expert_model],
            outputs=[expert_result],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
