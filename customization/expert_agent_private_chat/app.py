import os
import sys

import gradio as gr


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from customization.depression_scale_agent.ExpertLLM import ExpertLLM  # noqa: E402
from customization.snapshot_ui_service import (  # noqa: E402
    UserStub,
    create_checkpoint_game,
    list_agents as _list_agents,
    list_simulations as _list_simulations,
    load_config as _load_config,
    load_conversation as _load_conversation,
)


CHECKPOINTS_ROOT = os.path.join(BASE_DIR, "results", "checkpoints")
STATIC_ROOT = os.path.join(BASE_DIR, "frontend", "static")
DEFAULT_USER_NAME = "专家"


class ChatSession:
    def __init__(self, sim_name):
        self.sim_name = sim_name
        self.config = load_latest_config(sim_name)
        if self.config is None:
            raise ValueError(f"找不到存档: {sim_name}")
        self.conversation = load_conversation(sim_name)
        self.logger, self.game = create_checkpoint_game(
            sim_name,
            STATIC_ROOT,
            self.config,
            self.conversation,
        )
        self.agent_name = ""
        self.agent = None

    def set_agent(self, agent_name):
        self.agent_name = agent_name
        self.agent = self.game.get_agent(agent_name)
        self.agent.reset()

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


def list_simulations():
    return _list_simulations(CHECKPOINTS_ROOT)


def load_latest_config(sim_name):
    return _load_config(
        CHECKPOINTS_ROOT,
        sim_name,
        overwrite_agent_config_paths=True,
    )


def load_conversation(sim_name):
    return _load_conversation(CHECKPOINTS_ROOT, sim_name)


def list_agents(sim_name):
    return _list_agents(
        CHECKPOINTS_ROOT,
        sim_name,
        overwrite_agent_config_paths=True,
    )


def ensure_session(sim_name, session):
    if session and session.sim_name == sim_name:
        return session
    return ChatSession(sim_name)


def on_refresh():
    sims = list_simulations()
    return gr.Dropdown(choices=sims, value=(sims[0] if sims else None))


def on_sim_change(sim_name):
    if not sim_name:
        return gr.Dropdown(choices=[], value=None), None, [], ""
    session = ChatSession(sim_name)
    agents = list_agents(sim_name)
    return (
        gr.Dropdown(choices=agents, value=(agents[0] if agents else None)),
        session,
        [],
        "",
    )


def on_agent_change(agent_name, session):
    if not session or not agent_name:
        return session, [], ""
    session.set_agent(agent_name)
    return session, [], ""


def on_start_dialogue(
    sim_name,
    agent_name,
    session,
    system_prompt,
    jsonl_file,
    end_symbol,
    max_turns,
    model_name,
):
    if not sim_name or not agent_name:
        yield [], "请先选择存档和角色。", session
        return
    if not jsonl_file:
        yield [], "请先选择 jsonl 文件。", session
        return
    if not end_symbol:
        yield [], "请先设置终止符号。", session
        return
    if max_turns is None or max_turns <= 0:
        yield [], "最大次数需为正整数。", session
        return

    session = ensure_session(sim_name, session)
    if session.agent_name != agent_name:
        session.set_agent(agent_name)

    try:
        with open(jsonl_file, "r", encoding="utf-8") as f:
            expert_start_prompt = f.read().strip()
    except Exception as exc:
        yield [], f"读取 jsonl 文件失败: {exc}", session
        return
    if not expert_start_prompt:
        yield [], "jsonl 文件内容为空。", session
        return

    try:
        llm = ExpertLLM(model=model_name)
    except Exception as exc:
        yield [], f"专家模型初始化失败: {exc}", session
        return

    expert_messages = [{"role": "user", "content": expert_start_prompt}]
    chat_messages = []
    status = "进行中"
    yield chat_messages, status, session

    for idx in range(int(max_turns)):
        try:
            expert_reply = llm.chat(
                expert_messages,
                system_prompt=system_prompt or None,
                caller="expert_private_chat",
            )
        except Exception as exc:
            yield chat_messages, f"专家模型调用失败: {exc}", session
            return

        chat_messages.append({"role": "user", "content": expert_reply})
        status = f"进行中：{idx + 1}/{max_turns}"
        yield chat_messages, status, session

        if end_symbol in expert_reply:
            status = f"已终止：专家输出终止符号（{end_symbol}）"
            yield chat_messages, status, session
            return

        try:
            agent_reply = session.answer_without_memory(
                expert_reply, user_name=DEFAULT_USER_NAME
            )
        except Exception as exc:
            yield chat_messages, f"角色回复失败: {exc}", session
            return

        chat_messages.append({"role": "assistant", "content": agent_reply})
        expert_messages.append({"role": "assistant", "content": expert_reply})
        expert_messages.append({"role": "user", "content": agent_reply})
        yield chat_messages, status, session

    status = f"已终止：达到最大次数（{max_turns}）"
    yield chat_messages, status, session


def build_ui():
    sims = list_simulations()
    default_sim = sims[0] if sims else None
    agents = list_agents(default_sim) if default_sim else []
    default_agent = agents[0] if agents else None

    with gr.Blocks(title="专家评估私聊") as demo:
        gr.Markdown("# 专家评估私聊（Agent ⇄ 专家 LLM）")
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

        with gr.Row():
            system_prompt = gr.Textbox(
                label="System Prompt",
                placeholder="输入专家模型系统提示词（可选）",
                lines=2,
            )
            model_name = gr.Textbox(
                label="模型名称",
                value="deepseek-chat",
                interactive=True,
            )
        jsonl_file = gr.File(
            label="输入 jsonl 文件",
            file_types=[".jsonl"],
            type="filepath",
        )
        with gr.Row():
            end_symbol = gr.Textbox(
                label="终止符号",
                value="[[EVAL_DONE]]",
            )
            max_turns = gr.Number(
                label="最大次数",
                value=100,
                precision=0,
            )
        start_button = gr.Button("开始对话")
        status = gr.Textbox(label="状态", interactive=False)
        chatbot = gr.Chatbot(label="私聊对话", height=420)

        session_state = gr.State(None)

        refresh_button.click(
            on_refresh,
            inputs=[],
            outputs=[sim_dropdown],
        )
        sim_dropdown.change(
            on_sim_change,
            inputs=[sim_dropdown],
            outputs=[agent_dropdown, session_state, chatbot, status],
        )
        agent_dropdown.change(
            on_agent_change,
            inputs=[agent_dropdown, session_state],
            outputs=[session_state, chatbot, status],
        )
        start_button.click(
            on_start_dialogue,
            inputs=[
                sim_dropdown,
                agent_dropdown,
                session_state,
                system_prompt,
                jsonl_file,
                end_symbol,
                max_turns,
                model_name,
            ],
            outputs=[chatbot, status, session_state],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
