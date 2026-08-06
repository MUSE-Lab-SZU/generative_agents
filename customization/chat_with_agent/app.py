import os
import sys
import json

import gradio as gr


BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from modules.memory import Event  # noqa: E402
from modules import utils  # noqa: E402
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
DEFAULT_USER_NAME = "用户"


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
        return gr.Dropdown(choices=[], value=None), None, [], []
    session = ChatSession(sim_name)
    agents = list_agents(sim_name)
    messages = []
    return (
        gr.Dropdown(choices=agents, value=(agents[0] if agents else None)),
        session,
        messages,
        messages,
    )


def on_agent_change(agent_name, session):
    if not session or not agent_name:
        return session, [], []
    session.set_agent(agent_name)
    messages = []
    return session, messages, messages


def on_send(message, messages, sim_name, agent_name, session):
    if not sim_name:
        return "", messages, messages or [], session
    session = ensure_session(sim_name, session)
    if agent_name and session.agent_name != agent_name:
        session.set_agent(agent_name)
        messages = []
    reply = session.chat(message)
    messages = messages or []
    messages.append({"role": "user", "content": message})
    messages.append({"role": "assistant", "content": reply})
    return "", messages, messages, session


def on_clear(session):
    if session:
        session.clear_history()
    messages = []
    return messages, messages, ""


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
        chatbot = gr.Chatbot(label="对话", height=420)
        with gr.Row():
            message = gr.Textbox(
                label="输入内容",
                placeholder="对角色说点什么吧……",
                lines=2,
            )
            send_button = gr.Button("发送")
        clear_button = gr.Button("清空对话")

        session_state = gr.State(None)
        messages_state = gr.State([])

        refresh_button.click(
            on_refresh,
            inputs=[],
            outputs=[sim_dropdown],
        )
        sim_dropdown.change(
            on_sim_change,
            inputs=[sim_dropdown],
            outputs=[agent_dropdown, session_state, messages_state, chatbot],
        )
        agent_dropdown.change(
            on_agent_change,
            inputs=[agent_dropdown, session_state],
            outputs=[session_state, messages_state, chatbot],
        )
        send_button.click(
            on_send,
            inputs=[message, messages_state, sim_dropdown, agent_dropdown, session_state],
            outputs=[message, messages_state, chatbot, session_state],
        )
        message.submit(
            on_send,
            inputs=[message, messages_state, sim_dropdown, agent_dropdown, session_state],
            outputs=[message, messages_state, chatbot, session_state],
        )
        clear_button.click(
            on_clear,
            inputs=[session_state],
            outputs=[messages_state, chatbot, message],
        )

    return demo


if __name__ == "__main__":
    ui = build_ui()
    ui.launch()
