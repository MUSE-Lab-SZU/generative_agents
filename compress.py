import os
import json
import argparse
from datetime import datetime, timedelta

from modules.maze import Maze
from replay_protocol import FILE_MARKDOWN, FILE_MOVEMENT, FRAMES_PER_STEP
from simulation_roster import resolve_run_context as _shared_resolve_run_context


# 村庄模式的角色花名册兜底值（与 start.personas 保持一致）。正常回放时，
# 角色花名册由 _resolve_run_context 从存档动态读取。
def _assets_base(assets_root):
    """根据 assets 子目录名返回完整资源根路径。"""
    return f"frontend/static/assets/{assets_root}"


def _resolve_run_context(checkpoints_folder, assets_root_override=None):
    return _shared_resolve_run_context(checkpoints_folder, assets_root_override)

file_markdown = FILE_MARKDOWN
file_movement = FILE_MOVEMENT
frames_per_step = FRAMES_PER_STEP  # 每个step包含的帧数


# 从存档文件中读取stride
def get_stride(json_files):
    if len(json_files) < 1:
        return 1

    with open(json_files[-1], "r", encoding="utf-8") as f:
        config = json.load(f)

    return config["stride"]


# 将address转换为字符串
def get_location(address):
    # 仅为兼容原版
    # if address[0] == "<waiting>" or address[0] == "<persona>":
    #     return None

    # 不需要显示address第一级（"the Ville"）
    location = "，".join(address[1:])

    return location


def _extract_depression_current_stage(runtime):
    runtime = runtime if isinstance(runtime, dict) else {}
    chain_payload = runtime.get("chain_manager", {}) if isinstance(runtime.get("chain_manager", {}), dict) else {}
    config = chain_payload.get("config", {}) if isinstance(chain_payload.get("config", {}), dict) else {}
    stages = config.get("stages", []) if isinstance(config.get("stages", []), list) else []
    stage_catalog = {}
    for item in stages:
        if not isinstance(item, dict):
            continue
        stage_id = str(item.get("id", "") or "").strip()
        if stage_id:
            stage_catalog[stage_id] = item

    planned_chain = chain_payload.get("planned_chain", []) if isinstance(chain_payload.get("planned_chain", []), list) else []
    try:
        stage_index = int(chain_payload.get("stage_index", 0) or 0)
    except Exception:
        stage_index = 0

    stage_id = ""
    if planned_chain:
        bounded_index = min(max(stage_index, 0), len(planned_chain) - 1)
        stage_id = str(planned_chain[bounded_index] or "").strip()
    if not stage_id:
        stage_id = str(config.get("initial_stage_id", "") or "").strip()

    if stage_id and stage_id in stage_catalog:
        return stage_catalog[stage_id]
    if stage_id:
        return {"id": stage_id, "label": stage_id}
    return {}


def get_depression_runtime_snapshot(agent_data):
    """提取回放所需的抑郁运行态关键字段。"""
    snapshot = {
        "current_event_wording": "",
        "emotion_label": "",
        "emotion_style": "",
        "emotion_intensity": 0.0,
        "last_throttle_reason": "",
        "short_term_changed": False,
        "long_term_changed": False,
        "changed_paths": [],
    }
    if not isinstance(agent_data, dict):
        return snapshot

    runtime = agent_data.get("depression_dynamic_state", {})
    if not isinstance(runtime, dict) or not runtime:
        return snapshot

    stage = _extract_depression_current_stage(runtime)
    for key in ("summary", "label", "core_belief", "id"):
        text = str(stage.get(key, "") or "").strip()
        if text:
            snapshot["current_event_wording"] = text
            break

    emotion = runtime.get("last_emotion", {}) if isinstance(runtime.get("last_emotion", {}), dict) else {}
    snapshot["emotion_label"] = str(emotion.get("label", "") or "")
    snapshot["emotion_style"] = str(emotion.get("style", "") or "")
    try:
        snapshot["emotion_intensity"] = float(emotion.get("intensity", 0.0) or 0.0)
    except Exception:
        snapshot["emotion_intensity"] = 0.0
    return snapshot


# 插入第0帧数据（Agent的初始状态）
def insert_frame0(init_pos, movement, agent_name, assets_root):
    key = "0"
    if key not in movement.keys():
        movement[key] = dict()

    json_path = f"{_assets_base(assets_root)}/agents/{agent_name}/agent.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        address = json_data["spatial"]["address"]["living_area"]
    location = get_location(address)
    coord = json_data["coord"]
    init_pos[agent_name] = coord
    movement[key][agent_name] = {
        "location": location,
        "movement": coord,
        "description": "正在睡觉",
    }
    movement["description"][agent_name] = {
        "currently": json_data["currently"],
        "scratch": json_data["scratch"],
    }


# 从所有存档文件中提取数据（用于回放）
def generate_movement(checkpoints_folder, compressed_folder, compressed_file, assets_root):
    movement_file = os.path.join(compressed_folder, compressed_file)

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    files = sorted(os.listdir(checkpoints_folder))
    json_files = list()
    for file_name in files:
        if file_name.startswith("simulate-") and file_name.endswith(".json"):
            json_files.append(os.path.join(checkpoints_folder, file_name))

    persona_init_pos = dict()
    all_movement = dict()
    all_movement["description"] = dict()
    all_movement["conversation"] = dict()

    stride = get_stride(json_files)
    sec_per_step = stride

    result = {
        "start_datetime": "",  # 起始时间
        "stride": stride,  # 每个step对应的分钟数（必须与生成时的参数一致）
        "sec_per_step": sec_per_step,  # 回放时每一帧对应的秒数
        "persona_init_pos": persona_init_pos,  # 每个Agent的初始位置
        "all_movement": all_movement,  # 所有Agent在每个setp中的位置变化
    }

    last_location = dict()

    # 加载地图数据，用于计算Agent移动路径
    json_path = f"{_assets_base(assets_root)}/maze.json"
    with open(json_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
        maze = Maze(json_data, None)

    for file_name in json_files:
        # 依次读取所有存档文件
        with open(file_name, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            step = json_data["step"]
            agents = json_data["agents"]

            # 保存回放的起始时间
            if len(result["start_datetime"]) < 1:
                t = datetime.strptime(json_data["time"], "%Y%m%d-%H:%M")
                result["start_datetime"] = (
                    t - timedelta(minutes=max(0, step - 1) * stride)
                ).isoformat()

            # 遍历单个存档文件中的所有Agent
            for agent_name, agent_data in agents.items():
                # 插入第0帧
                if agent_name not in persona_init_pos:
                    insert_frame0(persona_init_pos, all_movement, agent_name, assets_root)

                source_coord = last_location.get(agent_name, all_movement["0"][agent_name])["movement"]
                target_coord = agent_data["coord"]
                location = get_location(agent_data["action"]["event"]["address"])
                if location is None:
                    location = last_location.get(agent_name, all_movement["0"][agent_name])["location"]
                    path = [source_coord]
                else:
                    path = maze.find_path(source_coord, target_coord)

                had_conversation = False
                step_conversation = ""
                persons_in_conversation = []
                step_time = json_data["time"]
                if step_time in conversation.keys():
                    for chats in conversation[step_time]:
                        for persons, chat in chats.items():
                            persons_in_conversation.append(persons.split(" @ ")[0].split(" -> "))
                            step_conversation += f"\n地点：{persons.split(' @ ')[1]}\n\n"
                            for c in chat:
                                agent = c[0]
                                text = c[1]
                                step_conversation += f"{agent}：{text}\n"

                for i in range(frames_per_step):
                    moving = len(path) > 1
                    if len(path) > 0:
                        movement = list(path[0])
                        path = path[1:]
                        if agent_name not in last_location.keys():
                            last_location[agent_name] = dict()
                        last_location[agent_name]["movement"] = movement
                        last_location[agent_name]["location"] = location
                    else:
                        movement = None

                    if moving:
                        action = f"前往 {location}"
                    elif movement is not None:
                        action = agent_data["action"]["event"]["describe"]
                        if len(action) < 1:
                            action = f'{agent_data["action"]["event"]["predicate"]}{agent_data["action"]["event"]["object"]}'

                        # 判断该存档文件中当前Agent是否有新的对话（用于设置图标）
                        for persons in persons_in_conversation:
                            if agent_name in persons:
                                had_conversation = True
                                break

                        # 针对睡觉和对话设置图标
                        if "睡觉" in action:
                            action = "😴 " + action
                        elif had_conversation:
                            action = "💬 " + action

                    step_key = "%d" % ((step-1) * frames_per_step + 1 + i)
                    if step_key not in all_movement.keys():
                        all_movement[step_key] = dict()

                    if movement is not None:
                        runtime_snapshot = get_depression_runtime_snapshot(agent_data)
                        all_movement[step_key][agent_name] = {
                            "location": location,
                            "movement": movement,
                            "action": action,
                            "depression_runtime": runtime_snapshot,
                        }
                all_movement["conversation"][step_time] = step_conversation

    # 保存数据
    with open(movement_file, "w", encoding="utf-8") as f:
        f.write(json.dumps(result, indent=2, ensure_ascii=False))

    return result


# 生成Markdown文档
def generate_report(checkpoints_folder, compressed_folder, compressed_file, assets_root, roster):
    last_state = dict()

    conversation_file = "conversation.json"
    conversation = {}
    if os.path.exists(os.path.join(checkpoints_folder, conversation_file)):
        with open(os.path.join(checkpoints_folder, conversation_file), "r", encoding="utf-8") as f:
            conversation = json.load(f)

    def extract_description():
        markdown_content = "# 基础人设\n\n"
        for agent_name in roster:
            json_path = f"{_assets_base(assets_root)}/agents/{agent_name}/agent.json"
            with open(json_path, "r", encoding="utf-8") as f:
                json_data = json.load(f)
                markdown_content += f"## {agent_name}\n\n"
                markdown_content += f"年龄：{json_data['scratch']['age']}岁  \n"
                markdown_content += f"先天：{json_data['scratch']['innate']}  \n"
                markdown_content += f"后天：{json_data['scratch']['learned']}  \n"
                markdown_content += f"生活习惯：{json_data['scratch']['lifestyle']}  \n"
                markdown_content += f"当前状态：{json_data['currently']}\n\n"
        return markdown_content

    def extract_action(json_data):
        markdown_content = ""
        agents = json_data["agents"]
        for agent_name, agent_data in agents.items():
            if agent_name not in last_state.keys():
                last_state[agent_name] = {"currently": "", "location": "", "action": ""}

            location = "，".join(agent_data["action"]["event"]["address"])
            action = agent_data["action"]["event"]["describe"]

            if location == last_state[agent_name]["location"] and action == last_state[agent_name]["action"]:
                continue

            last_state[agent_name]["location"] = location
            last_state[agent_name]["action"] = action

            if len(markdown_content) < 1:
                markdown_content = f"# {json_data['time']}\n\n"
                markdown_content += "## 活动记录：\n\n"

            markdown_content += f"### {agent_name}\n"

            if len(action) < 1:
                action = "睡觉"

            markdown_content += f"位置：{location}  \n"
            markdown_content += f"活动：{action}  \n"

            runtime_snapshot = get_depression_runtime_snapshot(agent_data)
            if runtime_snapshot["current_event_wording"]:
                markdown_content += f"抑郁运行态-主导事件：{runtime_snapshot['current_event_wording']}  \n"
            if runtime_snapshot["emotion_label"]:
                markdown_content += (
                    "抑郁运行态-当前说话情绪："
                    f"{runtime_snapshot['emotion_label']} "
                    f"(强度={float(runtime_snapshot['emotion_intensity'] or 0.0):.2f})  \n"
                )
            if runtime_snapshot["emotion_style"]:
                markdown_content += f"抑郁运行态-说话风格：{runtime_snapshot['emotion_style']}  \n"
            if runtime_snapshot["last_throttle_reason"]:
                markdown_content += f"抑郁运行态-最近节流原因：{runtime_snapshot['last_throttle_reason']}  \n"
            markdown_content += (
                "抑郁运行态-层变化："
                f"short_term={runtime_snapshot['short_term_changed']}，"
                f"long_term={runtime_snapshot['long_term_changed']}  \n"
            )
            if runtime_snapshot["changed_paths"]:
                markdown_content += "抑郁运行态-变更路径：{}  \n".format(
                    "；".join(runtime_snapshot["changed_paths"])
                )

            markdown_content += f"\n"

        if json_data['time'] not in conversation.keys():
            return markdown_content

        markdown_content += "## 对话记录：\n\n"
        for chats in conversation[json_data['time']]:
            for agents, chat in chats.items():
                markdown_content += f"### {agents}\n\n"
                for item in chat:
                    markdown_content += f"`{item[0]}`\n> {item[1]}\n\n"
        return markdown_content

    all_markdown_content = extract_description()
    files = sorted(os.listdir(checkpoints_folder))
    for file_name in files:
        if (not file_name.startswith("simulate-")) or (not file_name.endswith(".json")):
            continue

        file_path = os.path.join(checkpoints_folder, file_name)
        with open(file_path, "r", encoding="utf-8") as f:
            json_data = json.load(f)
            content = extract_action(json_data)
            all_markdown_content += content + "\n\n"
    with open(f"{compressed_folder}/{compressed_file}", "w", encoding="utf-8") as compressed_file:
        compressed_file.write(all_markdown_content)


parser = argparse.ArgumentParser()
parser.add_argument("--name", type=str, default="", help="the name of the simulation")
parser.add_argument(
    "--assets-root",
    choices=("village", "counsel_room"),
    default=None,
    help="assets 子目录；默认从存档 maze.path 自动检测。",
)
args = parser.parse_args()


if __name__ == "__main__":
    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name: ")

    while not os.path.exists(f"results/checkpoints/{name}"):
        name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")

    checkpoints_folder = f"results/checkpoints/{name}"
    compressed_folder = f"results/compressed/{name}"
    os.makedirs(compressed_folder, exist_ok=True)

    assets_root, roster = _resolve_run_context(checkpoints_folder, args.assets_root)
    print(f"[compress] assets_root={assets_root} roster={roster}")
    generate_report(checkpoints_folder, compressed_folder, file_markdown, assets_root, roster)
    generate_movement(checkpoints_folder, compressed_folder, file_movement, assets_root)
