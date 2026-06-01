import os
import copy
import json
import argparse
import datetime

from dotenv import load_dotenv, find_dotenv

from modules.game import create_game, get_game
from modules import utils
from modules.intervention_manager import InterventionManager
from modules.staged_eval_manager import StagedEvalManager

personas = [
    "卡布达",  # 抑郁症患者
    "金龟次郎",  # 家人（否认型父母）
    "田德莉娜",  # 好友
    "呱呱蛙",  # 邻居
    "蜻蜓队长",  # 心理医生
    "蟑螂恶霸" # 小混混
]


class SimulateServer:
    def __init__(self, name, static_root, checkpoints_folder, config, start_step=0, verbose="info", log_file=""):
        self.name = name
        self.static_root = static_root
        self.checkpoints_folder = checkpoints_folder

        # 历史存档数据（用于断点恢复）
        self.config = config

        os.makedirs(checkpoints_folder, exist_ok=True)

        # 载入历史对话数据（用于断点恢复）
        self.conversation_log = f"{checkpoints_folder}/conversation.json"
        if os.path.exists(self.conversation_log):
            with open(self.conversation_log, "r", encoding="utf-8") as f:
                conversation = json.load(f)
        else:
            conversation = {}

        if len(log_file) > 0:
            self.logger = utils.create_file_logger(f"{checkpoints_folder}/{log_file}", verbose)
        else:
            self.logger = utils.create_io_logger(verbose)

        # 创建游戏
        game = create_game(name, static_root, config, conversation, logger=self.logger)
        game.reset_game()

        self.game = get_game()
        self.intervention = InterventionManager(config=self.config, logger=self.logger)
        self.game.set_intervention_manager(self.intervention)
        self.staged_eval = StagedEvalManager(name, checkpoints_folder, self.config, logger=self.logger)
        self.tile_size = self.game.maze.tile_size
        self.agent_status = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        for agent_name, agent in config["agents"].items():
            agent_config = copy.deepcopy(agent_base)
            agent_config.update(self.load_static(agent["config_path"]))
            self.agent_status[agent_name] = {
                "coord": agent_config["coord"],
                "path": [],
            }
        self.think_interval = max(
            a.think_config["interval"] for a in self.game.agents.values()
        )
        self.start_step = start_step

    def _build_runtime_eval_config(self, step_no=None, sim_time=None):
        runtime_config = copy.deepcopy(self.config)
        if step_no is not None:
            runtime_config["step"] = int(step_no)
        if sim_time is not None:
            runtime_config["time"] = {"start": sim_time}
        for agent_name, agent in self.game.agents.items():
            runtime_config.setdefault("agents", {}).setdefault(agent_name, {})
            runtime_config["agents"][agent_name].update(agent.to_dict())
            status = self.agent_status.get(agent_name, {}) if isinstance(self.agent_status, dict) else {}
            if "coord" in status:
                runtime_config["agents"][agent_name]["coord"] = status["coord"]
        return runtime_config

    def simulate(self, step, stride=0):
        timer = utils.get_timer()
        for i in range(self.start_step, self.start_step + step):
            step_no = i + 1
            title = "Simulate Step[{}/{}, time: {}]".format(step_no, self.start_step + step, timer.get_date())
            self.logger.info("\n" + utils.split_line(title, "="))
            self.intervention.on_step_start(self.game, timer.get_date())
            if self.start_step == 0 and i == self.start_step:
                t0_sim_time = timer.get_date("%Y%m%d-%H:%M")
                self.staged_eval.maybe_run_t0(
                    runtime_config=self._build_runtime_eval_config(step_no=step_no, sim_time=t0_sim_time),
                    conversation=copy.deepcopy(self.game.conversation),
                    step_no=step_no,
                    sim_time=t0_sim_time,
                )
            for name, status in self.agent_status.items():
                plan = self.game.agent_think(name, status)["plan"]
                agent = self.game.get_agent(name)
                if name not in self.config["agents"]:
                    self.config["agents"][name] = {}
                self.config["agents"][name].update(agent.to_dict())
                if plan.get("path"):
                    status["coord"], status["path"] = plan["path"][-1], []
                self.config["agents"][name].update(
                    # {"coord": status["coord"], "path": plan["path"]}
                    {"coord": status["coord"]}
                )

            sim_time = timer.get_date("%Y%m%d-%H:%M")
            self.config.update(
                {
                    "time": sim_time,
                    "step": step_no,
                }
            )
            snapshot_name = f"simulate-{sim_time.replace(':', '')}.json"
            # 保存Agent活动数据
            with open(f"{self.checkpoints_folder}/{snapshot_name}", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.config, indent=2, ensure_ascii=False))
            # 保存对话数据
            with open(f"{self.checkpoints_folder}/conversation.json", "w", encoding="utf-8") as f:
                f.write(json.dumps(self.game.conversation, indent=2, ensure_ascii=False))
            judge_trace_dir = os.path.join(self.checkpoints_folder, "judge_traces")
            os.makedirs(judge_trace_dir, exist_ok=True)
            judge_trace_payload = {"sessions": []}
            if self.intervention and hasattr(self.intervention, "export_dialog_judge_trace_payload"):
                try:
                    payload = self.intervention.export_dialog_judge_trace_payload()
                    if isinstance(payload, dict):
                        judge_trace_payload = payload
                except Exception:
                    judge_trace_payload = {"sessions": []}
            with open(os.path.join(judge_trace_dir, "judge_conversation.json"), "w", encoding="utf-8") as f:
                f.write(json.dumps(judge_trace_payload, indent=2, ensure_ascii=False))
            if self.logger:
                sessions = judge_trace_payload.get("sessions", []) if isinstance(judge_trace_payload, dict) else []
                session_count = len(sessions) if isinstance(sessions, list) else 0
                self.logger.info(
                    "[DIALOG_JUDGE_TRACE_WRITE] path={} sessions_count={}".format(
                        os.path.join(judge_trace_dir, "judge_conversation.json"),
                        session_count,
                    )
                )
            forced_prompt_dir = os.path.join(self.checkpoints_folder, "forced_prompt_traces")
            os.makedirs(forced_prompt_dir, exist_ok=True)
            forced_prompt_payload = {"sessions": []}
            if self.intervention and hasattr(self.intervention, "export_forced_prompt_trace_payload"):
                try:
                    payload = self.intervention.export_forced_prompt_trace_payload()
                    if isinstance(payload, dict):
                        forced_prompt_payload = payload
                except Exception:
                    forced_prompt_payload = {"sessions": []}
            forced_sessions = (
                forced_prompt_payload.get("sessions", [])
                if isinstance(forced_prompt_payload, dict)
                else []
            )
            written_count = 0
            if isinstance(forced_sessions, list):
                for session_item in forced_sessions:
                    if not isinstance(session_item, dict):
                        continue
                    meeting = session_item.get("meeting", {}) if isinstance(session_item.get("meeting", {}), dict) else {}
                    meeting_id = str(meeting.get("meeting_id", "") or "").strip() or "meeting_unknown"
                    safe_meeting_id = (
                        meeting_id.replace(":", "")
                        .replace("/", "_")
                        .replace("\\", "_")
                    )
                    md_path = os.path.join(
                        forced_prompt_dir,
                        "{}_prompt_trace.md".format(safe_meeting_id),
                    )
                    if self.intervention and hasattr(self.intervention, "render_forced_prompt_trace_markdown"):
                        md_content = self.intervention.render_forced_prompt_trace_markdown(session_item)
                    else:
                        md_content = "# forced prompt trace\n"
                    with open(md_path, "w", encoding="utf-8") as f:
                        f.write(str(md_content or ""))
                    written_count += 1
            if self.logger:
                self.logger.info(
                    "[FORCED_PROMPT_TRACE_WRITE] dir={} files={}".format(
                        forced_prompt_dir,
                        written_count,
                    )
                )

            self.staged_eval.maybe_run_post_step_eval(
                runtime_config=copy.deepcopy(self.config),
                conversation=copy.deepcopy(self.game.conversation),
                step_no=step_no,
                sim_time=sim_time,
                snapshot_name=snapshot_name,
            )

            if stride > 0:
                timer.forward(stride)

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))


# 从存档数据中载入配置，用于断点恢复
def get_config_from_log(checkpoints_folder):
    files = sorted(os.listdir(checkpoints_folder))

    json_files = list()
    for file_name in files:
        if file_name.endswith(".json") and file_name != "conversation.json":
            json_files.append(os.path.join(checkpoints_folder, file_name))

    if len(json_files) < 1:
        return None

    with open(json_files[-1], "r", encoding="utf-8") as f:
        config = json.load(f)

    assets_root = os.path.join("assets", "village")

    start_time = datetime.datetime.strptime(config["time"], "%Y%m%d-%H:%M")
    start_time += datetime.timedelta(minutes=config["stride"])
    config["time"] = {"start": start_time.strftime("%Y%m%d-%H:%M")}
    agents = config["agents"]
    for a in agents:
        config["agents"][a]["config_path"] = os.path.join(assets_root, "agents", a.replace(" ", "_"), "agent.json")

    return config


# 为新游戏创建配置
def get_config(start_time="20240213-09:30", stride=15, agents=None):
    with open("data/config.json", "r", encoding="utf-8") as f:
        json_data = json.load(f)
        agent_config = json_data["agent"]
        intervention_config = copy.deepcopy(json_data.get("intervention", {}))
        staged_eval_config = copy.deepcopy(json_data.get("staged_eval", {}))

    assets_root = os.path.join("assets", "village")
    config = {
        "stride": stride,
        "time": {"start": start_time},
        "maze": {"path": os.path.join(assets_root, "maze.json")},
        "agent_base": agent_config,
        "agents": {},
    }
    if intervention_config:
        config["intervention"] = intervention_config
    if staged_eval_config:
        config["staged_eval"] = staged_eval_config
    for a in agents:
        config["agents"][a] = {
            "config_path": os.path.join(
                assets_root, "agents", a.replace(" ", "_"), "agent.json"
            ),
        }
    return config


load_dotenv(find_dotenv())

parser = argparse.ArgumentParser(description="console for village")
parser.add_argument("--name", type=str, default="", help="The simulation name")
parser.add_argument("--start", type=str, default="20240213-09:30", help="The starting time of the simulated ville")
parser.add_argument("--resume", action="store_true", help="Resume running the simulation")
parser.add_argument("--step", type=int, default=10, help="The simulate step")
parser.add_argument("--stride", type=int, default=10, help="The step stride in minute")
parser.add_argument("--verbose", type=str, default="debug", help="The verbose level")
parser.add_argument("--log", type=str, default="", help="Name of the log file")
args = parser.parse_args()


if __name__ == "__main__":
    checkpoints_path = "results/checkpoints"

    name = args.name
    if len(name) < 1:
        name = input("Please enter a simulation name (e.g. sim-test): ")

    resume = args.resume
    if resume:
        while not os.path.exists(f"{checkpoints_path}/{name}"):
            name = input(f"'{name}' doesn't exists, please re-enter the simulation name: ")
    else:
        while os.path.exists(f"{checkpoints_path}/{name}"):
            name = input(f"The name '{name}' already exists, please enter a new name: ")

    checkpoints_folder = f"{checkpoints_path}/{name}"

    start_time = args.start
    if resume:
        sim_config = get_config_from_log(checkpoints_folder)
        if sim_config is None:
            print("No checkpoint file found to resume running.")
            exit(0)
        start_step = sim_config["step"]
    else:
        sim_config = get_config(start_time, args.stride, personas)
        start_step = 0

    static_root = "frontend/static"

    server = SimulateServer(name, static_root, checkpoints_folder, sim_config, start_step, args.verbose, args.log)
    server.simulate(args.step, args.stride)
