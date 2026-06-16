import os
import copy
import json
import argparse
import datetime
import signal
import sys

from dotenv import load_dotenv, find_dotenv

from modules.game import create_game, get_game
from modules import utils
from modules.intervention_manager import InterventionManager
from modules.staged_eval_manager import StagedEvalManager

personas = [
    "卡布达",  # 抑郁症患者
    # "卡布达2",  # 抑郁症患者
    # "卡布达3",  # 抑郁症患者
    "金龟次郎",  # 家人（否认型父母）
    "田德莉娜",  # 好友
    "呱呱蛙",  # 邻居
    "蜻蜓队长",  # 心理医生
    "蟑螂恶霸" # 小混混
]

TRACE_STATE_SIDECAR_DIR = "trace_state_sidecars"
TRACE_STATE_SNAPSHOT_KEYS = (
    "forced_prompt_trace_state",
    "dialog_judge_trace_state",
)
_ACTIVE_SERVER = None


def _build_trace_state_sidecar_path(checkpoints_folder, state_key, snapshot_name):
    safe_key = str(state_key or "").strip() or "trace_state"
    safe_snapshot = str(snapshot_name or "").strip() or "snapshot"
    return os.path.join(
        checkpoints_folder,
        TRACE_STATE_SIDECAR_DIR,
        safe_key,
        "{}.json".format(safe_snapshot),
    )


def _write_json_file(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(json.dumps(payload, indent=2, ensure_ascii=False))


def _load_json_file(path):
    if not os.path.exists(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _strip_snapshot_trace_state(config):
    snapshot_config = copy.deepcopy(config)
    intervention_state = snapshot_config.get("intervention_state", {})
    if not isinstance(intervention_state, dict):
        return snapshot_config
    for key in TRACE_STATE_SNAPSHOT_KEYS:
        intervention_state.pop(key, None)
    return snapshot_config


def _restore_trace_state_sidecars(config, checkpoints_folder, snapshot_name):
    if not isinstance(config, dict):
        return config
    intervention_state = config.setdefault("intervention_state", {})
    if not isinstance(intervention_state, dict):
        intervention_state = {}
        config["intervention_state"] = intervention_state

    for state_key in TRACE_STATE_SNAPSHOT_KEYS:
        if intervention_state.get(state_key):
            continue
        sidecar_path = _build_trace_state_sidecar_path(checkpoints_folder, state_key, snapshot_name)
        payload = _load_json_file(sidecar_path)
        if isinstance(payload, dict):
            intervention_state[state_key] = payload
    return config


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
        self.stop_requested = False

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

    def _checkpointing_cfg(self):
        cfg = self.config.get("checkpointing", {}) or {}
        return cfg if isinstance(cfg, dict) else {}

    def _should_write_full_snapshot(self, step_no, final_step=False, staged_eval_triggered=False):
        cfg = self._checkpointing_cfg()
        if not bool(cfg.get("enabled", False)):
            return True
        if str(cfg.get("mode", "") or "").strip() != "consult_sparse":
            return True

        flags = {}
        if self.intervention and hasattr(self.intervention, "get_step_flags"):
            try:
                flags = self.intervention.get_step_flags()
            except Exception:
                flags = {}
        if bool(flags.get("forced_consult_happened", False)):
            return True
        if bool(flags.get("treatment_completed_this_step", False)):
            return True
        if bool(staged_eval_triggered) and bool(cfg.get("write_on_staged_eval_trigger", True)):
            return True

        periodic_every_steps = 0
        try:
            periodic_every_steps = int(cfg.get("periodic_every_steps", 0) or 0)
        except Exception:
            periodic_every_steps = 0
        if periodic_every_steps > 0 and int(step_no) % periodic_every_steps == 0:
            return True
        if bool(final_step) and bool(cfg.get("write_final_snapshot", True)):
            return True
        return False

    def _write_full_snapshot(self, snapshot_name, snapshot_stem, snapshot_config, judge_trace_payload, forced_prompt_payload):
        _write_json_file(
            _build_trace_state_sidecar_path(
                self.checkpoints_folder,
                "dialog_judge_trace_state",
                snapshot_stem,
            ),
            judge_trace_payload,
        )
        _write_json_file(
            _build_trace_state_sidecar_path(
                self.checkpoints_folder,
                "forced_prompt_trace_state",
                snapshot_stem,
            ),
            forced_prompt_payload,
        )
        _write_json_file(
            f"{self.checkpoints_folder}/{snapshot_name}",
            snapshot_config,
        )

    def simulate(self, step, stride=0):
        timer = utils.get_timer()
        try:
            for i in range(self.start_step, self.start_step + step):
                if self.stop_requested:
                    raise SystemExit(1)
                if self.staged_eval and hasattr(self.staged_eval, "begin_step"):
                    self.staged_eval.begin_step()
                self.staged_eval.poll()
                if self.staged_eval and hasattr(self.staged_eval, "should_auto_stop_after_t4_done"):
                    if self.staged_eval.should_auto_stop_after_t4_done():
                        self.logger.info("[SIMULATION_AUTO_STOP] reason=t4_done")
                        break
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
                snapshot_stem = os.path.splitext(snapshot_name)[0]
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
                _write_json_file(
                    os.path.join(judge_trace_dir, "judge_conversation.json"),
                    judge_trace_payload,
                )
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
                _write_json_file(
                    f"{self.checkpoints_folder}/conversation.json",
                    self.game.conversation,
                )

                self.staged_eval.maybe_run_post_step_eval(
                    runtime_config=copy.deepcopy(self.config),
                    conversation=copy.deepcopy(self.game.conversation),
                    step_no=step_no,
                    sim_time=sim_time,
                    snapshot_name=snapshot_name,
                )
                staged_eval_triggered = bool(
                    self.staged_eval
                    and hasattr(self.staged_eval, "was_trigger_enqueued_this_step")
                    and self.staged_eval.was_trigger_enqueued_this_step()
                )
                if staged_eval_triggered and self.intervention and hasattr(self.intervention, "mark_step_flag"):
                    self.intervention.mark_step_flag("staged_eval_triggered_this_step", True)

                snapshot_config = _strip_snapshot_trace_state(self.config)
                final_step = i == (self.start_step + step - 1)
                if self._should_write_full_snapshot(
                    step_no=step_no,
                    final_step=final_step,
                    staged_eval_triggered=staged_eval_triggered,
                ):
                    self._write_full_snapshot(
                        snapshot_name=snapshot_name,
                        snapshot_stem=snapshot_stem,
                        snapshot_config=snapshot_config,
                        judge_trace_payload=judge_trace_payload,
                        forced_prompt_payload=forced_prompt_payload,
                    )

                if stride > 0:
                    timer.forward(stride)
            self.staged_eval.drain()
        except BaseException:
            self.staged_eval.abort_pending_work(reason="simulation_aborted")
            raise

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))

    def request_stop(self, reason=""):
        self.stop_requested = True
        if self.logger:
            self.logger.warning("[SIMULATION_STOP] reason={}".format(str(reason or "")))
        self.staged_eval.abort_pending_work(reason=reason or "stop_requested")


# 从存档数据中载入配置，用于断点恢复
def get_config_from_log(checkpoints_folder):
    files = sorted(os.listdir(checkpoints_folder))

    json_files = list()
    for file_name in files:
        if file_name.startswith("simulate-") and file_name.endswith(".json"):
            json_files.append(os.path.join(checkpoints_folder, file_name))

    if len(json_files) < 1:
        return None

    snapshot_path = json_files[-1]
    snapshot_name = os.path.splitext(os.path.basename(snapshot_path))[0]
    with open(snapshot_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    config = _restore_trace_state_sidecars(config, checkpoints_folder, snapshot_name)
    if "checkpointing" not in config:
        try:
            with open("data/config.json", "r", encoding="utf-8") as defaults_f:
                defaults = json.load(defaults_f)
            checkpointing_cfg = defaults.get("checkpointing", {})
            if isinstance(checkpointing_cfg, dict) and checkpointing_cfg:
                config["checkpointing"] = copy.deepcopy(checkpointing_cfg)
        except Exception:
            pass

    assets_root = os.path.join("assets", "village")

    start_time = datetime.datetime.strptime(config["time"], "%Y%m%d-%H:%M")
    start_time += datetime.timedelta(minutes=config["stride"])
    config["time"] = {"start": start_time.strftime("%Y%m%d-%H:%M")}
    agents = config["agents"]
    for a in agents:
        config["agents"][a]["config_path"] = os.path.join(assets_root, "agents", a.replace(" ", "_"), "agent.json")

    return config


# 为新游戏创建配置
def get_config(start_time="20240213-09:30", stride=15, agents=None, runtime_config_path=""):
    config_path = str(runtime_config_path or "").strip() or "data/config.json"
    with open(config_path, "r", encoding="utf-8") as f:
        json_data = json.load(f)
    if {"agent_base", "maze", "agents"} <= set(json_data.keys()):
        config = copy.deepcopy(json_data)
        config["stride"] = stride
        config["time"] = {"start": start_time}
        return config

    agent_config = json_data["agent"]
    intervention_config = copy.deepcopy(json_data.get("intervention", {}))
    staged_eval_config = copy.deepcopy(json_data.get("staged_eval", {}))
    checkpointing_config = copy.deepcopy(json_data.get("checkpointing", {}))

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
    if checkpointing_config:
        config["checkpointing"] = checkpointing_config
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
parser.add_argument("--runtime-config", type=str, default="", help="Path to a per-run runtime config json")
args = parser.parse_args()


def _handle_termination(signum, _frame):
    del _frame
    if _ACTIVE_SERVER is not None:
        _ACTIVE_SERVER.request_stop(reason="signal_{}".format(signum))
    raise SystemExit(1)


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
        sim_config = get_config(start_time, args.stride, personas, runtime_config_path=args.runtime_config)
        start_step = 0

    static_root = "frontend/static"

    server = SimulateServer(name, static_root, checkpoints_folder, sim_config, start_step, args.verbose, args.log)
    _ACTIVE_SERVER = server
    signal.signal(signal.SIGTERM, _handle_termination)
    signal.signal(signal.SIGINT, _handle_termination)
    try:
        server.simulate(args.step, args.stride)
    finally:
        _ACTIVE_SERVER = None
