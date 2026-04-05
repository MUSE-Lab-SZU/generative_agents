"""generative_agents.game"""

import os
import copy

from modules.utils import GenerativeAgentsMap, GenerativeAgentsKey
from modules import utils
from .maze import Maze
from .agent import Agent


class Game:
    """The Game"""

    def __init__(self, name, static_root, config, conversation, logger=None):
        self.name = name
        self.static_root = static_root
        self.record_iterval = config.get("record_iterval", 30)
        self.logger = logger or utils.IOLogger()
        self.intervention = None
        self.maze = Maze(self.load_static(config["maze"]["path"]), self.logger)
        self.conversation = conversation
        self.agents = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        storage_root = os.path.join(f"results/checkpoints/{name}", "storage")
        if not os.path.isdir(storage_root):
            os.makedirs(storage_root)
        global_dynamic_cfg = copy.deepcopy(
            ((config.get("intervention", {}) or {}).get("depression_dynamic", {}) or {})
        )
        for name, agent in config["agents"].items():
            raw_agent_cfg = self.load_static(agent["config_path"])
            agent_config = utils.update_dict(
                copy.deepcopy(agent_base), raw_agent_cfg
            )
            agent_config = utils.update_dict(agent_config, agent)
            agent_config["_raw"] = raw_agent_cfg

            agent_config["storage_root"] = os.path.join(storage_root, name)
            config_path = str(agent.get("config_path", "") or "")
            agent_dir_rel = os.path.dirname(config_path)
            agent_config["agent_dir"] = os.path.normpath(
                os.path.join(self.static_root, agent_dir_rel)
            )
            agent_config["depression_dynamic_global"] = copy.deepcopy(global_dynamic_cfg)
            self.agents[name] = Agent(agent_config, self.maze, self.conversation, self.logger)

    def get_agent(self, name):
        return self.agents[name]

    def set_intervention_manager(self, manager):
        self.intervention = manager
        for agent in self.agents.values():
            agent.intervention = manager

    def agent_think(self, name, status):
        agent = self.get_agent(name)
        if self.intervention:
            self.intervention.before_agent_think(agent, self.agents, utils.get_timer().get_date())
        plan = agent.think(status, self.agents)
        info = {
            "currently": agent.scratch.currently,
            "associate": agent.associate.abstract(),
            "concepts": {c.node_id: c.abstract() for c in agent.concepts},
            "chats": [
                {"name": "self" if n == agent.name else n, "chat": c}
                for n, c in agent.chats
            ],
            "action": agent.action.abstract(),
            "schedule": agent.schedule.abstract(),
            "address": agent.get_tile().get_address(as_list=False),
        }
        if (
            utils.get_timer().daily_duration() - agent.last_record
        ) > self.record_iterval:
            info["record"] = True
            agent.last_record = utils.get_timer().daily_duration()
        else:
            info["record"] = False
        if agent.llm_available():
            info["llm"] = agent._llm.get_summary()
        title = "{}.summary @ {}".format(
            name, utils.get_timer().get_date("%Y%m%d-%H:%M:%S")
        )
        self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))
        return {"plan": plan, "info": info}

    def load_static(self, path):
        return utils.load_dict(os.path.join(self.static_root, path))
    
    # def load_static(self, path):
    #     full_path = os.path.join(self.static_root, path)
    #     print(f"[load_static] path={path}")
    #     print(f"[load_static] full_path={full_path}")

    #     # 检查文件是否存在、大小
    #     if not os.path.exists(full_path):
    #         raise FileNotFoundError(f"Config file not found: {full_path}")
    #     print(f"[load_static] file_size={os.path.getsize(full_path)} bytes")

    #     return utils.load_dict(full_path)

    def reset_game(self):
        for a_name, agent in self.agents.items():
            agent.reset()
            title = "{}.reset".format(a_name)
            self.logger.info("\n{}\n{}\n".format(utils.split_line(title), agent))


def create_game(name, static_root, config, conversation, logger=None):
    """Create the game"""

    utils.set_timer(**config.get("time", {}))
    GenerativeAgentsMap.set(GenerativeAgentsKey.GAME, Game(name, static_root, config, conversation, logger=logger))
    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)


def get_game():
    """Get the gloabl game"""

    return GenerativeAgentsMap.get(GenerativeAgentsKey.GAME)
