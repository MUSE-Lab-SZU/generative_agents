"""generative_agents.game"""

import os
import copy

from modules.utils import GenerativeAgentsMap, GenerativeAgentsKey
from modules import utils
from .maze import Maze
from .agent import Agent



class Game:
    """管理模拟世界中的地图、角色、对话和全局运行状态"""

    def __init__(self, name, static_root, config, conversation, logger=None):
        """根据配置初始化游戏世界、地图和所有 Agent"""
        self.name = name
        self.static_root = static_root
        self.record_interval = config.get("record_interval", 30)
        self.logger = logger or utils.IOLogger()

        self.maze = Maze(self.load_static(config["maze"]["path"]), self.logger) # 加载frontend/static/assets/village/maze.json文件
        self.conversation = conversation
        self.agents = {}
        if "agent_base" in config:
            agent_base = config["agent_base"]
        else:
            agent_base = {}
        storage_root = os.path.join(f"results/checkpoints/{name}", "storage")
        if not os.path.isdir(storage_root):
            os.makedirs(storage_root)
            
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
            self.agents[name] = Agent(agent_config, self.maze, self.conversation, self.logger)

    def get_agent(self, name):
        """按名称获取当前游戏中的 Agent"""
        return self.agents[name]

    def agent_think(self, name, status):
        """驱动指定 Agent 完成一次思考，并返回行动计划和状态摘要"""
        agent = self.get_agent(name)
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
        ) > self.record_interval:
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
        """从 static_root 下读取静态 JSON 配置"""
        return utils.load_dict(os.path.join(self.static_root, path))

    def reset_game(self):
        """重置所有 Agent 的运行时组件并记录初始状态"""
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
