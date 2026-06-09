"""generative_agents.agent"""

import os
import math
import random
import datetime
import copy

from modules import memory, prompt, utils
from modules.depression import DepressionSimulationEngine
from modules.model.llm_model import create_llm_model
from modules.memory.associate import Concept


class Agent:
    """游戏中的核心角色对象。

    阅读这个类时，可以把它拆成四层来看：
    1. `memory / schedule / spatial / associate`：角色“记住了什么”。
    2. `scratch + completion()`：角色“如何向 LLM 提问”。
    3. `think() / percept() / make_plan()`：角色“每一 tick 怎么做决定”。
    4. `depression_dynamic`：一个可选的动态抑郁表现插件，只在少数 prompt
       生成节点上插入额外约束，而不直接改动基础行动系统。
    """

    def __init__(self, config, maze, conversation, logger):
        """初始化角色实例。

        Args:
            config (dict): 角色配置字典，由上层组装
            maze: 当前游戏使用的迷宫对象，用于查询 tile、地址和路径等空间信息。
            conversation: 对话管理器或对话上下文对象，负责角色间的聊天记录与
                对话流程协作。
            logger: 日志记录器，用于输出角色思考、行动和 LLM 调用过程中的调试
                与运行日志。
        """
        self.name = config["name"]
        self.maze = maze
        self.conversation = conversation
        self._llm = None
        self.logger = logger

        # agent config
        self.percept_config = config["percept"] # 角色怎么“看见/注意到”周围环境，box 表示感知范围是一个方形区域
        self.think_config = config["think"] # 配置角色使用的 LLM
        self.chat_iter = config["chat_iter"] # 一次对话流程里最多允许多少轮对话

        # memory
        self.spatial = memory.Spatial(**config["spatial"]) # 空间记忆：知道哪些地点、如何根据提示词找到地点
        self.schedule = memory.Schedule(**config["schedule"]) # 日程记忆：保存/生成/拆分/修订当天计划
        self.associate = memory.Associate( # 角色的联想记忆/长期记忆：存储并检索事件、想法和聊天
            os.path.join(config["storage_root"], "associate"), **config["associate"]
        )
        self.concepts, self.chats = [], config.get("chats", []) # 当前关注概念/瞬时记忆 + 最近聊天缓存

        # prompt
        self.scratch = prompt.Scratch(self.name, config["currently"], config["scratch"])
        self.depression_dynamic = self._init_depression_dynamic(config) # 给当前角色创建 动态抑郁 实例

        # status 初始化角色状态
        status = {"poignancy": 0} # 角色当前积累的“情绪/事件冲击值”。这个值后面会影响角色是否进入 reflect() 反思流程。
        self.status = utils.update_dict(status, config.get("status", {})) # 用 config 里的状态去覆盖默认状态
        self.plan = config.get("plan", {})

        # record 角色上一次“记录摘要信息”的时间点，按当天经过了多少分钟来表示
        self.last_record = utils.get_timer().daily_duration()

        # action and events 这个角色当前正在做什么，以及他对应的物体/地点事件是什么。
        if "action" in config: # 恢复旧状态
            self.action = memory.Action.from_dict(config["action"])
            tiles = self.maze.get_address_tiles(self.get_event().address)
            config["coord"] = random.choice(list(tiles))
        else: # 如果没有历史动作，就根据初始坐标现场创建一个默认动作
            tile = self.maze.tile_at(config["coord"])
            address = tile.get_address("game_object", as_list=True)
            self.action = memory.Action(
                memory.Event(self.name, address=address),
                memory.Event(address[-1], address=address),
            )

        # update maze 把角色同步到地图
        self.coord, self.path = None, None
        self.move(config["coord"], config.get("path"))
        if self.coord is None:
            self.coord = config["coord"]

    def abstract(self):
        """
        把当前 Agent 的核心状态整理成一个“适合查看/打印/调试”的摘要字典。
        """
        # 整理基础字段
        des = {
            "name": self.name,
            "currently": self.scratch.currently,
            "tile": self.maze.tile_at(self.coord).abstract(),
            "status": self.status,
            "concepts": {c.node_id: c.abstract() for c in self.concepts},
            "chats": self.chats,
            "action": self.action.abstract(),
            "associate": self.associate.abstract(),
        }
        # 按条件补充可选字段
        if self.schedule.scheduled():
            des["schedule"] = self.schedule.abstract()
        if self.depression_dynamic:
            try:
                dyn = self.depression_dynamic.get_current_state_info()
                des["depression_dynamic"] = {
                    "stage_id": dyn.get("current_stage", {}).get("id", ""),
                    "stage_label": dyn.get("current_stage", {}).get("label", ""),
                    "interaction_count": dyn.get("interaction_count", 0),
                }
            except Exception:
                pass
        if self.llm_available():
            des["llm"] = self._llm.get_summary()
        # if self.plan.get("path"):
        #     des["path"] = "-".join(
        #         ["{},{}".format(c[0], c[1]) for c in self.plan["path"]]
        #     )
        return des

    def __str__(self):
        """
        把 Agent 对象转换成一个适合打印和日志输出的可读字符串。
        """
        return utils.dump_dict(self.abstract())

    def reset(self):
        """
        按需初始化 Agent 在思考、对话等流程中使用的 LLM 实例。
        """
        if not self._llm:
            self._llm = create_llm_model(self.think_config["llm"])
        self._initialize_depression_graph_window()

    def completion(self, func_hint, *args, **kwargs):
        """统一的 prompt -> LLM -> 结果回收入口。
        Agent 所有 LLM 推理任务的统一中间层。
        Args:
            func_hint:想调用哪一种 prompt/推理任务,最终会映射到 Scratch 里的某个方法
            *args 和 **kwargs:传给对应 Scratch.prompt_xxx(...) 的参数

        这里是阅读 `Agent` 时最关键的“总闸门”：
        - `Scratch.prompt_xxx(...)` 负责组织 prompt；
        - `self._llm.completion(...)` 负责真正请求模型；
        - 动态抑郁模块只在特定 `func_hint` 上做“前后置增强”。
        """
        assert hasattr(
            self.scratch, "prompt_" + func_hint
        ), "Can not find func prompt_{} from scratch".format(func_hint)
        prompt_kwargs = dict(kwargs)
        depression_chat_ctx = None
        if func_hint == "generate_chat":
            # 对话生成是动态抑郁模块最重要的挂载点：
            # 先 preview 当前轮应该呈现什么“主诉节点/情绪/偏差”，
            # 再把生成好的提示块注入原始聊天 prompt。
            prompt_kwargs, depression_chat_ctx = self._prepare_depression_generate_chat(
                args=args,
                kwargs=prompt_kwargs,
            )
        elif func_hint == "reflect_insights":
            # 反思阶段只注入“当前主诉节点的简单摘要”，
            # 作用比聊天时更弱，主要防止反思文本脱离人设。
            prompt_kwargs = self._prepare_depression_reflect_insights(prompt_kwargs)
        func = getattr(self.scratch, "prompt_" + func_hint)
        raw_res = func(*args, **prompt_kwargs)
        if hasattr(raw_res, "_asdict"):
            res = raw_res._asdict()
        elif isinstance(raw_res, dict):
            res = dict(raw_res)
        else:
            raise TypeError(
                "scratch prompt_{} must return dict or namedtuple-like object, got {}".format(
                    func_hint, type(raw_res).__name__
                )
            )
        title, msg = "{}.{}".format(self.name, func_hint), {}
        output = res.get("failsafe")
        if self.llm_available():
            self.logger.info("{} -> {}".format(self.name, func_hint))
            output = self._llm.completion(**res)
            msg = {"<PROMPT>": "\n" + res["prompt"] + "\n"}
            msg.update({"response": output})
        self.logger.debug(utils.block_msg(title, msg))
        if func_hint == "generate_chat" and depression_chat_ctx:
            self._commit_depression_generate_chat(depression_chat_ctx, output)
        return output

    def think(self, status, agents):
        """单轮主循环：移动 -> 排程 -> 感知 -> 决策 -> 反思。"""
        events = self.move(status["coord"], status.get("path"))
        plan, _ = self.make_schedule()

        if (plan["describe"] == "sleeping" or "睡" in plan["describe"]) and self.is_awake():
            self.logger.info("{} is going to sleep...".format(self.name))
            address = self.spatial.find_address("睡觉", as_list=True)
            tiles = self.maze.get_address_tiles(address)
            coord = random.choice(list(tiles))
            events = self.move(coord)
            self.action = memory.Action(
                memory.Event(self.name, "正在", "睡觉", address=address, emoji="😴"),
                memory.Event(
                    address[-1],
                    "被占用",
                    self.name,
                    address=address,
                    emoji="🛌",
                ),
                duration=plan["duration"],
                start=utils.get_timer().daily_time(plan["start"]),
            )
        if self.is_awake():
            self.percept()
            self.make_plan(agents)
            self.reflect()
        else:
            if self.action.finished():
                self.action = self._determine_action()

        emojis = {}
        if self.action:
            emojis[self.name] = {"emoji": self.get_event().emoji, "coord": self.coord}
        for eve, coord in events.items():
            if eve.subject in agents:
                continue
            emojis[":".join(eve.address)] = {"emoji": eve.emoji, "coord": coord}
        self.plan = {
            "name": self.name,
            "path": self.find_path(agents),
            "emojis": emojis,
        }
        return self.plan

    def move(self, coord, path=None):
        """更新角色在迷宫中的位置，并同步 tile / object 事件。"""
        events = {}

        def _update_tile(coord):
            tile = self.maze.tile_at(coord)
            if not self.action:
                return {}
            if not tile.update_events(self.get_event()):
                tile.add_event(self.get_event())
            obj_event = self.get_event(False)
            if obj_event:
                self.maze.update_obj(coord, obj_event)
            return {e: coord for e in tile.get_events()}

        if self.coord and self.coord != coord:
            tile = self.get_tile()
            tile.remove_events(subject=self.name)
            if tile.has_address("game_object"):
                addr = tile.get_address("game_object")
                self.maze.update_obj(
                    self.coord, memory.Event(addr[-1], address=addr)
                )
            events.update({e: self.coord for e in tile.get_events()})
        if not path:
            events.update(_update_tile(coord))
        self.coord = coord
        self.path = path or []

        return events

    def make_schedule(self):
        """生成或分解当天计划。

        这个函数主要还是基础 generative agents 逻辑；
        动态抑郁模块并不会直接改排程，因此阅读时可以把它当作
        “人格/日程层”，与“症状表达层”区分开。
        """
        if not self.schedule.scheduled():
            self.logger.info("{} is making schedule...".format(self.name))
            # update currently
            if self.associate.index.nodes_num > 0:
                self.associate.cleanup_index()
                focus = [
                    f"{self.name} 在 {utils.get_timer().daily_format_cn()} 的计划。",
                    f"在 {self.name} 的生活中，重要的近期事件。",
                ]
                retrieved = self.associate.retrieve_focus(focus)
                self.logger.info(
                    "{} retrieved {} concepts".format(self.name, len(retrieved))
                )
                if retrieved:
                    plan = self.completion("retrieve_plan", retrieved)
                    thought = self.completion("retrieve_thought", retrieved)
                    self.scratch.currently = self.completion(
                        "retrieve_currently", plan, thought
                    )
            # make init schedule
            self.schedule.create = utils.get_timer().get_date()
            wake_up = self.completion("wake_up")
            init_schedule = self.completion("schedule_init", wake_up)
            # make daily schedule
            hours = [f"{i}:00" for i in range(24)]
            # seed = [(h, "sleeping") for h in hours[:wake_up]]
            seed = [(h, "睡觉") for h in hours[:wake_up]]
            seed += [(h, "") for h in hours[wake_up:]]
            schedule = {}
            for _ in range(self.schedule.max_try):
                schedule = {h: s for h, s in seed[:wake_up]}
                schedule.update(
                    self.completion("schedule_daily", wake_up, init_schedule)
                )
                if len(set(schedule.values())) >= self.schedule.diversity:
                    break

            def _to_duration(date_str):
                return utils.daily_duration(utils.to_date(date_str, "%H:%M"))

            schedule = {_to_duration(k): v for k, v in schedule.items()}
            starts = list(sorted(schedule.keys()))
            for idx, start in enumerate(starts):
                end = starts[idx + 1] if idx + 1 < len(starts) else 24 * 60
                self.schedule.add_plan(schedule[start], end - start)
            schedule_time = utils.get_timer().time_format_cn(self.schedule.create)
            thought = "这是 {} 在 {} 的计划：{}".format(
                self.name, schedule_time, "；".join(init_schedule)
            )
            event = memory.Event(
                self.name,
                "计划",
                schedule_time,
                describe=thought,
                address=self.get_tile().get_address(),
            )
            self._add_concept(
                "thought",
                event,
                expire=self.schedule.create + datetime.timedelta(days=30),
            )
        # decompose current plan
        plan, _ = self.schedule.current_plan()
        if self.schedule.decompose(plan):
            decompose_schedule = self.completion(
                "schedule_decompose", plan, self.schedule
            )
            decompose, start = [], plan["start"]
            for describe, duration in decompose_schedule:
                decompose.append(
                    {
                        "idx": len(decompose),
                        "describe": describe,
                        "start": start,
                        "duration": duration,
                    }
                )
                start += duration
            plan["decompose"] = decompose
        return self.schedule.current_plan()

    def revise_schedule(self, event, start, duration):
        """用新的事件替换当前行动，并按需修订当前分解计划。"""
        self.action = memory.Action(event, start=start, duration=duration)
        plan, _ = self.schedule.current_plan()
        if len(plan["decompose"]) > 0:
            plan["decompose"] = self.completion(
                "schedule_revise", self.action, self.schedule
            )

    def percept(self):
        """感知周围环境并把新事件写入联想记忆。"""
        scope = self.maze.get_scope(self.coord, self.percept_config)
        # add spatial memory
        for tile in scope:
            if tile.has_address("game_object"):
                self.spatial.add_leaf(tile.address)
        events, arena = {}, self.get_tile().get_address("arena")
        # gather events in scope
        for tile in scope:
            if not tile.events or tile.get_address("arena") != arena:
                continue
            dist = math.dist(tile.coord, self.coord)
            for event in tile.get_events():
                if dist < events.get(event, float("inf")):
                    events[event] = dist
        events = list(sorted(events.keys(), key=lambda k: events[k]))
        # get concepts
        self.concepts, valid_num = [], 0
        for idx, event in enumerate(events[: self.percept_config["att_bandwidth"]]):
            recent_nodes = (
                self.associate.retrieve_events() + self.associate.retrieve_chats()
            )
            recent_nodes = set(n.describe for n in recent_nodes)
            if event.get_describe() not in recent_nodes:
                if event.object == "idle" or event.object == "空闲":
                    node = Concept.from_event(
                        "idle_" + str(idx), "event", event, poignancy=1
                    )
                else:
                    valid_num += 1
                    node_type = "chat" if event.fit(self.name, "对话") else "event"
                    node = self._add_concept(node_type, event)
                    self.status["poignancy"] += node.poignancy
                self.concepts.append(node)
        self.concepts = [c for c in self.concepts if c.event.subject != self.name]
        self.logger.info(
            "{} percept {}/{} concepts".format(self.name, valid_num, len(self.concepts))
        )

    def make_plan(self, agents):
        """根据感知结果和当前行动状态，决定是否互动或生成下一步行动。"""
        if self._reaction(agents):
            return
        if self.path:
            return
        if self.action.finished():
            self.action = self._determine_action()

    # create action && object events
    def make_event(self, subject, describe, address):
        """把自然语言行动描述整理成可写入地图和记忆的 Event 对象。"""
        # emoji = self.completion("describe_emoji", describe)
        # return self.completion(
        #     "describe_event", subject, subject + describe, address, emoji
        # )

        e_describe = describe.replace("(", "").replace(")", "").replace("<", "").replace(">", "")
        if e_describe.startswith(subject + "此时"):
            e_describe = e_describe[len(subject + "此时"):]
        if e_describe.startswith(subject):
            e_describe = e_describe[len(subject):]
        event = memory.Event(
            subject, "此时", e_describe, describe=describe, address=address
        )
        return event

    def reflect(self):
        """当 poignancy 积累到阈值后，对近期事件做抽象反思。"""
        def _add_thought(thought, evidence=None):
            # event = self.completion(
            #     "describe_event",
            #     self.name,
            #     thought,
            #     address=self.get_tile().get_address(),
            # )
            event = self.make_event(self.name, thought, self.get_tile().get_address())
            return self._add_concept("thought", event, filling=evidence)

        if self.status["poignancy"] < self.think_config["poignancy_max"]:
            return
        nodes = self.associate.retrieve_events() + self.associate.retrieve_thoughts()
        if not nodes:
            return
        self.logger.info(
            "{} reflect(P{}/{}) with {} concepts...".format(
                self.name,
                self.status["poignancy"],
                self.think_config["poignancy_max"],
                len(nodes),
            )
        )
        nodes = sorted(nodes, key=lambda n: n.access, reverse=True)[
            : self.associate.max_importance
        ]
        # summary thought
        reflection_entries = []
        focus = self.completion("reflect_focus", nodes, 3)
        retrieved = self.associate.retrieve_focus(focus, reduce_all=False)
        for r_nodes in retrieved.values():
            thoughts = self.completion("reflect_insights", r_nodes, 5)
            for thought, evidence in thoughts:
                node = _add_thought(thought, evidence)
                reflection_entries.append(
                    {
                        "thought": thought,
                        "evidence": evidence,
                        "node_id": getattr(node, "node_id", ""),
                    }
                )
        # summary chats
        if self.chats:
            recorded, evidence = set(), []
            for name, _ in self.chats:
                if name == self.name or name in recorded:
                    continue
                res = self.associate.retrieve_chats(name)
                if res and len(res) > 0:
                    node = res[-1]
                    evidence.append(node.node_id)
            thought = self.completion("reflect_chat_planing", self.chats)
            plan_thought = f"对于 {self.name} 的计划：{thought}"
            node = _add_thought(plan_thought, evidence)
            reflection_entries.append(
                {
                    "thought": plan_thought,
                    "evidence": evidence,
                    "node_id": getattr(node, "node_id", ""),
                }
            )
            thought = self.completion("reflect_chat_memory", self.chats)
            memory_thought = f"{self.name} {thought}"
            node = _add_thought(memory_thought, evidence)
            reflection_entries.append(
                {
                    "thought": memory_thought,
                    "evidence": evidence,
                    "node_id": getattr(node, "node_id", ""),
                }
            )
        self._commit_depression_reflection(focus, reflection_entries)
        self.status["poignancy"] = 0
        self.chats = []

    def find_path(self, agents):
        """根据当前行动的目标地址，为 Agent 计算一条可移动路径。"""
        address = self.get_event().address
        if self.path:
            return self.path
        if address == self.get_tile().get_address():
            return []
        if address[0] == "<waiting>":
            return []
        if address[0] == "<persona>":
            target_tiles = self.maze.get_around(agents[address[1]].coord)
        else:
            target_tiles = self.maze.get_address_tiles(address)
        if tuple(self.coord) in target_tiles:
            return []

        # filter tile with self event
        def _ignore_target(t_coord):
            if list(t_coord) == list(self.coord):
                return True
            events = self.maze.tile_at(t_coord).get_events()
            if any(e.subject in agents for e in events):
                return True
            return False

        target_tiles = [t for t in target_tiles if not _ignore_target(t)]
        if not target_tiles:
            return []
        if len(target_tiles) >= 4:
            target_tiles = random.sample(target_tiles, 4)
        pathes = {t: self.maze.find_path(self.coord, t) for t in target_tiles}
        target = min(pathes, key=lambda p: len(pathes[p]))
        return pathes[target][1:]

    def _determine_action(self):
        """根据当前日程和空间记忆，生成下一段具体行动及物体事件。"""
        self.logger.info("{} is determining action...".format(self.name))
        plan, de_plan = self.schedule.current_plan()
        describes = [plan["describe"], de_plan["describe"]]
        address = self.spatial.find_address(describes[0], as_list=True)
        if not address:
            tile = self.get_tile()
            kwargs = {
                "describes": describes,
                "spatial": self.spatial,
                "address": tile.get_address("world", as_list=True),
            }
            kwargs["address"].append(
                self.completion("determine_sector", **kwargs, tile=tile)
            )
            arenas = self.spatial.get_leaves(kwargs["address"])
            if len(arenas) == 1:
                kwargs["address"].append(arenas[0])
            else:
                kwargs["address"].append(self.completion("determine_arena", **kwargs))
            objs = self.spatial.get_leaves(kwargs["address"])
            if len(objs) == 1:
                kwargs["address"].append(objs[0])
            elif len(objs) > 1:
                kwargs["address"].append(self.completion("determine_object", **kwargs))
            address = kwargs["address"]

        event = self.make_event(self.name, describes[-1], address)
        obj_describe = self.completion("describe_object", address[-1], describes[-1])
        obj_event = self.make_event(address[-1], obj_describe, address)

        event.emoji = f"{de_plan['describe']}"

        return memory.Action(
            event,
            obj_event,
            duration=de_plan["duration"],
            start=utils.get_timer().daily_time(de_plan["start"]),
        )

    def _reaction(self, agents=None, ignore_words=None):
        """从当前感知概念中选择关注对象，并尝试聊天或等待对方。"""
        focus = None
        ignore_words = ignore_words or ["空闲"]

        def _focus(concept):
            return concept.event.subject in agents

        def _ignore(concept):
            return any(i in concept.describe for i in ignore_words)

        if agents:
            priority = [i for i in self.concepts if _focus(i)]
            if priority:
                focus = random.choice(priority)
        if not focus:
            priority = [i for i in self.concepts if not _ignore(i)]
            if priority:
                focus = random.choice(priority)
        if not focus or focus.event.subject not in agents:
            return
        other, focus = agents[focus.event.subject], self.associate.get_relation(focus)

        if self._chat_with(other, focus):
            return True
        if self._wait_other(other, focus):
            return True
        return False

    def _skip_react(self, other):
        """判断当前双方状态是否不适合触发社交反应。"""
        def _skip(event):
            if not event.address or "sleeping" in event.get_describe(False) or "睡觉" in event.get_describe(False):
                return True
            if event.predicate == "待开始":
                return True
            return False

        if utils.get_timer().daily_duration(mode="hour") >= 23:
            return True
        if _skip(self.get_event()) or _skip(other.get_event()):
            return True
        return False

    def _chat_with(self, other, focus):
        """尝试与另一个 Agent 展开多轮对话。"""
        if len(self.schedule.daily_schedule) < 1 or len(other.schedule.daily_schedule) < 1:
            # initializing
            return False
        if self._skip_react(other):
            return False
        if other.path:
            return False
        if self.get_event().fit(predicate="对话") or other.get_event().fit(predicate="对话"):
            return False

        chats = self.associate.retrieve_chats(other.name)
        if chats:
            delta = utils.get_timer().get_delta(chats[0].create)
            self.logger.info(
                "retrieved chat between {} and {}({} min):\n{}".format(
                    self.name, other.name, delta, chats[0]
                )
            )
            if delta < 60:
                return False

        if not self.completion("decide_chat", self, other, focus, chats):
            return False

        self.logger.info("{} decides chat with {}".format(self.name, other.name))
        start, chats = utils.get_timer().get_date(), []
        relations = [
            self.completion("summarize_relation", self, other.name),
            other.completion("summarize_relation", other, self.name),
        ]

        for i in range(self.chat_iter):
            # 发起方先说。此处的 `generate_chat` 会经过 completion()，
            # 因而可能被动态抑郁模块注入额外的“说话约束层”。
            text = self.completion(
                "generate_chat", self, other, relations[0], chats
            )

            if i > 0:
                # 对于发起对话的Agent，从第2轮对话开始，检查是否出现“复读”现象
                end = self.completion(
                    "generate_chat_check_repeat", self, chats, text
                )
                if end:
                    break

                # 对于发起对话的Agent，从第2轮对话开始，检查话题是否结束
                chats.append((self.name, text))
                end = self.completion(
                    "decide_chat_terminate", self, other, chats
                )
                if end:
                    break
            else :
                chats.append((self.name, text))

            text = other.completion(
                "generate_chat", other, self, relations[1], chats
            )
            if i > 0:
                # 对于响应对话的Agent，从第2轮开始，检查是否出现“复读”现象
                end = self.completion(
                    "generate_chat_check_repeat", other, chats, text
                )
                if end:
                    break

            chats.append((other.name, text))

            # 对于响应对话的Agent，从第1轮开始，检查话题是否结束
            end = other.completion(
                "decide_chat_terminate", other, self, chats
            )
            if end:
                break

        key = utils.get_timer().get_date("%Y%m%d-%H:%M")
        if key not in self.conversation.keys():
            self.conversation[key] = []
        self.conversation[key].append({f"{self.name} -> {other.name} @ {'，'.join(self.get_event().address)}": chats})

        self.logger.info(
            "{} and {} has chats\n  {}".format(
                self.name,
                other.name,
                "\n  ".join(["{}: {}".format(n, c) for n, c in chats]),
            )
        )
        chat_summary = self.completion("summarize_chats", chats)
        duration = int(sum([len(c[1]) for c in chats]) / 240)
        self.schedule_chat(
            chats, chat_summary, start, duration, other
        )
        other.schedule_chat(chats, chat_summary, start, duration, self)
        return True

    def _wait_other(self, other, focus):
        """在目标位置等待另一个 Agent 完成当前行动。"""
        if self._skip_react(other):
            return False
        if not self.path:
            return False
        if self.get_event().address != other.get_tile().get_address():
            return False
        if not self.completion("decide_wait", self, other, focus):
            return False
        self.logger.info("{} decides wait to {}".format(self.name, other.name))
        start = utils.get_timer().get_date()
        # duration = other.action.end - start
        t = other.action.end - start
        duration = int(t.total_seconds() / 60)
        event = memory.Event(
            self.name,
            "waiting to start",
            self.get_event().get_describe(False),
            # address=["<waiting>"] + self.get_event().address,
            address=self.get_event().address,
            emoji=f"⌛",
        )
        self.revise_schedule(event, start, duration)

    def schedule_chat(self, chats, chats_summary, start, duration, other, address=None):
        """把一次对话写入短期聊天缓存，并安排成当前行动。"""
        self.chats.extend(chats)
        event = memory.Event(
            self.name,
            "对话",
            other.name,
            describe=chats_summary,
            address=address or self.get_tile().get_address(),
            emoji=f"💬",
        )
        self.revise_schedule(event, start, duration)

    def _add_concept(
        self,
        e_type,
        event,
        create=None,
        expire=None,
        filling=None,
    ):
        """计算事件重要性并把事件、想法或聊天写入联想记忆。"""
        if event.fit(None, "is", "idle"):
            poignancy = 1
        elif event.fit(None, "此时", "空闲"):
            poignancy = 1
        elif e_type == "chat":
            poignancy = self.completion("poignancy_chat", event)
        else:
            poignancy = self.completion("poignancy_event", event)
        self.logger.debug("{} add associate {}".format(self.name, event))
        return self.associate.add_node(
            e_type,
            event,
            poignancy,
            create=create,
            expire=expire,
            filling=filling,
        )

    def get_tile(self):
        """返回 Agent 当前所在坐标对应的地图 tile。"""
        return self.maze.tile_at(self.coord)

    def get_event(self, as_act=True):
        """返回当前行动事件；`as_act=False` 时返回对应物体事件。"""
        return self.action.event if as_act else self.action.obj_event

    def is_awake(self):
        """判断 Agent 当前是否处于清醒状态。"""
        if not self.action:
            return True
        if self.get_event().fit(self.name, "is", "sleeping"):
            return False
        if self.get_event().fit(self.name, "正在", "睡觉"):
            return False
        return True

    def llm_available(self):
        """判断当前 Agent 是否已经初始化并可调用 LLM。"""
        if not self._llm:
            return False
        return self._llm.is_available()

    def to_dict(self, with_action=True):
        """把 Agent 当前状态序列化成可保存到 checkpoint 的字典。"""
        info = {
            "status": self.status,
            "schedule": self.schedule.to_dict(),
            "associate": self.associate.to_dict(),
            "chats": self.chats,
            "currently": self.scratch.currently,
        }
        if self.depression_dynamic:
            info.update(
                {
                    "depression_dynamic_state": self.depression_dynamic.to_dict(),
                    "depression_config_path": str(
                        getattr(self.depression_dynamic, "config_path", "") or ""
                    ),
                }
            )
        if with_action:
            info.update({"action": self.action.to_dict()})
        return info

    def _init_depression_dynamic(self, config):
        """初始化动态抑郁引擎。

        这里不是“总是开启”的：它依赖 agent 目录下是否存在`depression_config.json`。
        """
        explicit_config_path = str(config.get("depression_config_path", "") or "").strip() # or的作用：如果为空/None/不存在，就变成空字符串
        agent_dir = str(config.get("agent_dir", "") or "").strip()
        candidate_paths = []
        if explicit_config_path:
            candidate_paths.append(explicit_config_path)
        if agent_dir:
            candidate_paths.append(os.path.join(agent_dir, "depression_config.json"))

        # 动态抑郁人设配置文件路径
        config_path = ""
        for item in candidate_paths:
            path = os.path.abspath(str(item or "").strip())
            if path and os.path.isfile(path):
                config_path = path
                break

        # 对于没有配置动态抑郁人设的 agent，禁用动态抑郁模块。
        if not config_path:
            if self.logger:
                self.logger.info(
                    "[DEPRESSION_DYNAMIC] agent={} disabled: missing depression_config.json".format(
                        self.name
                    )
                )
            return None

        # 配置动态抑郁人设的 DepressionSimulationEngine 对象
        try:
            engine = DepressionSimulationEngine(
                {
                    "config_path": config_path,
                    "agent_dir": os.path.dirname(config_path),
                    "agent_name": self.name,
                },
                clock_provider=utils.get_timer().get_date,
            )
            engine.set_base_prompt(self._build_depression_base_prompt())

            # 如果这个 agent 之前已经跑过动态抑郁状态机，就把上一次保存下来的内部状态恢复回来。
            state_payload = config.get("depression_dynamic_state", {}) # 存在checkpoint文档
            if isinstance(state_payload, dict) and state_payload:
                engine.load_state(copy.deepcopy(state_payload))
                engine.set_base_prompt(self._build_depression_base_prompt())
            if self.logger:
                self.logger.info(
                    "[DEPRESSION_DYNAMIC] agent={} enabled config={}".format(
                        self.name, config_path
                    )
                )
            return engine
        except Exception as exc:
            if self.logger:
                self.logger.info(
                    "[DEPRESSION_DYNAMIC] agent={} init failed: {}".format(
                        self.name, exc
                    )
                )
            return None

    def _build_depression_base_prompt(self):
        """构造动态抑郁模块使用的基础角色描述。"""
        try:
            return self.scratch._base_desc()
        except Exception:
            return str(self.scratch.currently or "")

    def _initialize_depression_graph_window(self):
        """启动/恢复时让 LLM 补足运行态主诉图窗口。"""
        if not self.depression_dynamic or not self.llm_available():
            return None
        try:
            self.depression_dynamic.set_base_prompt(self._build_depression_base_prompt())
            return self.depression_dynamic.initialize_graph_window(
                location=self._dynamic_location(),
                time_of_day=self._dynamic_time_of_day(),
                roadmap_completion_func=self._depression_llm_completion,
            )
        except Exception as exc:
            if self.logger:
                self.logger.info(
                    "[DEPRESSION_DYNAMIC] agent={} graph window init failed: {}".format(
                        self.name, exc
                    )
                )
        return None

    def _prepare_depression_generate_chat(self, args, kwargs):
        """在真正生成对话前，预览“这一轮应该怎么说”。

        注意这是 preview，不会直接写状态；真正的状态提交发生在
        `_commit_depression_generate_chat()` 中。
        """
        if not self.depression_dynamic:
            return kwargs, None
        if "depression_chat_block" in kwargs and str(kwargs.get("depression_chat_block", "")).strip():
            return kwargs, None
        if len(args) < 4:
            return kwargs, None
        _, other, relation_summary, chats = args[:4]
        self.depression_dynamic.set_base_prompt(self._build_depression_base_prompt())

        context = self._build_depression_chat_context(
            other=other,
            relation_summary=relation_summary,
            chats=chats,
        )
        preview_prompt = self.depression_dynamic.preview_interaction_prompt(
            location=context["location"],
            time_of_day=context["time_of_day"],
            other_agent=context["other_agent"],
            relationship=context["relationship"],
            interaction_type=context["interaction_type"],
            conversation_content=context["conversation_content"],
            roadmap_completion_func=self._depression_llm_completion,
            emotion_completion_func=self._depression_llm_completion,
        )
        next_kwargs = dict(kwargs)
        next_kwargs["depression_chat_block"] = preview_prompt
        return next_kwargs, context

    def _prepare_depression_reflect_insights(self, kwargs):
        """给反思 prompt 附加一个更轻量的动态主诉摘要。"""
        if not self.depression_dynamic:
            return kwargs
        if "depression_reflect_block" in kwargs and str(kwargs.get("depression_reflect_block", "")).strip():
            return kwargs
        self.depression_dynamic.set_base_prompt(self._build_depression_base_prompt())
        next_kwargs = dict(kwargs)
        next_kwargs["depression_reflect_block"] = self.depression_dynamic.get_simple_prompt()
        return next_kwargs

    def _commit_depression_event(
        self,
        source,
        location,
        time_of_day,
        interaction_type,
        content,
        other_agent="",
        relationship="",
        metadata=None,
    ):
        """统一提交会影响主诉图的事件，仅允许对话和反思。"""
        if not self.depression_dynamic:
            return None
        event_source = str(source or "").strip()
        if event_source not in {"chat", "reflection"}:
            return None
        event_content = str(content or "").strip()
        if not event_content:
            return None

        self.depression_dynamic.set_base_prompt(self._build_depression_base_prompt())
        try:
            return self.depression_dynamic.commit_event(
                source=event_source,
                location=location,
                time_of_day=time_of_day,
                other_agent=other_agent,
                relationship=relationship,
                interaction_type=interaction_type,
                content=event_content,
                metadata=metadata if isinstance(metadata, dict) else {},
                roadmap_completion_func=self._depression_llm_completion,
                emotion_completion_func=self._depression_llm_completion,
            )
        except Exception as exc:
            if self.logger:
                self.logger.info(
                    "[DEPRESSION_DYNAMIC] agent={} {} commit failed: {}".format(
                        self.name, event_source, exc
                    )
                )
        return None

    def _commit_depression_reflection(self, focus, entries):
        """把一次 reflect() 中生成的 thought 合并成一次动态状态提交。"""
        if not entries:
            return None
        content, metadata = self._build_depression_reflection_payload(focus, entries)
        return self._commit_depression_event(
            source="reflection",
            location=self._dynamic_location(),
            time_of_day=self._dynamic_time_of_day(),
            interaction_type="内在反思",
            content=content,
            other_agent="",
            relationship="",
            metadata=metadata,
        )

    def _build_depression_reflection_payload(self, focus, entries):
        """把 reflect() 的焦点、结论和证据整理成动态模块事件负载。"""
        focus_items = self._normalize_depression_text_list(focus, limit=5)
        thoughts, evidence_ids, thought_node_ids = [], [], []
        for entry in entries or []:
            if not isinstance(entry, dict):
                continue
            thought = str(entry.get("thought", "") or "").strip()
            if thought:
                thoughts.append(thought)
            thought_node_id = str(entry.get("node_id", "") or "").strip()
            if thought_node_id:
                thought_node_ids.append(thought_node_id)
            evidence_ids.extend(self._normalize_depression_text_list(entry.get("evidence", []), limit=20))

        thoughts = self._dedupe_depression_texts(thoughts, limit=8)
        evidence_ids = self._dedupe_depression_texts(evidence_ids, limit=20)
        thought_node_ids = self._dedupe_depression_texts(thought_node_ids, limit=20)

        rows = []
        if focus_items:
            rows.append("反思焦点：" + "；".join(focus_items[:3]))
        if thoughts:
            rows.append("反思结论：" + "；".join(thoughts[:5]))
        rows.append("证据数量：{}".format(len(evidence_ids)))
        if evidence_ids:
            rows.append("证据线索：" + "，".join(evidence_ids[:8]))

        metadata = {
            "focus": focus_items,
            "thought_count": len(thoughts),
            "thoughts": thoughts,
            "evidence_ids": evidence_ids,
            "thought_node_ids": thought_node_ids,
        }
        return "\n".join(rows), metadata

    def _normalize_depression_text_list(self, value, limit=20):
        """把任意输入规整成去重、截断后的文本列表。"""
        if value is None:
            items = []
        elif isinstance(value, (list, tuple, set)):
            items = list(value)
        else:
            items = [value]
        return self._dedupe_depression_texts([str(item or "").strip() for item in items], limit=limit)

    @staticmethod
    def _dedupe_depression_texts(values, limit=20):
        """按原顺序去重文本，并限制数量和单条长度。"""
        results, seen = [], set()
        for item in values or []:
            text = str(item or "").strip()
            if not text or text in seen:
                continue
            seen.add(text)
            results.append(text[:160])
            if len(results) >= int(limit):
                break
        return results

    def _commit_depression_generate_chat(self, context, output):
        """在生成出本轮话语后，把“实际说出的内容”提交给动态引擎。

        这意味着：
        - preview 阶段看到的是“已有聊天上下文”；
        - commit 阶段提交的是“角色刚刚真的说出的这句话”。
        审查合理性时，这种 preview/commit 输入不完全相同值得重点留意。
        """
        if not self.depression_dynamic:
            return
        utterance = str(output or "").strip()
        if not utterance:
            return
        self._commit_depression_event(
            source="chat",
            location=context["location"],
            time_of_day=context["time_of_day"],
            other_agent=context["other_agent"],
            relationship=context["relationship"],
            interaction_type=context["interaction_type"],
            content=utterance,
            metadata={"origin": "generate_chat", "evidence_ids": []},
        )

    def _build_depression_chat_context(self, other, relation_summary, chats):
        """把游戏世界里的对象，压缩成动态抑郁模块需要的会话上下文。"""
        location = self._dynamic_location()
        relationship = self._infer_dynamic_relationship(other, relation_summary)
        interaction_type = self._infer_dynamic_interaction_type(
            other=other,
            relationship=relationship,
            chats=chats,
        )
        conversation_content = "\n".join(
            ["{}: {}".format(name, text) for name, text in (chats or [])]
        )
        return {
            "location": location,
            "time_of_day": self._dynamic_time_of_day(),
            "other_agent": getattr(other, "name", ""),
            "relationship": relationship,
            "interaction_type": interaction_type,
            "conversation_content": conversation_content,
        }

    def _dynamic_location(self):
        """把当前地图地址压缩成动态抑郁模块使用的位置文本。"""
        try:
            address = self.get_tile().get_address()
            if isinstance(address, list) and len(address) >= 2:
                return "，".join(address[-2:])
            return str(address)
        except Exception:
            return ""

    def _dynamic_time_of_day(self):
        """把当前模拟时间映射成 morning / afternoon / evening / night。"""
        hour = utils.get_timer().get_date().hour
        if 5 <= hour < 12:
            return "morning"
        if 12 <= hour < 18:
            return "afternoon"
        if 18 <= hour < 23:
            return "evening"
        return "night"

    def _infer_dynamic_relationship(self, other, relation_summary=""):
        """把原项目中的关系信息映射到动态抑郁模块的有限关系标签。"""
        if not self.depression_dynamic:
            return ""
        other_name = str(getattr(other, "name", "") or "").strip()
        summary = str(relation_summary or "")
        text = "{} {}".format(other_name, summary)
        heuristics = [
            ("治疗师", ["治疗", "咨询", "医生", "心理", "蜻蜓队长"]),
            ("家人", ["父", "母", "家人", "同住", "金龟次郎"]),
            ("朋友", ["朋友", "好友", "田德莉娜"]),
            ("邻居", ["邻居", "呱呱蛙"]),
            ("冲突关系", ["恶霸", "冲突", "蟑螂恶霸", "讨厌"]),
        ]
        for label, keywords in heuristics:
            if any(keyword in text for keyword in keywords):
                return label
        return "熟人"

    def _infer_dynamic_interaction_type(self, other, relationship, chats):
        """基于地点、关系和已有对话内容，粗略推断互动类型。"""
        other_name = str(getattr(other, "name", "") or "").strip()
        location = self._dynamic_location()
        conversation_text = "\n".join(
            ["{}: {}".format(name, text) for name, text in (chats or [])]
        )
        if relationship == "治疗师" or "心理咨询室" in location or other_name == "蜻蜓队长":
            return "治疗对话"
        if any(token in conversation_text for token in ["帮我", "怎么办", "想聊", "能不能", "求助"]):
            return "寻求帮助"
        if any(token in conversation_text for token in ["最近怎么样", "还好吗", "怎么了", "状态"]):
            return "被询问状况"
        if len(chats or []) >= 4:
            return "深度交流"
        if relationship in {"家人", "朋友", "熟人"}:
            return "闲聊"
        return "日常活动"

    def _depression_llm_completion(self, prompt):
        """给动态抑郁模块复用 Agent 当前的 LLM 配置。"""
        if not self.llm_available():
            return ""
        llm_cfg = self.think_config.get("llm", {}) if isinstance(self.think_config.get("llm", {}), dict) else {}
        return self._llm.completion(
            prompt=prompt,
            retry=int(llm_cfg.get("retry", 3) or 3),
            failsafe="",
            caller="depression_dynamic",
            temperature=float(llm_cfg.get("temperature", 0.5) or 0.5),
        ) or ""
