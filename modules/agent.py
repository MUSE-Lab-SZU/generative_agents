"""generative_agents.agent"""

import os
import math
import random
import datetime
import copy

from modules import memory, prompt, utils
from modules import depression_runtime_manager as drm
from modules import depression_dynamic_adapter as dda
from modules.model.llm_model import create_llm_model
from modules.memory.associate import Concept
from modules.external_memory_bridge import ExternalMemoryBridge


class Agent:
    def __init__(self, config, maze, conversation, logger):
        self.name = config["name"]
        self.maze = maze
        self.conversation = conversation
        self._llm = None
        self._forced_llm = None
        self._chat_route_ctx = None
        self.logger = logger

        # agent config
        self.percept_config = config["percept"]
        self.think_config = config["think"]
        self.chat_iter = config["chat_iter"]
        global_chat_history = config.get("chat_history", {}) or {}
        local_chat_history = (config.get("_raw", {}) or {}).get("chat_history", {})
        if not isinstance(local_chat_history, dict):
            local_chat_history = {}
        self.chat_history_config = {}
        if isinstance(global_chat_history, dict):
            self.chat_history_config.update(global_chat_history)
        self.chat_history_config.update(local_chat_history)
        global_chat_memory = config.get("chat_memory", {}) or {}
        local_chat_memory = (config.get("_raw", {}) or {}).get("chat_memory", {})
        if not isinstance(local_chat_memory, dict):
            local_chat_memory = {}
        self.chat_memory_config = {}
        if isinstance(global_chat_memory, dict):
            self.chat_memory_config.update(global_chat_memory)
        self.chat_memory_config.update(local_chat_memory)
        self.chat_summary_window_minutes = self._resolve_chat_summary_window_minutes(
            self.chat_history_config.get("summary_window_minutes", 480)
        )
        self.chat_history_max_read_items = self._resolve_chat_history_max_read_items(
            self.chat_history_config.get("max_read_items", 5)
        )
        self.chat_focus_retrieve_max = self._resolve_chat_focus_retrieve_max(
            self.chat_history_config.get("focus_retrieve_max", 15)
        )
        self.chat_recent_turn_focus_n = self._resolve_chat_recent_turn_focus_n(
            self.chat_history_config.get("recent_turn_focus_n", 4)
        )
        self.chat_memory_write_mode = self._resolve_chat_memory_write_mode(
            self.chat_memory_config.get("write_mode", "hybrid")
        )
        self.logger.info(
            "[CHAT_HISTORY_WINDOW] agent={} summary_window_minutes={} source={}".format(
                self.name,
                self.chat_summary_window_minutes,
                "agent.chat_history.summary_window_minutes",
            )
        )
        self.logger.info(
            "[CHAT_HISTORY_READ_CFG] agent={} max_read_items={} focus_retrieve_max={} recent_turn_focus_n={}".format(
                self.name,
                self.chat_history_max_read_items,
                self.chat_focus_retrieve_max,
                self.chat_recent_turn_focus_n,
            )
        )
        self.logger.info(
            "[CHAT_MEMORY_WRITE_CFG] agent={} write_mode={}".format(
                self.name,
                self.chat_memory_write_mode,
            )
        )
        global_external_memory = config.get("external_memory", {}) or {}
        local_external_memory = (config.get("_raw", {}) or {}).get("external_memory", {})
        if not isinstance(local_external_memory, dict):
            local_external_memory = {}
        self.external_memory_config = {}
        if isinstance(global_external_memory, dict):
            self.external_memory_config.update(global_external_memory)
        self.external_memory_config.update(local_external_memory)
        associate_storage_dir = os.path.join(config["storage_root"], "associate")
        self.external_memory_bridge = ExternalMemoryBridge.from_agent_config(
            agent_name=self.name,
            global_cfg=global_external_memory,
            local_cfg=local_external_memory,
            storage_dir=associate_storage_dir,
            logger=self.logger,
        )

        # memory
        self.spatial = memory.Spatial(**config["spatial"])
        self.schedule = memory.Schedule(**config["schedule"])
        self.associate = memory.Associate(
            associate_storage_dir,
            logger=self.logger,
            external_write_hook=self._external_memory_ingest_hook,
            **config["associate"]
        )
        self.concepts, self.chats = [], config.get("chats", [])

        # prompt
        self.scratch = prompt.Scratch(self.name, config["currently"], config["scratch"])
        # static persona profile used by depression chat prompt / emotion inference
        raw_profile = config.get("profile", (config.get("_raw", {}) or {}).get("profile", {}))
        if not isinstance(raw_profile, dict):
            raw_profile = {}
        self.profile = copy.deepcopy(raw_profile)

        # status
        status = {"poignancy": 0}
        self.status = utils.update_dict(status, config.get("status", {}))
        self.plan = config.get("plan", {})
        self.intervention = None
        self.depression_profile = drm.ensure_profile(config)
        self.depression_dynamic_enabled = False
        self.depression_dynamic_engine = None
        self.depression_dynamic_cfg = {}
        self.depression_dynamic_state = {}
        dda.init_runtime(self, config)

        # record
        self.last_record = utils.get_timer().daily_duration()

        # action and events
        if "action" in config:
            self.action = memory.Action.from_dict(config["action"])
            tiles = self.maze.get_address_tiles(self.get_event().address)
            config["coord"] = random.choice(list(tiles))
        else:
            tile = self.maze.tile_at(config["coord"])
            address = tile.get_address("game_object", as_list=True)
            self.action = memory.Action(
                memory.Event(self.name, address=address),
                memory.Event(address[-1], address=address),
            )

        # update maze
        self.coord, self.path = None, None
        self.move(config["coord"], config.get("path"))
        if self.coord is None:
            self.coord = config["coord"]

    def abstract(self):
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
        if self.schedule.scheduled():
            des["schedule"] = self.schedule.abstract()
        if self.llm_available():
            des["llm"] = self._llm.get_summary()
        # if self.plan.get("path"):
        #     des["path"] = "-".join(
        #         ["{},{}".format(c[0], c[1]) for c in self.plan["path"]]
        #     )
        return des

    def __str__(self):
        return utils.dump_dict(self.abstract())

    def reset(self):
        if not self._llm:
            self._llm = create_llm_model(self.think_config["llm"])

    def completion(self, func_hint, *args, **kwargs):
        assert hasattr(
            self.scratch, "prompt_" + func_hint
        ), "Can not find func prompt_{} from scratch".format(func_hint)
        func = getattr(self.scratch, "prompt_" + func_hint)
        prompt = func(*args, **kwargs)
        prompt = dda.patch_prompt(self, func_hint, prompt, args, kwargs)
        terminate_policy = {
            "retry": 2,
            "force_forced_llm": True,
            "source": "defaults",
        }
        if func_hint == "decide_chat_terminate":
            if self.intervention and hasattr(self.intervention, "get_decide_chat_terminate_runtime_policy"):
                try:
                    policy = self.intervention.get_decide_chat_terminate_runtime_policy()
                    if isinstance(policy, dict):
                        terminate_policy.update(policy)
                except Exception as e:
                    self.logger.warning(
                        "{} decide_chat_terminate policy error: {}".format(self.name, e)
                    )
            try:
                retry = max(1, int(terminate_policy.get("retry", 2) or 2))
            except Exception:
                retry = 2
            prompt["retry"] = retry
        title, msg = "{}.{}".format(self.name, func_hint), {}
        if self.llm_available():
            self.logger.info("{} -> {}".format(self.name, func_hint))
            route = "default"
            route_reason = "default_path"
            fallback = False
            output = None
            responses = []

            should_try_forced = False
            if self.intervention and isinstance(self._chat_route_ctx, dict):
                forced_flag = bool(self._chat_route_ctx.get("forced", False))
                peer_agent = self._chat_route_ctx.get("peer_agent")
                if forced_flag and peer_agent and hasattr(self.intervention, "should_route_forced_llm"):
                    should_try_forced = bool(
                        self.intervention.should_route_forced_llm(
                            speaker=self,
                            other=peer_agent,
                            forced=forced_flag,
                        )
                    )

            if func_hint == "decide_chat_terminate" and not bool(terminate_policy.get("force_forced_llm", True)):
                should_try_forced = False
                route_reason = "terminate_policy_force_default"

            if should_try_forced:
                route_reason = "forced_route_eligible"
                forced_cfg = None
                try:
                    forced_cfg = self.intervention.get_forced_llm_runtime_config()
                except Exception as e:
                    forced_cfg = None
                    route_reason = "forced_cfg_error:{}".format(e)
                    self.logger.warning("{} forced_llm cfg error: {}".format(self.name, e))

                if forced_cfg:
                    try:
                        if not self._forced_llm:
                            self._forced_llm = create_llm_model(forced_cfg)
                        forced_prompt = dict(prompt)
                        forced_prompt["failsafe"] = None
                        forced_prompt["retry"] = max(
                            1,
                            int(forced_prompt.get("retry", forced_cfg.get("retry", 2)) or 2),
                        )
                        forced_prompt["temperature"] = float(forced_cfg.get("temperature", forced_prompt.get("temperature", 0.5)))
                        output = self._forced_llm.completion(
                            **forced_prompt,
                            caller="{}_forced".format(func_hint),
                        )
                        responses = self._forced_llm.meta_responses
                        if output is not None:
                            route = "forced_llm"
                            route_reason = "forced_llm_success"
                        else:
                            route_reason = "forced_llm_empty_fallback"
                    except Exception as e:
                        route_reason = "forced_llm_error:{}".format(e)
                        self.logger.warning("{} forced_llm call error: {}".format(self.name, e))
                else:
                    route_reason = "forced_cfg_unavailable"

            if route != "forced_llm":
                fallback = should_try_forced
                output = self._llm.completion(**prompt, caller=func_hint)
                responses = self._llm.meta_responses

            if func_hint == "decide_chat_terminate":
                route_reason = "{}|terminate_retry={}|terminate_force_forced_llm={}|terminate_policy_source={}".format(
                    route_reason,
                    prompt.get("retry", 2),
                    bool(terminate_policy.get("force_forced_llm", True)),
                    terminate_policy.get("source", "defaults"),
                )

            self.logger.info(
                "{} -> {} route={} fallback={} reason={}".format(
                    self.name,
                    func_hint,
                    route,
                    fallback,
                    route_reason,
                )
            )
            msg = {"<PROMPT>": "\n" + prompt["prompt"] + "\n"}
            msg.update(
                {
                    "<RESPONSE[{}/{}]>".format(idx+1, len(responses)): "\n" + r + "\n"
                    for idx, r in enumerate(responses)
                }
            )
        else:
            output = prompt.get("failsafe")
        msg["<OUTPUT>"] = "\n" + str(output) + "\n"
        self.logger.debug(utils.block_msg(title, msg))
        return output

    def think(self, status, agents):
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
        if self.intervention:
            self.intervention.apply_forced_tasks(self, utils.get_timer().get_date())
        return self.schedule.current_plan()

    def revise_schedule(self, event, start, duration):
        self.action = memory.Action(event, start=start, duration=duration)
        plan, _ = self.schedule.current_plan()
        if len(plan["decompose"]) > 0:
            plan["decompose"] = self.completion(
                "schedule_revise", self.action, self.schedule
            )

    def percept(self):
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
                    if event.fit(predicate="对话") and event.subject != self.name:
                        self.logger.info(
                            "[CHAT_SEMANTIC_DEDUP] agent={} skip_mirror_chat_event subject={} object={} address={}".format(
                                self.name,
                                event.subject,
                                event.object,
                                ":".join(event.address),
                            )
                        )
                        continue
                    node_type = "chat" if event.fit(self.name, "对话") else "event"
                    node = self._add_concept(node_type, event)
                    self.status["poignancy"] += node.poignancy
                self.concepts.append(node)
        self.concepts = [c for c in self.concepts if c.event.subject != self.name]
        self.logger.info(
            "{} percept {}/{} concepts".format(self.name, valid_num, len(self.concepts))
        )

    def make_plan(self, agents):
        if self._reaction(agents):
            return
        if self.path:
            return
        if self.action.finished():
            self.action = self._determine_action()

    # create action && object events
    def make_event(self, subject, describe, address):
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
        focus = self.completion("reflect_focus", nodes, 3)
        retrieved = self.associate.retrieve_focus(focus, reduce_all=False)
        for r_nodes in retrieved.values():
            allow_reflect_constraint = True
            if self.intervention and isinstance(getattr(self.intervention, "config", None), dict):
                intervention_cfg = self.intervention.config.get("intervention", {}) or {}
                depression_cfg = intervention_cfg.get("depression_update", {}) or {}
                allow_reflect_constraint = bool(
                    depression_cfg.get("allow_reflection_constraint", True)
                )
            view = drm.build_intermediate_view(
                self.depression_profile,
                utils.get_timer().daily_duration(),
                stage="reflect",
                update_cfg=(
                    ((self.intervention.config.get("intervention", {}) or {}).get("depression_update", {}) or {})
                    if (self.intervention and isinstance(getattr(self.intervention, "config", None), dict))
                    else {}
                ),
            )
            thoughts = self.completion(
                "reflect_insights",
                r_nodes,
                5,
                depression_reflect_block=(
                    view.get("reflect_block", "") if allow_reflect_constraint else ""
                ),
            )
            self.logger.info(
                "========== [DEPR][REFLECT] agent={} allow_reflect_constraint={} reflect_block_len={} insights_count={} ==========".format(
                    self.name,
                    allow_reflect_constraint,
                    len(view.get("reflect_block", "") if allow_reflect_constraint else ""),
                    len(thoughts or []),
                )
            )
            for thought, evidence in thoughts:
                _add_thought(thought, evidence)
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
            _add_thought(f"对于 {self.name} 的计划：{thought}", evidence)
            thought = self.completion("reflect_chat_memory", self.chats)
            _add_thought(f"{self.name} {thought}", evidence)
        self.status["poignancy"] = 0
        self.chats = []

    def find_path(self, agents):
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
        focus = None
        ignore_words = ignore_words or ["空闲"]

        # 干预锁定优先：命中 lock 时优先尝试与 target 对话
        if self.intervention and agents:
            lock = self.status.get("intervention", {}).get("lock", {})
            target_name = lock.get("target_agent", "")
            if lock.get("enabled") and target_name in agents:
                other = agents[target_name]
                self.logger.info(
                    "========== [INTERVENTION][FORCED_REACTION] {} lock hit -> target={} meeting_id={} ==========".format(
                        self.name,
                        target_name,
                        lock.get("meeting_id", ""),
                    )
                )
                if self._chat_with(other, focus={"events": [], "thoughts": []}, forced=True):
                    return True
                self.logger.info(
                    "========== [INTERVENTION][FORCED_REACTION] {} forced chat failed with {} ==========".format(
                        self.name,
                        target_name,
                    )
                )
                return False

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

    def _resolve_chat_control_policy(self, other, forced=False):
        if not forced:
            return {
                "enabled": False,
                "source": "normal_chat",
                "forced_only": True,
                "forced_chat_iter": -1,
                "repeat_break_threshold": 2,
                "forced_chat_min_turns": -1,
                "forced_chat_max_turns": -1,
                "normalize_notes": ["normal_chat"],
            }

        mgr = getattr(self, "intervention", None)
        if not mgr or not hasattr(mgr, "get_chat_control_policy"):
            self.logger.info(
                "========== [INTERVENTION][CHAT_CTRL_POLICY_MISSING] initiator={} responder={} source=no_manager keep_main_loop=true ==========".format(
                    self.name,
                    other.name,
                )
            )
            return {
                "enabled": False,
                "source": "no_manager",
                "forced_only": True,
                "forced_chat_iter": -1,
                "repeat_break_threshold": 2,
                "forced_chat_min_turns": -1,
                "forced_chat_max_turns": -1,
                "normalize_notes": ["no_manager"],
            }

        policy = mgr.get_chat_control_policy(self, other, forced=True)
        if not isinstance(policy, dict):
            self.logger.info(
                "========== [INTERVENTION][CHAT_CTRL_POLICY_MISSING] initiator={} responder={} source=invalid_policy keep_main_loop=true ==========".format(
                    self.name,
                    other.name,
                )
            )
            return {
                "enabled": False,
                "source": "invalid_policy",
                "forced_only": True,
                "forced_chat_iter": -1,
                "repeat_break_threshold": 2,
                "forced_chat_min_turns": -1,
                "forced_chat_max_turns": -1,
                "normalize_notes": ["invalid_policy"],
            }
        return policy

    def _resolve_controlled_turn_limits(self, policy):
        budget = self.chat_iter
        min_turns = 1
        max_turns = budget

        forced_chat_iter = int(policy.get("forced_chat_iter", -1) or -1)
        if forced_chat_iter != -1:
            budget = forced_chat_iter

        forced_chat_min_turns = int(policy.get("forced_chat_min_turns", -1) or -1)
        if forced_chat_min_turns != -1:
            min_turns = forced_chat_min_turns

        forced_chat_max_turns = int(policy.get("forced_chat_max_turns", -1) or -1)
        if forced_chat_max_turns != -1:
            max_turns = forced_chat_max_turns

        if max_turns < min_turns:
            max_turns = min_turns
        if budget < 1:
            budget = 1

        return budget, min_turns, max_turns

    def _is_question_text(self, text):
        content = str(text or "").strip()
        if not content:
            return False
        if content.endswith("?") or content.endswith("？"):
            return True
        return ("吗" in content) or ("么" in content)

    def _set_chat_route_ctx(self, other, forced=False):
        prev_self_ctx = self._chat_route_ctx
        prev_other_ctx = getattr(other, "_chat_route_ctx", None)
        if bool(forced):
            self._chat_route_ctx = {
                "forced": True,
                "peer_name": getattr(other, "name", ""),
                "peer_agent": other,
            }
            if hasattr(other, "_chat_route_ctx"):
                other._chat_route_ctx = {
                    "forced": True,
                    "peer_name": getattr(self, "name", ""),
                    "peer_agent": self,
                }
        return prev_self_ctx, prev_other_ctx

    def _restore_chat_route_ctx(self, other, prev_self_ctx, prev_other_ctx):
        self._chat_route_ctx = prev_self_ctx
        if hasattr(other, "_chat_route_ctx"):
            other._chat_route_ctx = prev_other_ctx

    def _normalize_address_list(self, raw):
        if isinstance(raw, list):
            return [str(seg).strip() for seg in raw if str(seg).strip()]
        if isinstance(raw, str):
            text = raw.strip()
            if not text:
                return []
            if ":" in text:
                return [seg.strip() for seg in text.split(":") if seg.strip()]
            return [text]
        return []

    def _clone_event_with_address(self, event, address):
        if not event or not hasattr(event, "to_dict"):
            return event
        try:
            payload = event.to_dict()
            payload["address"] = self._normalize_address_list(address)
            return memory.Event.from_dict(payload)
        except Exception:
            return event

    def _resolve_forced_chat_memory_address(self, other, forced=False):
        if not bool(forced):
            return []
        mgr = getattr(self, "intervention", None)
        doctor_name = str(getattr(mgr, "doctor", "") or "")
        if not doctor_name:
            return []
        if str(self.name or "") == doctor_name:
            return self._normalize_address_list(self.get_tile().get_address())
        if str(getattr(other, "name", "") or "") == doctor_name and hasattr(other, "get_tile"):
            return self._normalize_address_list(other.get_tile().get_address())
        return []

    def _should_normalize_forced_persona_event(self, event):
        if not event:
            return False
        if str(getattr(event, "subject", "") or "") != str(self.name or ""):
            return False
        address = self._normalize_address_list(getattr(event, "address", []))
        if (not address) or address[0] != "<persona>":
            return False
        lock = self.status.get("intervention", {}).get("lock", {})
        if not isinstance(lock, dict):
            return False
        return bool(lock.get("enabled", False))

    def _chat_with(self, other, focus, forced=False):
        trace_scope = "INTERVENTION" if forced else "CHAT_CORE"
        self.logger.info(
            "========== [{}][CHAT_ATTEMPT] {} -> {} forced={} ==========".format(
                trace_scope,
                self.name,
                other.name,
                forced,
            )
        )
        if len(self.schedule.daily_schedule) < 1 or len(other.schedule.daily_schedule) < 1:
            # initializing
            self.logger.info(
                "========== [{}][CHAT_BLOCKED] reason=init_schedule_empty self={} other={} forced={} ==========".format(
                    trace_scope,
                    self.name,
                    other.name,
                    forced,
                )
            )
            return False
        if not forced and self._skip_react(other):
            self.logger.info(
                "========== [{}][CHAT_BLOCKED] reason=skip_react self={} other={} ==========".format(
                    trace_scope,
                    self.name,
                    other.name,
                )
            )
            return False
        if not forced and other.path:
            self.logger.info(
                "========== [{}][CHAT_BLOCKED] reason=other_path self={} other={} ==========".format(
                    trace_scope,
                    self.name,
                    other.name,
                )
            )
            return False
        if self.get_event().fit(predicate="对话") or other.get_event().fit(predicate="对话"):
            self.logger.info(
                "========== [{}][CHAT_BLOCKED] reason=already_chatting self={} other={} forced={} ==========".format(
                    trace_scope,
                    self.name,
                    other.name,
                    forced,
                )
            )
            return False

        prev_self_ctx, prev_other_ctx = self._set_chat_route_ctx(other, forced=forced)

        chats = self.associate.retrieve_chats(other.name)
        if chats:
            delta = utils.get_timer().get_delta(chats[0].create)
            self.logger.info(
                "retrieved chat between {} and {}({} min):\n{}".format(
                    self.name, other.name, delta, chats[0]
                )
            )
            if not forced and delta < 60:
                self._restore_chat_route_ctx(other, prev_self_ctx, prev_other_ctx)
                self.logger.info(
                    "========== [{}][CHAT_BLOCKED] reason=delta_lt_60 self={} other={} delta={} ==========".format(
                        trace_scope,
                        self.name,
                        other.name,
                        delta,
                    )
                )
                return False

        if not forced and (not self.completion("decide_chat", self, other, focus, chats)):
            self._restore_chat_route_ctx(other, prev_self_ctx, prev_other_ctx)
            self.logger.info(
                "========== [{}][CHAT_BLOCKED] reason=decide_chat_reject self={} other={} ==========".format(
                    trace_scope,
                    self.name,
                    other.name,
                )
            )
            return False

        policy = self._resolve_chat_control_policy(other, forced=forced)
        enhanced = bool(forced and policy.get("enabled", False))
        if forced and not enhanced:
            self.logger.info(
                "========== [INTERVENTION][CHAT_CTRL_POLICY_MISSING_OR_DISABLED] initiator={} responder={} source={} keep_main_loop=true ==========".format(
                    self.name,
                    other.name,
                    policy.get("source", "fallback"),
                )
            )

        chat_iter_budget = int(self.chat_iter)
        min_gate_turn = 1
        max_cap_turn = chat_iter_budget
        repeat_break_threshold = 1
        repeat_detection_enabled = True
        terminate_tail_window_enabled = False
        terminate_tail_window_turns = 2

        if enhanced:
            chat_iter_budget, min_gate_turn, max_cap_turn = self._resolve_controlled_turn_limits(policy)
            repeat_break_threshold = int(policy.get("repeat_break_threshold", 2) or 2)
            repeat_detection_enabled = bool(policy.get("repeat_detection_enabled", True))
            terminate_tail_window_enabled = bool(policy.get("terminate_detection_tail_window_enabled", False))
            terminate_tail_window_turns = int(policy.get("terminate_detection_tail_window_turns", 2) or 2)
        else:
            repeat_break_threshold = 1
            repeat_detection_enabled = True
            terminate_tail_window_enabled = False
            terminate_tail_window_turns = 2

        self.logger.info(
            "========== [{}][CHAT_LOOP_MODE] mode=main_loop enhanced={} forced={} source={} ==========".format(
                trace_scope,
                enhanced,
                forced,
                policy.get("source", "normal_chat") if isinstance(policy, dict) else "unknown",
            )
        )

        self.logger.info("{} decides chat with {}".format(self.name, other.name))
        self.logger.info(
            "========== [{}][CHAT_LOOP_CFG] initiator={} responder={} chat_iter={} forced={} ==========".format(
                trace_scope,
                self.name,
                other.name,
                int(chat_iter_budget),
                forced,
            )
        )
        start, chats = utils.get_timer().get_date(), []
        relations = [
            self.completion("summarize_relation", self, other.name),
            other.completion("summarize_relation", other, self.name),
        ]

        repeat_streak_self = 0
        repeat_streak_other = 0
        question_streak_self = 0
        question_streak_other = 0

        retrieval_profile = {}
        retrieval_profile_meta = {
            "enabled": False,
            "scope": "",
            "reason": "not_configured",
        }
        if self.intervention and hasattr(self.intervention, "get_memory_retrieval_profile"):
            profile_payload = self.intervention.get_memory_retrieval_profile(
                self, other, forced=forced
            )
            if isinstance(profile_payload, dict):
                retrieval_profile_meta = {
                    "enabled": bool(profile_payload.get("enabled", False)),
                    "scope": str(profile_payload.get("scope", "") or ""),
                    "reason": str(profile_payload.get("reason", "") or ""),
                }
                if retrieval_profile_meta["enabled"]:
                    retrieval_profile = profile_payload.get("profile", {}) or {}
        self.logger.info(
            "[CHAT_RETRIEVAL_PROFILE] agent={} other={} forced={} enabled={} scope={} reason={} profile={}".format(
                self.name,
                other.name,
                forced,
                retrieval_profile_meta.get("enabled", False),
                retrieval_profile_meta.get("scope", ""),
                retrieval_profile_meta.get("reason", ""),
                retrieval_profile,
            )
        )

        forced_chat_expire_days = None
        if self.intervention and hasattr(self.intervention, "get_forced_chat_expire_days"):
            forced_chat_expire_days = self.intervention.get_forced_chat_expire_days(
                self, other, forced=forced
            )
        self.logger.info(
            "[FORCED_CHAT_EXPIRE_RESOLVE] agent={} other={} forced={} expire_days={}".format(
                self.name,
                other.name,
                forced,
                forced_chat_expire_days,
            )
        )

        dialog_judge_enabled = False
        if forced and self.intervention and hasattr(self.intervention, "get_dialog_judge_runtime_policy"):
            try:
                policy_payload = self.intervention.get_dialog_judge_runtime_policy()
                if isinstance(policy_payload, dict):
                    dialog_judge_enabled = bool(policy_payload.get("enabled", False))
            except Exception as e:
                dialog_judge_enabled = False
                self.logger.warning(
                    "[DIALOG_JUDGE_POLICY_ERROR] agent={} other={} error={}".format(
                        self.name,
                        other.name,
                        e,
                    )
                )
        self.logger.info(
            "[DIALOG_JUDGE_SWITCH] agent={} other={} forced={} enabled={}".format(
                self.name,
                other.name,
                forced,
                dialog_judge_enabled,
            )
        )
        doctor_turn_judge_cache = {
            "valid": False,
            "speaker": "",
            "turn_no": -1,
            "terminate": False,
            "advice": "",
        }

        for i in range(chat_iter_budget):
            turn_no = i + 1
            terminate_check_enabled_this_turn = True
            if enhanced and terminate_tail_window_enabled:
                tail_turns = max(1, int(terminate_tail_window_turns))
                if tail_turns > int(max_cap_turn):
                    tail_turns = int(max_cap_turn)
                terminate_start_turn = int(max_cap_turn) - tail_turns + 1
                terminate_check_enabled_this_turn = bool(turn_no >= terminate_start_turn)
            self.logger.info(
                "========== [{}][CHAT_TURN] role=initiator i={} chat_iter={} chats_so_far={} ==========".format(
                    trace_scope,
                    i,
                    int(chat_iter_budget),
                    len(chats),
                )
            )
            doctor_session_prompt_injection = ""
            doctor_consult_record_injection = ""
            judge_session_prompt_injection = ""
            if self.intervention:
                doctor_consult_record_injection = self.intervention.get_doctor_consult_record_injection(
                    self,
                    other,
                    forced,
                )
                judge_session_prompt_injection = self.intervention.get_session_prompt_for_judge(
                    self,
                    other,
                    forced,
                )
            self_is_doctor_turn = bool(
                self.intervention
                and str(self.name or "") == str(getattr(self.intervention, "doctor", "") or "")
            )
            if (
                dialog_judge_enabled
                and self_is_doctor_turn
                and self.intervention
                and hasattr(self.intervention, "judge_forced_dialog_before_doctor_speak")
            ):
                doctor_turn_judge_cache["valid"] = False
                judge = self.intervention.judge_forced_dialog_before_doctor_speak(
                    speaker=self,
                    other=other,
                    chats=chats,
                    forced=forced,
                    turn_no=turn_no,
                    session_prompt_text=judge_session_prompt_injection,
                )
                judge_valid = True if (isinstance(judge, dict) and judge.get("valid") is True) else False
                terminate_flag = True if (isinstance(judge, dict) and judge.get("terminate") is True) else False
                advice_text = ""
                if isinstance(judge, dict):
                    advice_text = str(judge.get("advice", "") or "")
                doctor_turn_judge_cache = {
                    "valid": judge_valid,
                    "speaker": self.name,
                    "turn_no": int(turn_no),
                    "terminate": terminate_flag,
                    "advice": advice_text,
                }
                self.logger.info(
                    "[DIALOG_JUDGE_PRE] turn={} terminate={}".format(
                        turn_no,
                        terminate_flag,
                    )
                )
                if advice_text:
                    doctor_session_prompt_injection += (
                        "\n<医生回复建议>\n"
                        + advice_text
                        + "\n</医生回复建议>"
                )
            self.depression_profile = drm.infer_chat_emotion(
                patient_agent=self,
                profile=self.depression_profile,
                now_step=utils.get_timer().daily_duration(),
                static_profile=getattr(self, "profile", {}),
                other_agent=getattr(other, "name", ""),
                relationship=relations[0],
                conversation_content=dda.serialize_conversation(chats),
            )
            chat_view = drm.build_intermediate_view(
                self.depression_profile,
                utils.get_timer().daily_duration(),
                stage="chat",
                update_cfg=(
                    ((self.intervention.config.get("intervention", {}) or {}).get("depression_update", {}) or {})
                    if (self.intervention and isinstance(getattr(self.intervention, "config", None), dict))
                    else {}
                ),
                static_profile=getattr(self, "profile", {}),
            )
            text = self._completion_generate_chat_with_external_route(
                other=other,
                relation=relations[0],
                chats=chats,
                depression_chat_block=chat_view.get("chat_block", ""),
                doctor_session_prompt_injection=doctor_session_prompt_injection,
                doctor_consult_record_injection=doctor_consult_record_injection,
                retrieval_profile=retrieval_profile,
                is_initiator=True,
                turn_no=turn_no,
            )

            if self._is_question_text(text):
                question_streak_self += 1
            else:
                question_streak_self = 0

            if i > 0:
                # 对于发起对话的Agent，从第2轮对话开始，检查是否出现“复读”现象
                if repeat_detection_enabled:
                    end = self.completion(
                        "generate_chat_check_repeat", self, chats, text
                    )
                    repeat_end = bool(end)
                else:
                    repeat_end = False
                    if enhanced:
                        self.logger.info(
                            "========== [{}][CHAT_REPEAT_SKIP] role=initiator i={} reason=repeat_detection_disabled ==========".format(
                                trace_scope,
                                i,
                            )
                        )
                if repeat_end:
                    if enhanced:
                        repeat_streak_self += 1
                    else:
                        repeat_streak_self = 1
                else:
                    repeat_streak_self = 0
                self.logger.info(
                    "========== [{}][CHAT_REPEAT] role=initiator i={} repeat_end={} repeat_streak={} threshold={} ==========".format(
                        trace_scope,
                        i,
                        repeat_end,
                        repeat_streak_self,
                        repeat_break_threshold,
                    )
                )
                if repeat_end:
                    if repeat_streak_self >= repeat_break_threshold and (not enhanced or turn_no >= min_gate_turn):
                        self.logger.info(
                            "========== [{}][CHAT_BREAK] reason=repeat role=initiator i={} repeat_streak={} ==========".format(
                                trace_scope,
                                i,
                                repeat_streak_self,
                            )
                        )
                        break

                # 对于发起对话的Agent，从第2轮对话开始，检查话题是否结束
                chats.append((self.name, text))
                if not dialog_judge_enabled:
                    terminate_end = False
                    if terminate_check_enabled_this_turn:
                        end = self.completion(
                            "decide_chat_terminate", self, other, chats
                        )
                        terminate_end = bool(end)
                    else:
                        marker_hit = False
                        if enhanced and self.intervention and hasattr(self.intervention, "is_forced_chat_advance_marker_hit"):
                            marker_hit = bool(
                                self.intervention.is_forced_chat_advance_marker_hit(
                                    self,
                                    other,
                                    text,
                                    forced=forced,
                                )
                            )
                        self.logger.info(
                            "========== [{}][CHAT_TERMINATE_SKIP] role=initiator i={} reason=tail_window_not_reached marker_hit={} ==========".format(
                                trace_scope,
                                i,
                                marker_hit,
                            )
                        )
                        if marker_hit:
                            self.logger.info(
                                "========== [{}][CHAT_BREAK] reason=advance_marker_pre_terminate_window role=initiator i={} ==========".format(
                                    trace_scope,
                                    i,
                                )
                            )
                            break
                    self.logger.info(
                        "========== [{}][CHAT_TERMINATE] role=initiator i={} terminate_end={} ==========".format(
                            trace_scope,
                            i,
                            terminate_end,
                        )
                    )
                    if terminate_end:
                        can_break = True
                        if enhanced:
                            can_break = turn_no >= min_gate_turn
                        if can_break:
                            self.logger.info(
                                "========== [{}][CHAT_BREAK] reason=terminate role=initiator i={} ==========".format(
                                    trace_scope,
                                    i,
                                )
                            )
                            break
            else:
                chats.append((self.name, text))

            if dialog_judge_enabled and self_is_doctor_turn:
                judge_cache_hit = (
                    str(doctor_turn_judge_cache.get("speaker", "") or "") == str(self.name or "")
                    and int(doctor_turn_judge_cache.get("turn_no", -1) or -1) == int(turn_no)
                )
                cache_hit = bool(judge_cache_hit and doctor_turn_judge_cache.get("valid", False))
                terminate_eval = False
                if cache_hit:
                    terminate_eval = bool(doctor_turn_judge_cache.get("terminate", False))
                    if enhanced:
                        terminate_eval = bool(terminate_eval and (turn_no >= min_gate_turn))
                self.logger.info(
                    "[DIALOG_JUDGE_POST] turn={} terminate_eval={}".format(
                        turn_no,
                        terminate_eval,
                    )
                )
                if judge_cache_hit and self.intervention and hasattr(self.intervention, "append_dialog_judge_trace_record"):
                    self.intervention.append_dialog_judge_trace_record(
                        speaker=self,
                        other=other,
                        chats=chats,
                        doctor_turn_judge_cache=doctor_turn_judge_cache,
                        doctor_utterance=text,
                    )
                doctor_turn_judge_cache["valid"] = False
                if terminate_eval:
                    self.logger.info(
                        "[DIALOG_JUDGE_BREAK] turn={} reason=doctor_terminate_after_utterance".format(
                            turn_no
                        )
                    )
                    break

            other_doctor_session_prompt_injection = ""
            other_doctor_consult_record_injection = ""
            other_judge_session_prompt_injection = ""
            if self.intervention:
                other_doctor_consult_record_injection = self.intervention.get_doctor_consult_record_injection(
                    other,
                    self,
                    forced,
                )
                other_judge_session_prompt_injection = self.intervention.get_session_prompt_for_judge(
                    other,
                    self,
                    forced,
                )
            other_is_doctor_turn = bool(
                self.intervention
                and str(other.name or "") == str(getattr(self.intervention, "doctor", "") or "")
            )
            if (
                dialog_judge_enabled
                and other_is_doctor_turn
                and self.intervention
                and hasattr(self.intervention, "judge_forced_dialog_before_doctor_speak")
            ):
                doctor_turn_judge_cache["valid"] = False
                judge = self.intervention.judge_forced_dialog_before_doctor_speak(
                    speaker=other,
                    other=self,
                    chats=chats,
                    forced=forced,
                    turn_no=turn_no,
                    session_prompt_text=other_judge_session_prompt_injection,
                )
                judge_valid = True if (isinstance(judge, dict) and judge.get("valid") is True) else False
                terminate_flag = True if (isinstance(judge, dict) and judge.get("terminate") is True) else False
                advice_text = ""
                if isinstance(judge, dict):
                    advice_text = str(judge.get("advice", "") or "")
                doctor_turn_judge_cache = {
                    "valid": judge_valid,
                    "speaker": other.name,
                    "turn_no": int(turn_no),
                    "terminate": terminate_flag,
                    "advice": advice_text,
                }
                self.logger.info(
                    "[DIALOG_JUDGE_PRE] turn={} terminate={}".format(
                        turn_no,
                        terminate_flag,
                    )
                )
                if advice_text:
                    other_doctor_session_prompt_injection += (
                        "\n<医生回复建议>\n"
                        + advice_text
                        + "\n</医生回复建议>"
                    )
            other.depression_profile = drm.infer_chat_emotion(
                patient_agent=other,
                profile=other.depression_profile,
                now_step=utils.get_timer().daily_duration(),
                static_profile=getattr(other, "profile", {}),
                other_agent=getattr(self, "name", ""),
                relationship=relations[1],
                conversation_content=dda.serialize_conversation(chats),
            )
            text = other._completion_generate_chat_with_external_route(
                other=self,
                relation=relations[1],
                chats=chats,
                depression_chat_block=drm.build_intermediate_view(
                    other.depression_profile,
                    utils.get_timer().daily_duration(),
                    stage="chat",
                    update_cfg=(
                        ((self.intervention.config.get("intervention", {}) or {}).get("depression_update", {}) or {})
                        if (self.intervention and isinstance(getattr(self.intervention, "config", None), dict))
                        else {}
                    ),
                    static_profile=getattr(other, "profile", {}),
                ).get("chat_block", ""),
                doctor_session_prompt_injection=other_doctor_session_prompt_injection,
                doctor_consult_record_injection=other_doctor_consult_record_injection,
                retrieval_profile=retrieval_profile,
                is_initiator=False,
                turn_no=turn_no,
            )

            if self._is_question_text(text):
                question_streak_other += 1
            else:
                question_streak_other = 0

            if i > 0:
                # 对于响应对话的Agent，从第2轮开始，检查是否出现“复读”现象
                if repeat_detection_enabled:
                    end = self.completion(
                        "generate_chat_check_repeat", other, chats, text
                    )
                    repeat_end = bool(end)
                else:
                    repeat_end = False
                    if enhanced:
                        self.logger.info(
                            "========== [{}][CHAT_REPEAT_SKIP] role=responder i={} reason=repeat_detection_disabled ==========".format(
                                trace_scope,
                                i,
                            )
                        )
                if repeat_end:
                    if enhanced:
                        repeat_streak_other += 1
                    else:
                        repeat_streak_other = 1
                else:
                    repeat_streak_other = 0
                self.logger.info(
                    "========== [{}][CHAT_REPEAT] role=responder i={} repeat_end={} repeat_streak={} threshold={} ==========".format(
                        trace_scope,
                        i,
                        repeat_end,
                        repeat_streak_other,
                        repeat_break_threshold,
                    )
                )
                if repeat_end:
                    if repeat_streak_other >= repeat_break_threshold and (not enhanced or turn_no >= min_gate_turn):
                        self.logger.info(
                            "========== [{}][CHAT_BREAK] reason=repeat role=responder i={} repeat_streak={} ==========".format(
                                trace_scope,
                                i,
                                repeat_streak_other,
                            )
                        )
                        break

            chats.append((other.name, text))

            # 对于响应对话的Agent，从第1轮开始，检查话题是否结束
            if dialog_judge_enabled and other_is_doctor_turn:
                judge_cache_hit = (
                    str(doctor_turn_judge_cache.get("speaker", "") or "") == str(other.name or "")
                    and int(doctor_turn_judge_cache.get("turn_no", -1) or -1) == int(turn_no)
                )
                cache_hit = bool(judge_cache_hit and doctor_turn_judge_cache.get("valid", False))
                terminate_eval = False
                if cache_hit:
                    terminate_eval = bool(doctor_turn_judge_cache.get("terminate", False))
                    if enhanced:
                        terminate_eval = bool(terminate_eval and (turn_no >= min_gate_turn))
                self.logger.info(
                    "[DIALOG_JUDGE_POST] turn={} terminate_eval={}".format(
                        turn_no,
                        terminate_eval,
                    )
                )
                if judge_cache_hit and self.intervention and hasattr(self.intervention, "append_dialog_judge_trace_record"):
                    self.intervention.append_dialog_judge_trace_record(
                        speaker=other,
                        other=self,
                        chats=chats,
                        doctor_turn_judge_cache=doctor_turn_judge_cache,
                        doctor_utterance=text,
                    )
                doctor_turn_judge_cache["valid"] = False
                if terminate_eval:
                    self.logger.info(
                        "[DIALOG_JUDGE_BREAK] turn={} reason=doctor_terminate_after_utterance".format(
                            turn_no
                        )
                    )
                    break
            elif not dialog_judge_enabled:
                terminate_end = False
                if terminate_check_enabled_this_turn:
                    end = other.completion(
                        "decide_chat_terminate", other, self, chats
                    )
                    terminate_end = bool(end)
                else:
                    marker_hit = False
                    if enhanced and self.intervention and hasattr(self.intervention, "is_forced_chat_advance_marker_hit"):
                        marker_hit = bool(
                            self.intervention.is_forced_chat_advance_marker_hit(
                                other,
                                self,
                                text,
                                forced=forced,
                            )
                        )
                    self.logger.info(
                        "========== [{}][CHAT_TERMINATE_SKIP] role=responder i={} reason=tail_window_not_reached marker_hit={} ==========".format(
                            trace_scope,
                            i,
                            marker_hit,
                        )
                    )
                    if marker_hit:
                        self.logger.info(
                            "========== [{}][CHAT_BREAK] reason=advance_marker_pre_terminate_window role=responder i={} ==========".format(
                                trace_scope,
                                i,
                            )
                        )
                        break
                self.logger.info(
                    "========== [{}][CHAT_TERMINATE] role=responder i={} terminate_end={} ==========".format(
                        trace_scope,
                        i,
                        terminate_end,
                    )
                )
                if terminate_end:
                    can_break = True
                    if enhanced:
                        can_break = turn_no >= min_gate_turn
                    if can_break:
                        self.logger.info(
                            "========== [{}][CHAT_BREAK] reason=terminate role=responder i={} ==========".format(
                                trace_scope,
                                i,
                            )
                        )
                        break

            if enhanced and turn_no >= max_cap_turn:
                self.logger.info(
                    "========== [{}][CHAT_BREAK] reason=max_turn_cap i={} turn_no={} max_turn_cap={} ==========".format(
                        trace_scope,
                        i,
                        turn_no,
                        max_cap_turn,
                    )
                )
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
        self.logger.info(
            "========== [{}][CHAT_DONE] pair={}<->{} utterances={} ==========".format(
                trace_scope,
                self.name,
                other.name,
                len(chats),
            )
        )
        chat_summary = self.completion("summarize_chats", chats)
        duration = int(sum([len(c[1]) for c in chats]) / 240)

        chat_expire = None
        if isinstance(forced_chat_expire_days, int):
            if forced_chat_expire_days == -1:
                chat_expire = start + datetime.timedelta(days=36500)
            elif forced_chat_expire_days > 0:
                chat_expire = start + datetime.timedelta(days=forced_chat_expire_days)
        self.logger.info(
            "[CHAT_MEMORY_EXPIRE] pair={}<->{} forced={} expire_days={} expire_at={}".format(
                self.name,
                other.name,
                forced,
                forced_chat_expire_days,
                chat_expire.strftime("%Y%m%d-%H:%M:%S") if chat_expire else "<default>",
            )
        )

        doctor_memory_address = self._resolve_forced_chat_memory_address(
            other,
            forced=forced,
        )
        chat_meta_common = {
            "forced": bool(forced),
            "expire_days": forced_chat_expire_days,
            "retrieval_profile_enabled": bool(
                retrieval_profile_meta.get("enabled", False)
            ),
            "retrieval_scope": retrieval_profile_meta.get("scope", ""),
        }
        if doctor_memory_address:
            chat_meta_common["memory_address_override"] = doctor_memory_address

        self.schedule_chat(
            chats,
            chat_summary,
            start,
            duration,
            other,
            chat_expire=chat_expire,
            chat_meta=copy.deepcopy(chat_meta_common),
        )
        other.schedule_chat(
            chats,
            chat_summary,
            start,
            duration,
            self,
            chat_expire=chat_expire,
            chat_meta=copy.deepcopy(chat_meta_common),
        )
        self._restore_chat_route_ctx(other, prev_self_ctx, prev_other_ctx)
        if self.intervention:
            self.intervention.after_chat(
                self,
                other,
                chats,
                chat_summary,
                start,
            )
        return True

    def _external_memory_ingest_hook(self, payload):
        bridge = getattr(self, "external_memory_bridge", None)
        if bridge is None or not isinstance(payload, dict):
            return
        try:
            bridge.ingest_from_local_node(
                node_id=payload.get("node_id", ""),
                node_type=payload.get("node_type", ""),
                event=payload.get("event"),
                create_time=payload.get("create"),
            )
        except Exception as exc:
            self.logger.warning(
                "[EXT_MEMORY_INGEST_FAIL] agent={} node_id={} error={}".format(
                    self.name,
                    payload.get("node_id", ""),
                    exc,
                )
            )

    def _completion_generate_chat_with_external_route(
        self,
        other,
        relation,
        chats,
        depression_chat_block="",
        doctor_session_prompt_injection="",
        doctor_consult_record_injection="",
        retrieval_profile=None,
        is_initiator=False,
        turn_no=1,
    ):
        bridge = getattr(self, "external_memory_bridge", None)
        if bridge and bridge.enabled_for_chat_read():
            retrieval = bridge.retrieve_chat_context(
                chats=chats,
                other_name=getattr(other, "name", ""),
                is_initiator=is_initiator,
                turn_no=turn_no,
            )
            query = str(retrieval.get("query", "") or "").replace("\n", " ").strip()
            if len(query) > 120:
                query = query[:120] + "..."
            retrieval_ok = bool(retrieval.get("ok", False))
            route_reason = str(retrieval.get("reason", "") or "")
            if retrieval_ok or (not bridge.fallback_to_local):
                external_memory_context = str(retrieval.get("context", "") or "")
                if retrieval_ok:
                    route_reason = "retrieve_ok"
                elif not route_reason:
                    route_reason = "retrieve_failed_no_fallback"
                self.logger.info(
                    "[EXT_MEMORY_CHAT_ROUTE] agent={} other={} route=external reason={} turn_no={} is_initiator={} query={} context_len={}".format(
                        self.name,
                        getattr(other, "name", ""),
                        route_reason,
                        turn_no,
                        bool(is_initiator),
                        query,
                        len(external_memory_context),
                    )
                )
                return self.completion(
                    "generate_chat",
                    self,
                    other,
                    relation,
                    chats,
                    depression_chat_block=depression_chat_block,
                    doctor_session_prompt_injection=doctor_session_prompt_injection,
                    doctor_consult_record_injection=doctor_consult_record_injection,
                    retrieval_profile=retrieval_profile,
                    memory_source="external",
                    external_memory_context=external_memory_context,
                )
            self.logger.info(
                "[EXT_MEMORY_CHAT_ROUTE] agent={} other={} route=local reason={} turn_no={} is_initiator={} query={}".format(
                    self.name,
                    getattr(other, "name", ""),
                    route_reason or "retrieve_failed_fallback",
                    turn_no,
                    bool(is_initiator),
                    query,
                )
            )
        return self.completion(
            "generate_chat",
            self,
            other,
            relation,
            chats,
            depression_chat_block=depression_chat_block,
            doctor_session_prompt_injection=doctor_session_prompt_injection,
            doctor_consult_record_injection=doctor_consult_record_injection,
            retrieval_profile=retrieval_profile,
            memory_source="local",
        )

    def _wait_other(self, other, focus):
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

    def schedule_chat(
        self,
        chats,
        chats_summary,
        start,
        duration,
        other,
        address=None,
        chat_expire=None,
        chat_meta=None,
    ):
        self.chats.extend(chats)
        event = memory.Event(
            self.name,
            "对话",
            other.name,
            describe=chats_summary,
            address=address or self.get_tile().get_address(),
            emoji=f"💬",
        )
        self._pending_chat_memory_meta = {
            "expire": chat_expire,
            "meta": copy.deepcopy(chat_meta or {}),
            "start": start,
            "duration": duration,
            "other": getattr(other, "name", ""),
        }
        self.logger.info(
            "[SCHEDULE_CHAT_META] agent={} other={} start={} duration={} expire_at={} meta={}".format(
                self.name,
                getattr(other, "name", ""),
                start.strftime("%Y%m%d-%H:%M:%S") if isinstance(start, datetime.datetime) else str(start),
                duration,
                chat_expire.strftime("%Y%m%d-%H:%M:%S") if isinstance(chat_expire, datetime.datetime) else "<default>",
                self._pending_chat_memory_meta.get("meta", {}),
            )
        )
        persisted_now = False
        if self.chat_memory_write_mode in {"immediate", "hybrid"}:
            persisted_now = self._persist_chat_memory_now(
                event=event,
                start=start,
                expire=chat_expire,
                other_name=getattr(other, "name", ""),
                chat_meta=self._pending_chat_memory_meta.get("meta", {}),
            )
            self.logger.info(
                "[CHAT_MEMORY_WRITE_MODE] agent={} mode={} persisted_now={} delayed_fallback={}".format(
                    self.name,
                    self.chat_memory_write_mode,
                    persisted_now,
                    not persisted_now,
                )
            )
        if persisted_now:
            self._pending_chat_memory_meta = None
        self.revise_schedule(event, start, duration)
        forced = bool((chat_meta or {}).get("forced", False))
        dda.commit_event(
            self,
            event_key="chat_event",
            forced=forced,
            fallback_hint="generate_chat",
            other_agent=getattr(other, "name", ""),
            conversation_content=dda.serialize_conversation(chats),
        )

    def _persist_chat_memory_now(
        self, event, start, expire=None, other_name="", chat_meta=None
    ):
        try:
            node = self._add_concept(
                "chat",
                event,
                create=start,
                expire=expire,
            )
            self.logger.info(
                "[CHAT_MEMORY_WRITE_IMMEDIATE] agent={} other={} node_id={} create={} expire={} meta={}".format(
                    self.name,
                    other_name,
                    getattr(node, "node_id", ""),
                    start.strftime("%Y%m%d-%H:%M:%S")
                    if isinstance(start, datetime.datetime)
                    else str(start),
                    expire.strftime("%Y%m%d-%H:%M:%S")
                    if isinstance(expire, datetime.datetime)
                    else "<default>",
                    chat_meta or {},
                )
            )
            return True
        except Exception as e:
            self.logger.warning(
                "[CHAT_MEMORY_WRITE_IMMEDIATE_FAIL] agent={} other={} mode={} error={}".format(
                    self.name,
                    other_name,
                    self.chat_memory_write_mode,
                    str(e),
                )
            )
            return False

    def _add_concept(
        self,
        e_type,
        event,
        create=None,
        expire=None,
        filling=None,
    ):
        if event.fit(None, "is", "idle"):
            poignancy = 1
        elif event.fit(None, "此时", "空闲"):
            poignancy = 1
        elif e_type == "chat":
            poignancy = self.completion("poignancy_chat", event)
        else:
            poignancy = self.completion("poignancy_event", event)
        self.logger.debug("{} add associate {}".format(self.name, event))
        event_for_memory = event
        if e_type == "chat":
            pending = getattr(self, "_pending_chat_memory_meta", None)
            if isinstance(pending, dict) and pending:
                pending_expire = pending.get("expire")
                if isinstance(pending_expire, datetime.datetime):
                    expire = pending_expire
                meta = pending.get("meta", {}) if isinstance(pending.get("meta"), dict) else {}
                override_address = self._normalize_address_list(
                    meta.get("memory_address_override", [])
                )
                if override_address:
                    event_for_memory = self._clone_event_with_address(
                        event,
                        override_address,
                    )
                self.logger.info(
                    "[CHAT_MEMORY_WRITE] agent={} other={} create={} expire={} meta={}".format(
                        self.name,
                        pending.get("other", ""),
                        create.strftime("%Y%m%d-%H:%M:%S") if isinstance(create, datetime.datetime) else "<timer_default>",
                        expire.strftime("%Y%m%d-%H:%M:%S") if isinstance(expire, datetime.datetime) else "<default>",
                        pending.get("meta", {}),
                    )
                )
                self._pending_chat_memory_meta = None
        elif e_type == "event":
            if self._should_normalize_forced_persona_event(event):
                current_tile_address = self._normalize_address_list(
                    self.get_tile().get_address()
                )
                if current_tile_address:
                    event_for_memory = self._clone_event_with_address(
                        event,
                        current_tile_address,
                    )
        return self.associate.add_node(
            e_type,
            event_for_memory,
            poignancy,
            create=create,
            expire=expire,
            filling=filling,
        )

    def get_tile(self):
        return self.maze.tile_at(self.coord)

    def get_event(self, as_act=True):
        return self.action.event if as_act else self.action.obj_event

    def _resolve_chat_summary_window_minutes(self, value):
        default_window = 480
        if isinstance(value, bool):
            self.logger.warning(
                "[CHAT_HISTORY_WINDOW] agent={} invalid_bool={} fallback={}".format(
                    self.name,
                    value,
                    default_window,
                )
            )
            return default_window
        try:
            window = int(value)
        except Exception:
            self.logger.warning(
                "[CHAT_HISTORY_WINDOW] agent={} invalid_value={} fallback={}".format(
                    self.name,
                    value,
                    default_window,
                )
            )
            return default_window
        if window == -1:
            return -1
        if window <= 0:
            self.logger.warning(
                "[CHAT_HISTORY_WINDOW] agent={} non_positive_value={} fallback={}".format(
                    self.name,
                    value,
                    default_window,
                )
            )
            return default_window
        return window

    def _resolve_chat_history_max_read_items(self, value):
        default_items = 5
        if isinstance(value, bool):
            return default_items
        try:
            limit = int(value)
        except Exception:
            return default_items
        if limit == -1:
            return -1
        if limit <= 0:
            return default_items
        return limit

    def _resolve_chat_focus_retrieve_max(self, value):
        default_limit = 15
        if isinstance(value, bool):
            return default_limit
        try:
            limit = int(value)
        except Exception:
            return default_limit
        if limit == -1:
            return -1
        if limit <= 0:
            return default_limit
        return limit

    def _resolve_chat_recent_turn_focus_n(self, value):
        default_recent_turn = 4
        if isinstance(value, bool):
            return default_recent_turn
        try:
            recent_turn = int(value)
        except Exception:
            return default_recent_turn
        if recent_turn < 0:
            return default_recent_turn
        return recent_turn

    def _resolve_chat_memory_write_mode(self, value):
        mode = str(value or "hybrid").strip().lower()
        if mode not in {"delayed", "immediate", "hybrid"}:
            self.logger.warning(
                "[CHAT_MEMORY_WRITE_CFG] agent={} invalid_mode={} fallback=hybrid".format(
                    self.name,
                    value,
                )
            )
            return "hybrid"
        return mode

    def get_chat_summary_window_minutes(self):
        return self.chat_summary_window_minutes

    def get_chat_history_max_read_items(self):
        return self.chat_history_max_read_items

    def get_chat_focus_retrieve_max(self):
        return self.chat_focus_retrieve_max

    def get_chat_recent_turn_focus_n(self):
        return self.chat_recent_turn_focus_n

    def is_awake(self):
        if not self.action:
            return True
        if self.get_event().fit(self.name, "is", "sleeping"):
            return False
        if self.get_event().fit(self.name, "正在", "睡觉"):
            return False
        return True

    def llm_available(self):
        if not self._llm:
            return False
        return self._llm.is_available()

    def to_dict(self, with_action=True):
        info = {
            "status": self.status,
            "schedule": self.schedule.to_dict(),
            "associate": self.associate.to_dict(),
            "chats": self.chats,
            "currently": self.scratch.currently,
            "depression_profile": self.depression_profile,
            "depression_dynamic_state": dda.dump_state(self),
        }
        if with_action:
            info.update({"action": self.action.to_dict()})
        return info
