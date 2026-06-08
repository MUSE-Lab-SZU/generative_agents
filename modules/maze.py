"""generative_agents.maze"""

import random
from itertools import product

from modules import utils
from modules.memory.event import Event


class Tile:
    """地图中的单个格子，保存坐标、地址、碰撞和事件信息"""

    def __init__(
        self,
        coord,
        world,
        address_keys,
        address=None,
        collision=False,
    ):
        """初始化地图格子的空间地址、碰撞状态和默认事件"""
        # in order: world, sector, arena, game_object
        self.coord = coord
        self.address = [world]
        if address:
            self.address += address
        self.address_keys = address_keys
        self.address_map = dict(zip(address_keys[: len(self.address)], self.address))
        self.collision = collision
        self.event_cnt = 0
        self._events = {}
        if len(self.address) == 4:
            self.add_event(Event(self.address[-1], address=self.address))

    def abstract(self):
        """返回格子的简要可读信息"""
        address = ":".join(self.address)
        if self.collision:
            address += "(collision)"
        return {
            "coord[{},{}]".format(self.coord[0], self.coord[1]): address,
            "events": {k: str(v) for k, v in self.events.items()},
        }

    def __str__(self):
        """将格子信息转换为格式化字符串"""
        return utils.dump_dict(self.abstract())

    def __eq__(self, other):
        """按坐标判断两个格子是否相同"""
        if isinstance(other, Tile):
            return hash(self.coord) == hash(other.coord)
        return False

    def get_events(self):
        """获取当前格子上的所有事件对象"""
        return self.events.values()

    def add_event(self, event):
        """向当前格子添加事件，已存在的相同事件不会重复添加"""
        if isinstance(event, (tuple, list)):
            event = Event.from_list(event)
        if all(e != event for e in self._events.values()):
            self._events["e_" + str(self.event_cnt)] = event
            self.event_cnt += 1
        return event

    def remove_events(self, subject=None, event=None):
        """按事件主体或事件对象移除当前格子上的事件"""
        r_events = {}
        for tag, eve in self._events.items():
            if subject and eve.subject == subject:
                r_events[tag] = eve
            if event and eve == event:
                r_events[tag] = eve
        for r_eve in r_events:
            self._events.pop(r_eve)
        return r_events

    def update_events(self, event, match="subject"):
        """按匹配规则更新当前格子上的事件"""
        u_events = {}
        for tag, eve in self._events.items():
            if match == "subject" and eve.subject == event.subject:
                self._events[tag] = event
                u_events[tag] = event
        return u_events

    def has_address(self, key):
        """判断当前格子是否包含指定层级的地址"""
        return key in self.address_map

    def get_address(self, level=None, as_list=True):
        """获取当前格子到指定层级的地址"""
        level = level or self.address_keys[-1]
        assert level in self.address_keys, "Can not find {} from {}".format(
            level, self.address_keys
        )
        pos = self.address_keys.index(level) + 1
        if as_list:
            return self.address[:pos]
        return ":".join(self.address[:pos])

    def get_addresses(self):
        """获取当前格子的所有可索引地址前缀"""
        addresses = []
        if len(self.address) > 1:
            addresses = [
                ":".join(self.address[:i]) for i in range(2, len(self.address) + 1)
            ]
        return addresses

    @property
    def events(self):
        """获取当前格子的事件字典"""
        return self._events

    @property
    def is_empty(self):
        """判断当前格子是否为空白格子"""
        return len(self.address) == 1 and not self._events


class Maze:
    """管理地图格子、地址索引、视野范围和路径搜索"""

    def __init__(self, config, logger):
        """根据地图配置初始化所有格子和地址索引"""
        # define tiles
        self.maze_height, self.maze_width = config["size"]
        self.tile_size = config["tile_size"]
        address_keys = config["tile_address_keys"]
        self.tiles = [
            [
                Tile((x, y), config["world"], address_keys)
                for x in range(self.maze_width)
            ]
            for y in range(self.maze_height)
        ]
        for tile in config["tiles"]:
            x, y = tile.pop("coord")
            self.tiles[y][x] = Tile((x, y), config["world"], address_keys, **tile) # 坐标表达和二维数组索引的顺序不一样

        # define address
        # 地址：坐标点集合
        self.address_tiles = dict()
        for i in range(self.maze_height):
            for j in range(self.maze_width):
                for add in self.tile_at([j, i]).get_addresses():
                    self.address_tiles.setdefault(add, set()).add((j, i))

        self.logger = logger

    def _warn(self, message):
        """通过 logger 输出地图相关警告"""
        if self.logger:
            self.logger.warning(message)

    def find_path(self, src_coord, dst_coord):
        """BFS从起点坐标搜索到终点坐标的可行路径,返回路径坐标列表"""
        raw_src_coord, raw_dst_coord = src_coord, dst_coord
        src_coord, dst_coord = tuple(src_coord), tuple(dst_coord)

        if not isinstance(raw_src_coord, tuple) or not isinstance(raw_dst_coord, tuple):
            self._warn(
                "find_path coord normalized: "
                f"src_type={type(raw_src_coord).__name__}, dst_type={type(raw_dst_coord).__name__}, "
                f"src={raw_src_coord}, dst={raw_dst_coord}"
            )

        map = [[0 for _ in range(self.maze_width)] for _ in range(self.maze_height)]

        def _in_bounds(coord):
            """判断坐标是否在地图边界内"""
            return 0 <= coord[0] < self.maze_width and 0 <= coord[1] < self.maze_height

        if not _in_bounds(src_coord) or not _in_bounds(dst_coord):
            self._warn(
                "find_path skip: coord out of bounds, "
                f"src={src_coord}, dst={dst_coord}, "
                f"raw_src_type={type(raw_src_coord).__name__}, raw_dst_type={type(raw_dst_coord).__name__}"
            )
            return []

        frontier, visited = [src_coord], {src_coord}
        map[src_coord[1]][src_coord[0]] = 1

        # 关键修复：目标不可达时，frontier 会耗尽；此前循环无退出条件会导致卡死。
        while map[dst_coord[1]][dst_coord[0]] == 0 and frontier:
            new_frontier = []
            for f in frontier:
                for c in self.get_around(f):
                    if (
                        0 < c[0] < self.maze_width - 1
                        and 0 < c[1] < self.maze_height - 1
                        and map[c[1]][c[0]] == 0
                        and c not in visited
                    ):
                        map[c[1]][c[0]] = map[f[1]][f[0]] + 1
                        new_frontier.append(c)
                        visited.add(c)
            frontier = new_frontier

        if map[dst_coord[1]][dst_coord[0]] == 0:
            self._warn(
                f"find_path unreachable: src={src_coord}, dst={dst_coord}, visited={len(visited)}"
            )
            return []

        step = map[dst_coord[1]][dst_coord[0]]
        path = [dst_coord]
        while step > 1:
            for c in self.get_around(path[-1]):
                if map[c[1]][c[0]] == step - 1:
                    path.append(c)
                    break
            step -= 1
        return path[::-1]

    def tile_at(self, coord):
        """获取指定坐标上的格子"""
        return self.tiles[coord[1]][coord[0]]

    def update_obj(self, coord, obj_event):
        """同步更新同一游戏对象地址上的对象事件"""
        tile = self.tile_at(coord)
        if not tile.has_address("game_object"):
            return
        if obj_event.address != tile.get_address("game_object"):
            return
        addr = ":".join(obj_event.address)
        if addr not in self.address_tiles:
            return
        for c in self.address_tiles[addr]:
            self.tile_at(c).update_events(obj_event)

    def get_scope(self, coord, config):
        """根据感知配置获取指定坐标周围的可见格子"""
        coords = []
        vision_r = config["vision_r"]
        if config["mode"] == "box":
            x_range = [
                max(coord[0] - vision_r, 0),
                min(coord[0] + vision_r + 1, self.maze_width),
            ]
            y_range = [
                max(coord[1] - vision_r, 0),
                min(coord[1] + vision_r + 1, self.maze_height),
            ]
            coords = list(product(list(range(*x_range)), list(range(*y_range)))) # *参数展开
        return [self.tile_at(c) for c in coords]

    def get_around(self, coord, no_collision=True):
        """获取指定坐标上下左右相邻的格子坐标"""
        coords = [
            (coord[0] - 1, coord[1]),
            (coord[0] + 1, coord[1]),
            (coord[0], coord[1] - 1),
            (coord[0], coord[1] + 1),
        ]
        # 只保留没有碰撞的格子
        if no_collision:
            coords = [c for c in coords if not self.tile_at(c).collision]
        return coords

    def get_address_tiles(self, address):
        """获取指定地址对应的一组格子坐标"""
        addr = ":".join(address)
        if addr in self.address_tiles:
            return self.address_tiles[addr]
        # TODO: 返回随机地址可能导致agent行为看起来奇怪
        return random.choice(list(self.address_tiles.values()))
