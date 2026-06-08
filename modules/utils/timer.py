"""generative_agents.utils.timer"""

import datetime

from .namespace import GenerativeAgentsMap, GenerativeAgentsKey


def to_date(date_str, date_format="%Y%m%d-%H:%M:%S"):
    """将日期字符串按指定格式转换为 datetime 对象"""
    if date_format == "%H:%M" and date_str.startswith("24:"):
        date_str = date_str.replace("24:", "0:")
    # 日期字符串转为datetime对象
    return datetime.datetime.strptime(date_str, date_format)


def daily_duration(date, mode="minute"):
    """计算指定时间在当天已经经过的时长"""
    duration = date.hour % 24
    if mode == "hour":
        return duration
    duration = duration * 60 + date.minute
    if mode == "minute":
        return duration
    return datetime.timedelta(minutes=duration) # 时间间隔对象


class Timer:
    """模拟世界使用的全局时钟"""

    def __init__(self, start=None):
        """初始化模拟时钟，未传 start 时使用当前真实时间"""
        self._mode = "on_time"
        if start:
            d_format = "%Y%m%d-%H:%M" if "-" in start else "%H:%M"
            self._offset = to_date(start, d_format)
        else:
            self._offset = datetime.datetime.now()

    def forward(self, offset):
        """时间向前推进 offset 分钟"""
        self._offset += datetime.timedelta(minutes=offset)

    def get_date(self, date_format=""):
        """获取当前模拟时间，可按 date_format 格式化为字符串"""
        date = self._offset
        if date_format:
            return date.strftime(date_format)
        return date

    def get_delta(self, start, end=None, mode="minute"):
        """计算 start 到 end 的时间差，默认到当前模拟时间"""
        end = end or self.get_date()
        seconds = (end - start).total_seconds()
        if mode == "second":
            return seconds
        if mode == "minute":
            return round(seconds / 60)
        if mode == "hour":
            return round(seconds / 3600)
        return end - start

    def daily_format(self):
        """返回当前模拟日期的英文月份日期格式"""
        return self.get_date("%A %B %d")

    def get_weekday(self, t):
        """获取指定时间对应的中文星期"""
        weekday_dict = {
            0: "星期一",
            1: "星期二",
            2: "星期三",
            3: "星期四",
            4: "星期五",
            5: "星期六",
            6: "星期日"
        }
        weekday = weekday_dict[t.weekday()]
        return weekday

    def daily_format_cn(self):
        """返回当前模拟日期的中文日期格式"""
        weekday = self.get_weekday(self.get_date())
        date = self.get_date("%Y年%m月%d日")
        return f"{date}（{weekday}）"

    def time_format_cn(self, t):
        """返回指定时间的中文日期时间格式"""
        weekday = self.get_weekday(t)
        date = t.strftime("%Y年%m月%d日")
        time = t.strftime("%H:%M")
        return f"{date}（{weekday}）{time}"

    def daily_duration(self, mode="minute"):
        """计算当前模拟时间在当天已经经过的时长"""
        return daily_duration(self.get_date(), mode)

    def daily_time(self, duration):
        """将当天经过的分钟数转换为当前模拟日期上的具体时间"""
        base = self.get_date().replace(hour=0, minute=0, second=0, microsecond=0)
        return base + datetime.timedelta(minutes=duration)

    @property
    def mode(self):
        """获取当前时钟模式"""
        return self._mode


def set_timer(start=None):
    """创建全局模拟时钟并注册到全局容器"""
    GenerativeAgentsMap.set(GenerativeAgentsKey.TIMER, Timer(start=start))
    return GenerativeAgentsMap.get(GenerativeAgentsKey.TIMER)


def get_timer():
    """获取全局模拟时钟，不存在时自动创建"""
    if not GenerativeAgentsMap.get(GenerativeAgentsKey.TIMER):
        set_timer()
    return GenerativeAgentsMap.get(GenerativeAgentsKey.TIMER)
