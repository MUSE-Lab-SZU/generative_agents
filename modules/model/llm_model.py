"""generative_agents.model.llm_model"""

import time
import re
import requests


class LLMModel:
    """LLM 访问层基类，统一封装重试、解析、兜底值和调用统计。"""

    def __init__(self, config):
        """保存模型连接配置，并交给子类建立具体 provider 的调用句柄。"""
        self._api_key = config["api_key"]
        self._base_url = config["base_url"]
        self._model = config["model"]
        self._meta_responses = [] # 最近一次 completion 中，每次重试拿到的模型原始文本。
        self._summary = {"total": [0, 0, 0]} # 调用统计：[请求次数, 成功次数, 失败次数]。

        self._handle = self.setup(config) # provider 的连接句柄，例如 OpenAI client；Ollama 暂时不需要。
        self._enabled = True # 上层用它判断是否真实调用 LLM，False 时直接走 failsafe。

    def setup(self, config):
        """由子类实现 provider 初始化逻辑，例如创建 SDK client 或连接配置。"""
        raise NotImplementedError(
            "setup is not support for " + str(self.__class__)
        )

    def completion(
        self,
        prompt,
        retry=10,
        callback=None,
        failsafe=None,
        caller="llm_normal",
        **kwargs
    ):
        """执行一次带重试的模型调用，并用 callback 把原始文本解析成业务结果。"""
        response, self._meta_responses = None, []
        self._summary.setdefault(caller, [0, 0, 0])
        for _ in range(retry):
            try:
                meta_response = self._completion(prompt, **kwargs).strip()
                self._meta_responses.append(meta_response)
                self._summary["total"][0] += 1
                self._summary[caller][0] += 1
                if callback:
                    response = callback(meta_response)
                else:
                    response = meta_response
            except Exception as e:
                print(f"LLMModel.completion() caused an error: {e}")
                time.sleep(5)
                response = None
                continue
            if response is not None:
                break
        pos = 2 if response is None else 1
        self._summary["total"][pos] += 1
        self._summary[caller][pos] += 1
        return response or failsafe

    def _completion(self, prompt, **kwargs):
        """由子类实现单次真实模型请求，返回未经业务解析的原始文本。"""
        raise NotImplementedError(
            "_completion is not support for " + str(self.__class__)
        )

    def is_available(self):
        """返回当前模型封装是否处于可调用状态。"""
        return self._enabled  # and self._summary["total"][2] <= 10

    def get_summary(self):
        """汇总模型调用次数、成功次数和失败次数，供日志和前端调试查看。"""
        des = {}
        for k, v in self._summary.items():
            des[k] = "S:{},F:{}/R:{}".format(v[1], v[2], v[0])
        return {"model": self._model, "summary": des}

    def disable(self):
        """手动关闭当前模型封装，让上层跳过真实 LLM 调用。"""
        self._enabled = False

    @property
    def meta_responses(self):
        """返回最近一次 completion 重试过程中收集到的原始模型输出。"""
        return self._meta_responses


class OpenAILLMModel(LLMModel):
    """使用 OpenAI 兼容 Chat Completions API 的模型实现。"""

    def setup(self, config):
        """创建 OpenAI SDK client，支持传入兼容服务的 base_url。"""
        from openai import OpenAI

        return OpenAI(api_key=self._api_key, base_url=self._base_url)

    def _completion(self, prompt, temperature=0.5):
        """向 OpenAI 兼容接口发送单轮用户消息，并返回第一条候选回复。"""
        messages = [{"role": "user", "content": prompt}]
        response = self._handle.chat.completions.create(
            model=self._model, messages=messages, temperature=temperature
        )
        if len(response.choices) > 0:
            return response.choices[0].message.content
        return ""


class OllamaLLMModel(LLMModel):
    """使用 Ollama OpenAI 兼容接口的本地模型实现。"""

    def setup(self, config):
        """Ollama 当前不需要持久 SDK client，因此返回空句柄。"""
        return None

    def ollama_chat(self, messages, temperature):
        """按 OpenAI 兼容 chat/completions 格式向 Ollama 服务发送请求。"""
        headers = {
            "Content-Type": "application/json"
        }
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }

        response = requests.post(
            url=f"{self._base_url}/chat/completions",
            headers=headers,
            json=params,
            stream=False
        )
        return response.json()

    def _completion(self, prompt, temperature=0.5):
        """发送 Ollama 聊天请求，并清理可能影响解析的 Qwen think 内容。"""
        if "qwen3" in self._model and "\n/no_think" not in prompt:
            # 针对Qwen3模型禁用think，提高推理速度
            prompt += "\n/no_think"
        messages = [{"role": "user", "content": prompt}]
        response = self.ollama_chat(messages=messages, temperature=temperature)
        if response and len(response["choices"]) > 0:
            ret = response["choices"][0]["message"]["content"]
            # 从输出结果中过滤掉<think>标签内的文字，以免影响后续逻辑
            return re.sub(r"<think>.*</think>", "", ret, flags=re.DOTALL)
        return ""


def create_llm_model(llm_config):
    """根据配置中的 provider 创建对应的 LLMModel 子类实例。"""

    if llm_config["provider"] == "ollama":
        return OllamaLLMModel(llm_config)

    elif llm_config["provider"] == "openai":
        return OpenAILLMModel(llm_config)
    else:
        raise NotImplementedError(
            "llm provider {} is not supported".format(llm_config["provider"])
        )
    return None


def parse_llm_output(response, patterns, mode="match_last", ignore_empty=False):
    """用一组正则逐行解析 LLM 输出，并按指定模式返回首个、最后或全部匹配。"""
    if isinstance(patterns, str):
        patterns = [patterns]
    rets = []
    for line in response.split("\n"):
        line = line.replace("**", "").strip()
        for pattern in patterns:
            if pattern:
                matchs = re.findall(pattern, line)
            else:
                matchs = [line]
            if len(matchs) >= 1:
                rets.append(matchs[0])
                break
    if not ignore_empty:
        assert rets, "Failed to match llm output"
    if mode == "match_first":
        return rets[0]
    if mode == "match_last":
        return rets[-1]
    if mode == "match_all":
        return rets
    return None
