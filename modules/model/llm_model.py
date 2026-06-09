"""generative_agents.model.llm_model"""

import time
import re
import requests


DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS = 600
DEFAULT_LLM_RETRY = 10


def resolve_ollama_timeout_seconds(config, default=DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS):
    """Resolve a positive timeout from config for Ollama requests."""
    if not isinstance(config, dict):
        return default

    raw_value = config.get("request_timeout_seconds", config.get("timeout", default))
    try:
        timeout_value = float(raw_value)
    except (TypeError, ValueError):
        return default

    if timeout_value <= 0:
        return default
    return timeout_value


def prepare_prompt_for_model(prompt, model_name):
    if not isinstance(prompt, str):
        return prompt
    if "qwen3" in str(model_name).lower() and "\n/nothink" not in prompt:
        # Keep Qwen3 in non-thinking mode to reduce latency and avoid leaking reasoning text.
        return prompt + "\n/nothink"
    return prompt


def strip_qwen_think_tags(text):
    if not isinstance(text, str):
        return text
    return re.sub(r"<think>.*</think>", "", text, flags=re.DOTALL).strip()


class LLMModel:
    def __init__(self, config):
        self._api_key = config["api_key"]
        self._base_url = config["base_url"]
        self._model = config["model"]
        self._meta_responses = []
        self._summary = {"total": [0, 0, 0]}
        self._default_retry = config.get("retry", DEFAULT_LLM_RETRY)
        self._request_timeout_seconds = resolve_ollama_timeout_seconds(config)

        self._handle = self.setup(config)
        self._enabled = True

    def setup(self, config):
        raise NotImplementedError(
            "setup is not support for " + str(self.__class__)
        )

    def completion(
        self,
        prompt,
        retry=None,
        callback=None,
        failsafe=None,
        caller="llm_normal",
        **kwargs
    ):
        retry = self._default_retry if retry is None else retry
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
        raise NotImplementedError(
            "_completion is not support for " + str(self.__class__)
        )

    def is_available(self):
        return self._enabled  # and self._summary["total"][2] <= 10

    def get_summary(self):
        des = {}
        for k, v in self._summary.items():
            des[k] = "S:{},F:{}/R:{}".format(v[1], v[2], v[0])
        return {"model": self._model, "summary": des}

    def disable(self):
        self._enabled = False

    @property
    def meta_responses(self):
        return self._meta_responses


class OpenAILLMModel(LLMModel):
    def setup(self, config):
        from openai import OpenAI

        return OpenAI(
            api_key=self._api_key,
            base_url=self._base_url,
            timeout=self._request_timeout_seconds,
        )

    def _completion(self, prompt, temperature=0.5):
        prompt = prepare_prompt_for_model(prompt, self._model)
        messages = [{"role": "user", "content": prompt}]
        response = self._handle.chat.completions.create(
            model=self._model, messages=messages, temperature=temperature
        )
        if len(response.choices) > 0:
            return strip_qwen_think_tags(response.choices[0].message.content)
        return ""


class OllamaLLMModel(LLMModel):
    def setup(self, config):
        return None

    def ollama_chat(self, messages, temperature):
        headers = {
            "Content-Type": "application/json"
        }
        params = {
            "model": self._model,
            "messages": messages,
            "temperature": temperature,
            "stream": False,
        }
        request_url = f"{self._base_url}/chat/completions"

        try:
            response = requests.post(
                url=request_url,
                headers=headers,
                json=params,
                stream=False,
                timeout=self._request_timeout_seconds,
            )
        except requests.exceptions.Timeout as exc:
            print(
                "[OLLAMA_TIMEOUT] model={} url={} timeout={}s error={}".format(
                    self._model,
                    request_url,
                    self._request_timeout_seconds,
                    exc,
                )
            )
            raise
        return response.json()

    def _completion(self, prompt, temperature=0.5):
        prompt = prepare_prompt_for_model(prompt, self._model)
        messages = [{"role": "user", "content": prompt}]
        response = self.ollama_chat(messages=messages, temperature=temperature)
        if response and len(response["choices"]) > 0:
            ret = response["choices"][0]["message"]["content"]
            return strip_qwen_think_tags(ret)
        return ""


def create_llm_model(llm_config):
    """Create llm model"""

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
