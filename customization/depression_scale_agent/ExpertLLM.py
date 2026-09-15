import os
from datetime import datetime, timezone
from modules.model.forced_config import load_forced_llm_config, resolve_deepseek_config

from dotenv import find_dotenv, load_dotenv
from modules.model.llm_model import (
    classify_call_exception,
    record_deepseek_call_failure,
    record_prompt_cache_usage,
)

load_dotenv(find_dotenv())


class ExpertLLM:
    def __init__(
        self,
        api_key=None,
        model=None,
        base_url=None,
        timeout=60,
        thinking=None,
        reasoning_effort=None,
    ):
        from openai import OpenAI

        defaults = load_forced_llm_config()
        config = resolve_deepseek_config({
            **defaults,
            "model": model or defaults["model"],
            "base_url": base_url or defaults["base_url"],
            "api_key": api_key or defaults.get("api_key") or os.getenv(defaults.get("api_key_env") or "DEEPSEEK_API_KEY"),
            "thinking": thinking if thinking is not None else defaults.get("thinking"),
            "reasoning_effort": reasoning_effort if reasoning_effort is not None else defaults.get("reasoning_effort"),
        })
        resolved_key = config.get("api_key")
        if not resolved_key:
            raise ValueError("Missing API key configured by forced_llm.api_key_env")
        resolved_model = config["model"]
        resolved_base_url = config["base_url"]
        thinking = config.get("thinking")
        reasoning_effort = config.get("reasoning_effort", config.get("reasoning-effort"))
        self._temperature = config.get("temperature", 0.2)
        self._caller_overrides = config.get("caller_overrides") or {}
        self._retry = config.get("retry", 2)

        self._model = resolved_model
        self._base_url = resolved_base_url
        self._thinking = thinking
        self._reasoning_effort = reasoning_effort
        self._client = OpenAI(
            api_key=resolved_key,
            base_url=resolved_base_url,
            timeout=config.get("timeout", timeout),
            max_retries=self._retry,
        )

    def _request(self, *, messages, temperature, caller, **kwargs):
        request_kwargs = dict(kwargs)
        override = self._caller_overrides.get(caller, {})
        thinking = override.get("thinking", self._thinking)
        reasoning_effort = override.get(
            "reasoning_effort", override.get("reasoning-effort", self._reasoning_effort)
        )
        if thinking is not None:
            if isinstance(thinking, str):
                thinking = {"type": thinking}
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            extra_body["thinking"] = thinking
            request_kwargs["extra_body"] = extra_body
        if isinstance(thinking, dict) and thinking.get("type") == "disabled":
            reasoning_effort = None
        if reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = reasoning_effort

        request_started_at = datetime.now(timezone.utc)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=self._temperature if temperature is None else temperature,
                **request_kwargs,
            )
        except Exception as exc:
            record_deepseek_call_failure(
                caller=caller,
                provider="openai",
                model=self._model,
                base_url=self._base_url,
                request_started_at=request_started_at,
                error_category=classify_call_exception(exc),
            )
            raise
        record_prompt_cache_usage(
            response,
            caller=caller,
            provider="openai",
            model=self._model,
            base_url=self._base_url,
            request_started_at=request_started_at,
            response_received_at=datetime.now(timezone.utc),
        )
        return response

    def generate(
        self,
        user_prompt,
        system_prompt=None,
        temperature=None,
        caller="expert_generate",
        **kwargs
    ):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})

        response = self._request(
            messages=messages,
            temperature=temperature,
            caller=caller,
            **kwargs,
        )
        if response.choices:
            return response.choices[0].message.content
        return ""

    def chat(
        self,
        messages,
        system_prompt=None,
        temperature=None,
        caller="expert_chat",
        **kwargs
    ):
        if not isinstance(messages, list):
            raise TypeError("messages must be a list of role/content dicts.")

        merged = []
        if system_prompt:
            merged.append({"role": "system", "content": system_prompt})
        merged.extend(messages)

        response = self._request(
            messages=merged,
            temperature=temperature,
            caller=caller,
            **kwargs,
        )
        if response.choices:
            return response.choices[0].message.content
        return ""
