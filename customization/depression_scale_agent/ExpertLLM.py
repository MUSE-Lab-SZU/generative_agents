import os
import json
from datetime import datetime, timezone

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

        resolved_key = (
            api_key
            or os.getenv("EXPERT_LLM_API_KEY")
            or os.getenv("DEEPSEEK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not resolved_key:
            raise ValueError("Missing API key. Set DEEPSEEK_API_KEY/OPENAI_API_KEY or pass api_key.")

        resolved_model = model or os.getenv("EXPERT_LLM_MODEL") or "deepseek-v4-flash"
        resolved_base_url = base_url or os.getenv("EXPERT_LLM_BASE_URL") or "https://api.deepseek.com"

        if thinking is None and os.getenv("EXPERT_LLM_THINKING_JSON"):
            thinking = json.loads(os.environ["EXPERT_LLM_THINKING_JSON"])
        if reasoning_effort is None:
            reasoning_effort = os.getenv("EXPERT_LLM_REASONING_EFFORT") or None

        self._model = resolved_model
        self._base_url = resolved_base_url
        self._thinking = thinking
        self._reasoning_effort = reasoning_effort
        self._client = OpenAI(api_key=resolved_key, base_url=resolved_base_url, timeout=timeout)

    def _request(self, *, messages, temperature, caller, **kwargs):
        request_kwargs = dict(kwargs)
        if self._thinking is not None:
            thinking = self._thinking
            if isinstance(thinking, str):
                thinking = {"type": thinking}
            extra_body = dict(request_kwargs.pop("extra_body", {}) or {})
            extra_body["thinking"] = thinking
            request_kwargs["extra_body"] = extra_body
        if self._reasoning_effort is not None:
            request_kwargs["reasoning_effort"] = self._reasoning_effort

        request_started_at = datetime.now(timezone.utc)
        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=temperature,
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
        temperature=0.2,
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
        temperature=0.2,
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
