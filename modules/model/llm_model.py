"""generative_agents.model.llm_model"""

import json
import re
import time
from urllib.parse import urlsplit, urlunsplit

import requests


DEFAULT_OLLAMA_REQUEST_TIMEOUT_SECONDS = 600
DEFAULT_LLM_RETRY = 10
ERROR_LOG_TEXT_LIMIT = 500
ERROR_CAUSE_CHAIN_LIMIT = 3


_SENSITIVE_LOG_PATTERNS = (
    re.compile(r"(?i)(authorization\s*[:=]\s*)(?:bearer\s+)?[^\s,;}]+"),
    re.compile(r"(?i)((?:api[_-]?key|x-api-key)\s*[:=]\s*)[^\s,;}]+"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._~+/=-]+"),
)


def _safe_exception_text(value, limit=ERROR_LOG_TEXT_LIMIT):
    """Return a redacted, single-line diagnostic string without raising."""
    try:
        text = str(value)
    except Exception:
        try:
            text = repr(value)
        except Exception:
            text = "<unprintable>"
    text = " ".join(text.replace("\x00", "\\x00").split())
    for pattern in _SENSITIVE_LOG_PATTERNS:
        text = pattern.sub(lambda match: match.group(1) + "<redacted>", text)
    if len(text) > limit:
        text = text[:limit] + "...<truncated>"
    return text


def sanitize_endpoint_for_log(base_url):
    """Remove credentials, query parameters, and fragments from a logged URL."""
    raw_url = _safe_exception_text(base_url, limit=ERROR_LOG_TEXT_LIMIT)
    if not raw_url:
        return ""
    try:
        parsed = urlsplit(raw_url)
        if not parsed.scheme or not parsed.netloc:
            return raw_url.split("?", 1)[0].split("#", 1)[0]
        hostname = parsed.hostname or ""
        if ":" in hostname and not hostname.startswith("["):
            hostname = "[{}]".format(hostname)
        netloc = hostname
        if parsed.port is not None:
            netloc = "{}:{}".format(netloc, parsed.port)
        return urlunsplit((parsed.scheme, netloc, parsed.path, "", ""))
    except Exception:
        without_query = raw_url.split("?", 1)[0].split("#", 1)[0]
        return re.sub(r"(?<=//)[^/@\s]+@", "<redacted>@", without_query)


def _safe_attr(obj, name, default=None):
    try:
        return getattr(obj, name, default)
    except Exception:
        return default


def _exception_message(exc):
    message = _safe_exception_text(exc)
    if message:
        return message
    return _safe_exception_text(repr(exc)) or type(exc).__name__


def safe_exception_message_for_log(exc):
    """Expose the bounded/redacted exception message used by diagnostic logs."""
    return _exception_message(exc)


def _exception_cause_chain(exc, limit=ERROR_CAUSE_CHAIN_LIMIT):
    chain, seen = [], {id(exc)}
    current = exc
    while len(chain) < limit:
        cause = _safe_attr(current, "__cause__")
        if cause is None and not bool(_safe_attr(current, "__suppress_context__", False)):
            cause = _safe_attr(current, "__context__")
        if cause is None or id(cause) in seen:
            break
        seen.add(id(cause))
        chain.append(
            {
                "type": type(cause).__name__,
                "message": _exception_message(cause),
            }
        )
        current = cause
    return chain


def _exception_http_metadata(exc):
    status_code = _safe_attr(exc, "status_code")
    request_id = _safe_attr(exc, "request_id")
    response = _safe_attr(exc, "response")
    if status_code is None and response is not None:
        status_code = _safe_attr(response, "status_code")
    if not request_id and response is not None:
        headers = _safe_attr(response, "headers")
        if headers is not None:
            try:
                request_id = headers.get("x-request-id") or headers.get("request-id")
            except Exception:
                request_id = None
    try:
        normalized_status = int(status_code) if status_code is not None else None
    except (TypeError, ValueError):
        normalized_status = _safe_exception_text(status_code)
    return normalized_status, _safe_exception_text(request_id)


def classify_call_exception(exc, stage="request"):
    """Classify a client-side failure without importing a provider SDK."""
    if stage in {"callback", "response_normalization", "output_parse"}:
        return "output_parse_or_validation"

    status_code, _request_id = _exception_http_metadata(exc)
    if status_code == 429:
        return "rate_limit_or_concurrency"
    if status_code in {401, 403}:
        return "authentication"
    if status_code in {408, 504}:
        return "timeout"
    if isinstance(status_code, int) and 500 <= status_code <= 599:
        return "server_error"
    if isinstance(status_code, int) and 400 <= status_code <= 499:
        return "client_error"

    chain = [{"type": type(exc).__name__, "message": _exception_message(exc)}]
    chain.extend(_exception_cause_chain(exc))
    searchable = " ".join(
        "{} {}".format(item.get("type", ""), item.get("message", ""))
        for item in chain
    ).lower()
    if any(token in searchable for token in ("rate limit", "ratelimit", "too many requests", "concurrency")):
        return "rate_limit_or_concurrency"
    if any(token in searchable for token in ("timeout", "timed out", "deadline exceeded")):
        return "timeout"
    if any(
        token in searchable
        for token in (
            "connection",
            "connecterror",
            "dns",
            "name resolution",
            "network is unreachable",
            "broken pipe",
        )
    ):
        return "connection"
    return "unexpected"


def format_call_error_details(
    exc,
    *,
    caller,
    stage,
    provider,
    model,
    base_url,
    attempt,
    total_attempts,
    retrying,
    elapsed_ms=None,
):
    """Format safe structured error details for stdout/stderr logs."""
    try:
        status_code, request_id = _exception_http_metadata(exc)
        details = {
            "caller": _safe_exception_text(caller),
            "stage": _safe_exception_text(stage),
            "provider": _safe_exception_text(provider),
            "model": _safe_exception_text(model),
            "endpoint": sanitize_endpoint_for_log(base_url),
            "attempt": int(attempt),
            "total_attempts": int(total_attempts),
            "retrying": bool(retrying),
            "category": classify_call_exception(exc, stage=stage),
            "exception_type": type(exc).__name__,
            "message": _exception_message(exc),
            "exception_repr": _safe_exception_text(repr(exc)),
            "cause_chain": _exception_cause_chain(exc),
        }
        if elapsed_ms is not None:
            details["elapsed_ms"] = max(0, int(round(float(elapsed_ms))))
        if status_code is not None:
            details["status_code"] = status_code
        if request_id:
            details["request_id"] = request_id
        return json.dumps(details, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    except Exception as format_exc:
        fallback = {
            "caller": _safe_exception_text(caller),
            "stage": _safe_exception_text(stage),
            "category": "unexpected",
            "exception_type": type(exc).__name__,
            "message": _exception_message(exc),
            "format_error": _exception_message(format_exc),
        }
        return json.dumps(fallback, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ollama_error_response_summary(response):
    """Return a bounded error-only response summary without logging request data."""
    try:
        payload = response.json()
    except Exception:
        return _safe_exception_text(_safe_attr(response, "text", ""), limit=300)
    if isinstance(payload, dict):
        selected = {
            key: payload.get(key)
            for key in ("error", "message", "detail")
            if key in payload
        }
        if selected:
            return _safe_exception_text(
                json.dumps(selected, ensure_ascii=False, default=str),
                limit=300,
            )
        return "body_keys={}".format(sorted(str(key) for key in payload.keys()))
    return "body_type={}".format(type(payload).__name__)


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
        self._provider = config.get("provider", "")
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
        for attempt in range(1, retry + 1):
            started_at = time.monotonic()
            stage = "request"
            try:
                meta_response = self._completion(prompt, **kwargs)
                stage = "response_normalization"
                meta_response = meta_response.strip()
                self._meta_responses.append(meta_response)
                self._summary["total"][0] += 1
                self._summary[caller][0] += 1
                if callback:
                    stage = "callback"
                    response = callback(meta_response)
                else:
                    response = meta_response
            except Exception as e:
                elapsed_ms = (time.monotonic() - started_at) * 1000
                details = format_call_error_details(
                    e,
                    caller=caller,
                    stage=stage,
                    provider=self._provider,
                    model=self._model,
                    base_url=self._base_url,
                    attempt=attempt,
                    total_attempts=retry,
                    retrying=attempt < retry,
                    elapsed_ms=elapsed_ms,
                )
                print(
                    "LLMModel.completion() caused an error: {} | [LLM_CALL_ERROR] {}".format(
                        safe_exception_message_for_log(e),
                        details,
                    ),
                    flush=True,
                )
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
                    sanitize_endpoint_for_log(request_url),
                    self._request_timeout_seconds,
                    safe_exception_message_for_log(exc),
                )
            )
            raise
        status_code = _safe_attr(response, "status_code")
        try:
            is_http_error = int(status_code) >= 400
        except (TypeError, ValueError):
            is_http_error = False
        if is_http_error:
            print(
                "[OLLAMA_HTTP_ERROR] {}".format(
                    json.dumps(
                        {
                            "model": _safe_exception_text(self._model),
                            "endpoint": sanitize_endpoint_for_log(request_url),
                            "status_code": int(status_code),
                            "response_summary": _ollama_error_response_summary(response),
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    )
                ),
                flush=True,
            )
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
