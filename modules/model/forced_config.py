"""Workspace authority for requests to the official DeepSeek API.

Read at client creation, so archived runtime configs cannot pin an old model.
Local and third-party endpoints retain their own configuration.
"""

import copy
import json
import os
from pathlib import Path
from urllib.parse import urlsplit

CONFIG_PATH = Path(__file__).resolve().parents[2] / "data" / "config.json"


def load_forced_llm_config():
    with CONFIG_PATH.open(encoding="utf-8") as handle:
        config = json.load(handle)["intervention"]["forced_llm"]
    if not isinstance(config, dict) or not config:
        raise ValueError("intervention.forced_llm must be a nonempty object")
    return copy.deepcopy(config)


def is_official_deepseek(base_url):
    return urlsplit(str(base_url or "")).hostname == "api.deepseek.com"


def resolve_deepseek_config(config):
    if not is_official_deepseek(config.get("base_url")):
        return copy.deepcopy(config)
    resolved = load_forced_llm_config()
    if str(resolved.get("enabled", False)).lower() not in {"true", "1"}:
        raise ValueError("DeepSeek API disabled by intervention.forced_llm.enabled")
    for field in ("provider", "model", "base_url"):
        if not resolved.get(field):
            raise ValueError(f"Missing intervention.forced_llm.{field}")
    env_name = resolved.get("api_key_env") or "DEEPSEEK_API_KEY"
    resolved["api_key"] = resolved.get("api_key") or os.getenv(env_name, "")
    if not resolved["api_key"]:
        raise ValueError(f"Missing API key configured by forced_llm.api_key_env: {env_name}")
    return resolved
