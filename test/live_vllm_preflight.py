"""Local vLLM preflight: verify chat and embedding services before a run."""

from __future__ import annotations

import argparse
import time
from typing import Any, Dict, List, Tuple

import requests


def _root_url(base_url: str) -> str:
    value = str(base_url).rstrip("/")
    if value.endswith("/v1"):
        return value[:-3]
    return value


def _api_base(base_url: str) -> str:
    return _root_url(base_url) + "/v1"


def _json_dict(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        text = response.text.strip()
        return {"text": text} if text else {}
    if isinstance(payload, dict):
        return payload
    return {"payload": payload}


def _timed_request(
    method: str,
    url: str,
    *,
    timeout: float,
    headers: Dict[str, str] | None = None,
    **kwargs: Any,
) -> Tuple[bool, str, Dict[str, Any]]:
    started = time.monotonic()
    session = requests.Session()
    session.trust_env = False
    try:
        response = session.request(method, url, timeout=timeout, headers=headers, **kwargs)
    except requests.Timeout as exc:
        elapsed = time.monotonic() - started
        return False, f"{method} {url} timed out after {elapsed:.2f}s: {exc}", {}
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return False, f"{method} {url} request failed after {elapsed:.2f}s: {exc}", {}
    finally:
        session.close()

    elapsed = time.monotonic() - started
    payload = _json_dict(response)
    if response.status_code >= 400:
        return (
            False,
            f"{method} {url} HTTP {response.status_code} after {elapsed:.2f}s, body={payload or response.text}",
            payload,
        )
    return True, f"{method} {url} ok in {elapsed:.2f}s", payload


def check_health(base_url: str, timeout: float, label: str) -> Tuple[bool, str]:
    ok, message, payload = _timed_request("GET", f"{_root_url(base_url)}/health", timeout=timeout)
    if not ok:
        return False, message
    return True, f"{message}, service={label}, body_keys={sorted(payload.keys()) if payload else []}"


def check_version(base_url: str, timeout: float, label: str) -> Tuple[bool, str]:
    ok, message, payload = _timed_request("GET", f"{_root_url(base_url)}/version", timeout=timeout)
    if not ok:
        return False, message

    version = str(payload.get("version") or payload.get("text") or "").strip()
    if not version:
        return False, f"{message}, but response missing version: {payload}"
    return True, f"{message}, service={label}, version={version}"


def check_models(base_url: str, timeout: float, model_name: str, label: str) -> Tuple[bool, str]:
    ok, message, payload = _timed_request("GET", f"{_api_base(base_url)}/models", timeout=timeout)
    if not ok:
        return False, message

    models = payload.get("data")
    if not isinstance(models, list):
        return False, f"{message}, but response missing data list: {payload}"

    names = {str(item.get("id", "")).strip() for item in models if isinstance(item, dict)}
    if model_name not in names:
        return False, f"{message}, service={label}, missing model={model_name}, available={sorted(names)}"
    return True, f"{message}, service={label}, found model={model_name}"


def check_chat(base_url: str, timeout: float, model: str, prompt: str, temperature: float) -> Tuple[bool, str]:
    if "qwen3" in model.lower() and "\n/nothink" not in prompt:
        prompt = f"{prompt}\n/nothink"

    ok, message, payload = _timed_request(
        "POST",
        f"{_api_base(base_url)}/chat/completions",
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        json={
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": temperature,
            "stream": False,
        },
    )
    if not ok:
        return False, message

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        return False, f"{message}, but response missing choices: {payload}"

    content = str((((choices[0] or {}).get("message") or {}).get("content") or "")).strip()
    if not content:
        return False, f"{message}, but response content is empty: {payload}"
    preview = content.replace("\n", "\\n")[:80]
    return True, f"{message}, reply_preview={preview!r}"


def check_embed(base_url: str, timeout: float, model: str, text: str) -> Tuple[bool, str]:
    ok, message, payload = _timed_request(
        "POST",
        f"{_api_base(base_url)}/embeddings",
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        json={"model": model, "input": text},
    )
    if not ok:
        return False, message

    data = payload.get("data")
    if not isinstance(data, list) or not data:
        return False, f"{message}, but response missing data list: {payload}"

    embedding = (data[0] or {}).get("embedding") if isinstance(data[0], dict) else None
    dim = len(embedding) if isinstance(embedding, list) else 0
    if dim <= 0:
        return False, f"{message}, but embedding dimension is invalid: {payload}"
    return True, f"{message}, embedding_dim={dim}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local vLLM preflight before simulations")
    parser.add_argument("--chat-base-url", default="http://127.0.0.1:18000", help="vLLM chat service base URL")
    parser.add_argument("--embed-base-url", default="http://127.0.0.1:18001", help="vLLM embedding service base URL")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    parser.add_argument("--chat-model", default="qwen3-8b-vllm", help="served chat model name")
    parser.add_argument("--embed-model", default="bge-m3-vllm", help="served embedding model name")
    parser.add_argument("--chat-prompt", default="只回复ok", help="chat warm-up prompt")
    parser.add_argument("--embed-text", default="hello", help="text used for embedding warm-up")
    parser.add_argument("--temperature", type=float, default=0.5, help="chat temperature")
    parser.add_argument("--skip-chat", action="store_true", help="skip chat service checks")
    parser.add_argument("--skip-embed", action="store_true", help="skip embedding service checks")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checks: List[Tuple[str, Tuple[bool, str]]] = []

    if not args.skip_chat:
        checks.extend(
            [
                ("chat_health", check_health(args.chat_base_url, args.timeout, "chat")),
                ("chat_version", check_version(args.chat_base_url, args.timeout, "chat")),
                ("chat_models", check_models(args.chat_base_url, args.timeout, args.chat_model, "chat")),
                (
                    "chat",
                    check_chat(
                        base_url=args.chat_base_url,
                        timeout=args.timeout,
                        model=args.chat_model,
                        prompt=args.chat_prompt,
                        temperature=args.temperature,
                    ),
                ),
            ]
        )

    if not args.skip_embed:
        checks.extend(
            [
                ("embed_health", check_health(args.embed_base_url, args.timeout, "embed")),
                ("embed_version", check_version(args.embed_base_url, args.timeout, "embed")),
                ("embed_models", check_models(args.embed_base_url, args.timeout, args.embed_model, "embed")),
                (
                    "embed",
                    check_embed(
                        base_url=args.embed_base_url,
                        timeout=args.timeout,
                        model=args.embed_model,
                        text=args.embed_text,
                    ),
                ),
            ]
        )

    failed = False
    for name, (ok, message) in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {message}")
        if not ok:
            failed = True

    if failed:
        print("vLLM preflight: FAIL")
        return 1

    print("vLLM preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
