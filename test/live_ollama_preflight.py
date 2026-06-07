"""Local Ollama preflight: verify server, chat, and embedding paths before a run."""

from __future__ import annotations

import argparse
import sys
import time
from typing import Any, Dict, Tuple

import requests


def _json_dict(response: requests.Response) -> Dict[str, Any]:
    try:
        payload = response.json()
    except ValueError:
        return {}
    if isinstance(payload, dict):
        return payload
    return {"payload": payload}


def _timed_request(method: str, url: str, *, timeout: float, **kwargs: Any) -> Tuple[bool, str, Dict[str, Any]]:
    started = time.monotonic()
    try:
        response = requests.request(method, url, timeout=timeout, **kwargs)
    except requests.Timeout as exc:
        elapsed = time.monotonic() - started
        return False, f"{method} {url} timed out after {elapsed:.2f}s: {exc}", {}
    except requests.RequestException as exc:
        elapsed = time.monotonic() - started
        return False, f"{method} {url} request failed after {elapsed:.2f}s: {exc}", {}

    elapsed = time.monotonic() - started
    payload = _json_dict(response)
    if response.status_code >= 400:
        return (
            False,
            f"{method} {url} HTTP {response.status_code} after {elapsed:.2f}s, body={payload or response.text}",
            payload,
        )
    return True, f"{method} {url} ok in {elapsed:.2f}s", payload


def check_version(base_url: str, timeout: float) -> Tuple[bool, str]:
    ok, message, payload = _timed_request("GET", f"{base_url}/api/version", timeout=timeout)
    if not ok:
        return False, message

    version = str(payload.get("version", "")).strip()
    if not version:
        return False, f"{message}, but response missing version: {payload}"
    return True, f"{message}, version={version}"


def check_tags(base_url: str, timeout: float, chat_model: str, embed_model: str) -> Tuple[bool, str]:
    ok, message, payload = _timed_request("GET", f"{base_url}/api/tags", timeout=timeout)
    if not ok:
        return False, message

    models = payload.get("models")
    if not isinstance(models, list):
        return False, f"{message}, but response missing models list: {payload}"

    names = {str(item.get("name", "")).strip() for item in models if isinstance(item, dict)}
    missing = [name for name in (chat_model, embed_model) if name not in names]
    if missing:
        return False, f"{message}, missing models={missing}, available_count={len(names)}"
    return True, f"{message}, found chat_model={chat_model}, embed_model={embed_model}"


def check_chat(base_url: str, timeout: float, model: str, prompt: str, temperature: float) -> Tuple[bool, str]:
    if "qwen3" in model and "\n/nothink" not in prompt:
        prompt = f"{prompt}\n/nothink"

    ok, message, payload = _timed_request(
        "POST",
        f"{base_url}/v1/chat/completions",
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
        f"{base_url}/api/embed",
        timeout=timeout,
        headers={"Content-Type": "application/json"},
        json={"model": model, "input": text},
    )
    if not ok:
        return False, message

    embeddings = payload.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings:
        return False, f"{message}, but response missing embeddings: {payload}"
    first = embeddings[0]
    dim = len(first) if isinstance(first, list) else 0
    if dim <= 0:
        return False, f"{message}, but embedding dimension is invalid: {payload}"
    return True, f"{message}, embedding_dim={dim}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local Ollama preflight before simulations")
    parser.add_argument("--base-url", default="http://127.0.0.1:11434", help="Ollama base URL")
    parser.add_argument("--timeout", type=float, default=30.0, help="per-request timeout in seconds")
    parser.add_argument("--chat-model", default="qwen3:8b-q4_K_M", help="chat model name")
    parser.add_argument("--embed-model", default="bge-m3:latest", help="embedding model name")
    parser.add_argument("--chat-prompt", default="只回复ok", help="chat warm-up prompt")
    parser.add_argument("--embed-text", default="hello", help="text used for embedding warm-up")
    parser.add_argument("--temperature", type=float, default=0.5, help="chat temperature")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = str(args.base_url).rstrip("/")

    checks = [
        ("version", check_version(base_url=base_url, timeout=args.timeout)),
        (
            "tags",
            check_tags(
                base_url=base_url,
                timeout=args.timeout,
                chat_model=args.chat_model,
                embed_model=args.embed_model,
            ),
        ),
        (
            "chat",
            check_chat(
                base_url=base_url,
                timeout=args.timeout,
                model=args.chat_model,
                prompt=args.chat_prompt,
                temperature=args.temperature,
            ),
        ),
        (
            "embed",
            check_embed(
                base_url=base_url,
                timeout=args.timeout,
                model=args.embed_model,
                text=args.embed_text,
            ),
        ),
    ]

    failed = False
    for name, (ok, message) in checks:
        status = "PASS" if ok else "FAIL"
        print(f"[{status}] {name}: {message}")
        if not ok:
            failed = True

    if failed:
        print("Ollama preflight: FAIL")
        return 1

    print("Ollama preflight: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
