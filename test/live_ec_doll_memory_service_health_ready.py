"""EC-Doll memory service availability probe: only /health and /ready."""

from __future__ import annotations

import argparse
import sys
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


def check_health(base_url: str, timeout: float) -> Tuple[bool, str]:
    url = f"{base_url}/health"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"GET /health request failed: {exc}"

    payload = _json_dict(response)
    if response.status_code != 200:
        return False, f"GET /health HTTP {response.status_code}, body={payload or response.text}"

    if str(payload.get("status", "")).strip().lower() != "ok":
        return False, f"GET /health unexpected payload: {payload}"

    return True, f"GET /health ok: status={payload.get('status')}, version={payload.get('version')}"


def check_ready(base_url: str, timeout: float) -> Tuple[bool, str]:
    url = f"{base_url}/ready"
    try:
        response = requests.get(url, timeout=timeout)
    except requests.RequestException as exc:
        return False, f"GET /ready request failed: {exc}"

    payload = _json_dict(response)
    redis_ok = bool(payload.get("redis"))
    postgres_ok = bool(payload.get("postgres"))
    chroma_ok = bool(payload.get("chroma"))
    status_text = str(payload.get("status", "")).strip().lower()

    if response.status_code == 200 and status_text == "ready" and redis_ok and postgres_ok and chroma_ok:
        return (
            True,
            "GET /ready ok: status=ready, redis=true, postgres=true, chroma=true",
        )

    if response.status_code == 503:
        return (
            False,
            "GET /ready not ready (503): "
            f"status={payload.get('status')}, redis={payload.get('redis')}, "
            f"postgres={payload.get('postgres')}, chroma={payload.get('chroma')}",
        )

    return False, f"GET /ready unexpected response: HTTP {response.status_code}, body={payload or response.text}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="EC-Doll memory service /health + /ready probe")
    parser.add_argument("--base-url", default="http://localhost:8031", help="memory service base URL")
    parser.add_argument("--timeout", type=float, default=5.0, help="request timeout in seconds")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    base_url = str(args.base_url).rstrip("/")

    health_ok, health_msg = check_health(base_url=base_url, timeout=args.timeout)
    ready_ok, ready_msg = check_ready(base_url=base_url, timeout=args.timeout)

    print(health_msg)
    print(ready_msg)

    if health_ok and ready_ok:
        print("Memory service availability check: PASS")
        return 0

    print("Memory service availability check: FAIL")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
