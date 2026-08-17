"""Process-local round-robin endpoint selection for model services."""

from __future__ import annotations

from threading import Lock
from typing import Any, Dict, List, Tuple
from urllib.parse import urlsplit, urlunsplit


_ROUND_ROBIN_LOCK = Lock()
_ROUND_ROBIN_COUNTERS: Dict[Tuple[str, Tuple[str, ...]], int] = {}


def _replace_port(base_url: str, port: Any) -> str:
    """Return *base_url* with its network port replaced."""
    try:
        parsed = urlsplit(str(base_url).strip())
        port_number = int(port)
        if not parsed.scheme or not parsed.hostname or not 1 <= port_number <= 65535:
            raise ValueError
        host = parsed.hostname
        if ":" in host and not host.startswith("["):
            host = "[{}]".format(host)
        if parsed.username:
            credentials = parsed.username
            if parsed.password:
                credentials += ":" + parsed.password
            host = credentials + "@" + host
        return urlunsplit((parsed.scheme, "{}:{}".format(host, port_number), parsed.path, parsed.query, parsed.fragment)).rstrip("/")
    except (TypeError, ValueError):
        raise ValueError("invalid load_balancing port {!r} for base_url={!r}".format(port, base_url))


def resolve_endpoint_urls(config: Dict[str, Any]) -> List[str]:
    """Resolve configured model endpoints, falling back to the legacy base URL.

    ``load_balancing.ports`` intentionally contains only ports: the host, scheme,
    and API path continue to come from ``base_url`` so a runtime config can move
    between local and remote platforms without duplicating an address.
    """
    base_url = str(config["base_url"]).rstrip("/")
    options = config.get("load_balancing", {})
    if not isinstance(options, dict) or not options.get("enabled", False):
        return [base_url]

    ports = options.get("ports", [])
    if not isinstance(ports, list) or not ports:
        raise ValueError("load_balancing.enabled requires a non-empty ports list")
    return [_replace_port(base_url, port) for port in ports]


def next_endpoint(config: Dict[str, Any], urls: List[str]) -> str:
    """Select an endpoint using counters shared by the whole Python process."""
    if not urls:
        raise ValueError("endpoint pool is empty")
    service_name = str(config.get("model", "") or config.get("provider", "model"))
    key = (service_name, tuple(urls))
    with _ROUND_ROBIN_LOCK:
        index = _ROUND_ROBIN_COUNTERS.get(key, 0)
        _ROUND_ROBIN_COUNTERS[key] = index + 1
    return urls[index % len(urls)]
