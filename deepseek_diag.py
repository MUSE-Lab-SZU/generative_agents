#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
DeepSeek API 诊断脚本：
1. 检查环境变量和代理
2. 检查 DNS / TCP / TLS
3. 检查 /models
4. 检查 /chat/completions
5. 根据 HTTP 状态码给出可能原因

用法：
  export DEEPSEEK_API_KEY="sk-xxx"
  python3 deepseek_diag.py

可选：
  python3 deepseek_diag.py --model deepseek-v4-pro
  python3 deepseek_diag.py --base-url https://api.deepseek.com
  python3 deepseek_diag.py --ignore-proxy
  python3 deepseek_diag.py --skip-chat
"""

import argparse
import json
import os
import platform
import socket
import ssl
import sys
import time
from urllib.parse import urlparse

def load_dotenv_simple(path=".env"):
    if not os.path.exists(path):
        return

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()

            if not line or line.startswith("#"):
                continue

            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")

            # 不覆盖已经存在的环境变量
            os.environ.setdefault(key, value)

try:
    import requests
except ImportError:
    print("缺少 requests，请先运行：python3 -m pip install requests")
    sys.exit(1)


def section(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def ok(msg):
    print(f"[OK] {msg}")


def warn(msg):
    print(f"[WARN] {msg}")


def fail(msg):
    print(f"[FAIL] {msg}")


def redact_key(key: str) -> str:
    if not key:
        return "<EMPTY>"
    if len(key) <= 10:
        return key[:2] + "***"
    return key[:6] + "..." + key[-4:] + f"  len={len(key)}"


def print_env(api_key_name: str):
    section("1. 本机环境检查")

    print(f"Python: {sys.version.split()[0]}")
    print(f"Platform: {platform.platform()}")
    print(f"API key env: {api_key_name}")
    print(f"API key: {redact_key(os.getenv(api_key_name, ''))}")

    proxy_vars = [
        "HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
        "http_proxy", "https_proxy", "all_proxy", "no_proxy",
    ]
    found_proxy = False
    for k in proxy_vars:
        v = os.getenv(k)
        if v:
            found_proxy = True
            print(f"{k}={v}")

    if not found_proxy:
        print("未检测到代理环境变量")

    for k in ["OPENAI_API_KEY", "OPENAI_BASE_URL", "DEEPSEEK_BASE_URL", "REQUESTS_CA_BUNDLE", "SSL_CERT_FILE"]:
        v = os.getenv(k)
        if v:
            if "KEY" in k:
                v = redact_key(v)
            print(f"{k}={v}")


def resolve_host(host: str):
    section("2. DNS 解析检查")
    try:
        infos = socket.getaddrinfo(host, 443, type=socket.SOCK_STREAM)
        addrs = sorted({i[4][0] for i in infos})
        ok(f"{host} 解析成功")
        for addr in addrs:
            print(f"  - {addr}")
        return True
    except Exception as e:
        fail(f"DNS 解析失败：{type(e).__name__}: {e}")
        print("可能原因：服务器 DNS 配置异常、公司/机房 DNS 污染、域名被拦截。")
        return False


def tcp_check(host: str, port: int, timeout: float):
    section("3. TCP 连接检查")
    start = time.time()
    try:
        with socket.create_connection((host, port), timeout=timeout):
            elapsed = (time.time() - start) * 1000
            ok(f"TCP {host}:{port} 连接成功，耗时 {elapsed:.1f} ms")
            return True
    except Exception as e:
        fail(f"TCP 连接失败：{type(e).__name__}: {e}")
        print("可能原因：服务器出站 443 被防火墙拦截、云安全组限制、代理配置错误、目标网络不可达。")
        return False


def tls_check(host: str, port: int, timeout: float):
    section("4. TLS/证书检查")
    try:
        ctx = ssl.create_default_context()
        with socket.create_connection((host, port), timeout=timeout) as sock:
            with ctx.wrap_socket(sock, server_hostname=host) as ssock:
                cert = ssock.getpeercert()
                ok("TLS 握手成功")
                print(f"TLS version: {ssock.version()}")
                print(f"Cipher: {ssock.cipher()}")
                print(f"Cert subject: {cert.get('subject')}")
                print(f"Cert issuer: {cert.get('issuer')}")
                print(f"Cert notAfter: {cert.get('notAfter')}")
                return True
    except ssl.SSLError as e:
        fail(f"TLS/证书失败：{type(e).__name__}: {e}")
        print("可能原因：系统 CA 证书过旧、被代理做了 MITM、REQUESTS_CA_BUNDLE/SSL_CERT_FILE 配错。")
        return False
    except Exception as e:
        fail(f"TLS 检查失败：{type(e).__name__}: {e}")
        return False


def explain_status(status_code: int):
    mapping = {
        400: "请求体格式错误。重点检查 JSON、messages、model、thinking 等字段。",
        401: "鉴权失败。重点检查 DEEPSEEK_API_KEY 是否正确、是否多了空格/换行、是否拿错平台 key。",
        402: "余额不足。需要检查 DeepSeek 控制台余额。",
        422: "参数非法。常见原因：模型名错误、字段不兼容、max_tokens 等参数不合法。",
        429: "触发限流。请求太快、并发过高，或者服务侧限制。",
        500: "DeepSeek 服务端错误。通常不是你代码本身的问题。",
        503: "DeepSeek 服务过载。通常是服务侧高负载，可以稍后重试。",
    }
    return mapping.get(status_code, "非典型状态码，请优先看响应体 error.message。")


def http_request(method, url, api_key, timeout, ignore_proxy=False, json_body=None):
    session = requests.Session()
    if ignore_proxy:
        session.trust_env = False

    headers = {
        "Accept": "application/json",
        "User-Agent": "deepseek-diag/1.0",
    }
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    if json_body is not None:
        headers["Content-Type"] = "application/json"

    start = time.time()
    try:
        resp = session.request(
            method=method,
            url=url,
            headers=headers,
            json=json_body,
            timeout=timeout,
        )
        elapsed = (time.time() - start) * 1000

        print(f"{method} {url}")
        print(f"ignore_proxy={ignore_proxy}")
        print(f"status={resp.status_code}, elapsed={elapsed:.1f} ms")
        print("response headers:")
        for k in ["content-type", "x-request-id", "cf-ray", "date"]:
            if k in resp.headers:
                print(f"  {k}: {resp.headers.get(k)}")

        text = resp.text or ""
        if len(text) > 3000:
            text = text[:3000] + "\n...<truncated>"

        try:
            parsed = resp.json()
            print("response json:")
            print(json.dumps(parsed, ensure_ascii=False, indent=2))
        except Exception:
            print("response text:")
            print(text)

        if resp.ok:
            ok("HTTP 请求成功")
        else:
            fail(explain_status(resp.status_code))

        return resp.status_code, text

    except requests.exceptions.ProxyError as e:
        fail(f"代理错误：{type(e).__name__}: {e}")
        print("建议：检查 HTTP_PROXY/HTTPS_PROXY，或加 --ignore-proxy 对比测试。")
    except requests.exceptions.SSLError as e:
        fail(f"SSL 错误：{type(e).__name__}: {e}")
        print("建议：检查 CA 证书、代理 MITM、REQUESTS_CA_BUNDLE/SSL_CERT_FILE。")
    except requests.exceptions.ConnectTimeout as e:
        fail(f"连接超时：{type(e).__name__}: {e}")
        print("建议：检查服务器是否能访问 api.deepseek.com:443，或是否需要代理。")
    except requests.exceptions.ReadTimeout as e:
        fail(f"读取超时：{type(e).__name__}: {e}")
        print("建议：可能是网络抖动、服务端慢、代理慢；可增大 --timeout。")
    except requests.exceptions.ConnectionError as e:
        fail(f"连接错误：{type(e).__name__}: {e}")
        print("建议：检查 DNS、防火墙、安全组、代理、运营商网络。")
    except Exception as e:
        fail(f"未知 HTTP 异常：{type(e).__name__}: {e}")

    return None, None


def models_check(base_url, api_key, timeout, ignore_proxy):
    section("5. DeepSeek /models 检查")
    url = base_url.rstrip("/") + "/models"
    return http_request("GET", url, api_key, timeout, ignore_proxy=ignore_proxy)


def chat_check(base_url, api_key, model, timeout, ignore_proxy):
    section("6. DeepSeek /chat/completions 最小请求检查")
    url = base_url.rstrip("/") + "/chat/completions"

    payload = {
        "model": model,
        "messages": [
            {"role": "user", "content": "Reply with only: ok"}
        ],
        "stream": False,
        "max_tokens": 4,
        "thinking": {"type": "disabled"},
    }

    return http_request(
        "POST",
        url,
        api_key,
        timeout,
        ignore_proxy=ignore_proxy,
        json_body=payload,
    )


def main():
    from modules.model.forced_config import load_forced_llm_config

    load_dotenv_simple()
    forced = load_forced_llm_config()
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default=forced["base_url"])
    parser.add_argument("--model", default=forced["model"])
    parser.add_argument("--api-key-env", default=forced.get("api_key_env", "DEEPSEEK_API_KEY"))
    parser.add_argument("--timeout", type=float, default=20)
    parser.add_argument("--ignore-proxy", action="store_true", help="忽略 HTTP_PROXY/HTTPS_PROXY 等环境代理")
    parser.add_argument("--skip-chat", action="store_true", help="只测网络和 /models，不发聊天请求")
    args = parser.parse_args()
    if not args.skip_chat and str(forced.get("enabled", False)).lower() not in {"true", "1"}:
        parser.error("Chat disabled by intervention.forced_llm.enabled; use --skip-chat for diagnostics")

    api_key = os.getenv(args.api_key_env, "").strip()
    parsed = urlparse(args.base_url)
    host = parsed.hostname
    port = parsed.port or 443

    if not host:
        fail(f"base-url 不合法：{args.base_url}")
        sys.exit(2)

    print_env(args.api_key_env)

    if not api_key:
        section("结论")
        fail(f"没有读取到 {args.api_key_env}")
        print(f"请先设置：export {args.api_key_env}='你的 DeepSeek API Key'")
        sys.exit(2)

    print(f"\nBase URL: {args.base_url}")
    print(f"Model: {args.model}")
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Ignore proxy: {args.ignore_proxy}")

    resolve_host(host)
    tcp_check(host, port, args.timeout)
    tls_check(host, port, args.timeout)

    models_check(args.base_url, api_key, args.timeout, args.ignore_proxy)

    if not args.skip_chat:
        chat_check(args.base_url, api_key, args.model, args.timeout, args.ignore_proxy)

    section("7. 快速判断建议")
    print("""
看上面的第一个失败点：

- DNS 失败：优先查 /etc/resolv.conf、公司 DNS、机房 DNS、域名污染。
- TCP 失败：优先查服务器安全组、防火墙、iptables、出站 443、是否需要代理。
- TLS 失败：优先查 CA 证书、代理 MITM、系统时间、REQUESTS_CA_BUNDLE/SSL_CERT_FILE。
- /models 401：API key 错、环境变量没生效、key 里有空格/换行。
- /models 402：余额不足。
- /chat 422：模型名或参数错误；可试 --model deepseek-v4-pro 或 --model deepseek-v4-flash。
- 429：限流或并发过高。
- 500/503：服务侧异常或过载，代码大概率不是主因。
- 开了代理但失败：运行一次 python3 deepseek_diag.py --ignore-proxy 对比。
- 没开代理但 TCP 超时：服务器所在网络可能需要代理才能出站访问。
""")


if __name__ == "__main__":
    main()