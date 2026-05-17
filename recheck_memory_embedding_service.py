"""调用外置记忆服务的 embedding 重探活接口。"""

from __future__ import annotations

import argparse
import json
import sys

from modules.ec_doll_memory_service_client import ECDollMemoryServiceClient

DEFAULT_BASE_URL = "http://localhost:8031"
DEFAULT_TIMEOUT_SECONDS = 30.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="重新探测外置记忆服务的 embedding 可用性")
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help="外置记忆服务地址，默认 http://localhost:8031",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=DEFAULT_TIMEOUT_SECONDS,
        help="请求超时时间（秒）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 这个脚本用于手动触发 /api/embed/recheck。
    # 典型场景：
    # 1. embedding 服务比记忆服务启动得更晚；
    # 2. 切换了 embedding 相关配置；
    # 3. 服务之前退化到了 MOCK，需要在不重启记忆服务的前提下重新探活。
    client = ECDollMemoryServiceClient(
        base_url=str(args.base_url or DEFAULT_BASE_URL),
        timeout=float(args.timeout or DEFAULT_TIMEOUT_SECONDS),
    )

    try:
        # 直接透传服务端返回，方便观察当前探活结果和状态描述。
        result = client.recheck_embedding_service()
    except Exception as exc:
        print("[ERROR] 调用 /api/embed/recheck 失败：{}".format(exc), file=sys.stderr)
        return 1

    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
