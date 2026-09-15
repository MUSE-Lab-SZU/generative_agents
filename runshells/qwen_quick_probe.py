#!/usr/bin/env python3
"""Small standard-library-only vLLM A/B runner and project integration probe."""
import argparse
import ast
import concurrent.futures
import json
import math
import os
from pathlib import Path
import re
import signal
import socket
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
MODEL_ROOT = "/share/home/tm866039793920000/a874457430/MODEL"
ACTIVE = None


def api(base, path, payload=None, timeout=45):
    data = None if payload is None else json.dumps(payload).encode()
    req = urllib.request.Request(base.rstrip("/") + path, data=data, headers={
        "Content-Type": "application/json",
        "Authorization": "Bearer " + os.environ.get("API_KEY", "EMPTY"),
    })
    try:
        with urllib.request.urlopen(req, timeout=timeout) as response:
            return json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"HTTP {exc.code}: {exc.read().decode(errors='replace')[:1200]}") from exc


def project_functions():
    # Execute the two actual project functions without importing the simulation dependencies.
    source = ROOT / "modules/model/llm_model.py"
    tree = ast.parse(source.read_text())
    names = {"prepare_prompt_for_model", "strip_qwen_think_tags"}
    selected = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    if len(selected) != 2:
        raise RuntimeError("项目 prompt/think 清理函数已变化，请更新探针")
    namespace = {"re": re}
    exec(compile(ast.Module(body=selected, type_ignores=[]), str(source), "exec"), namespace)
    return namespace


def chat(base, model, prompt, timeout, thinking=False, tokens=128, fixed=False):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "temperature": 0, "max_tokens": tokens, "stream": False}
    if thinking is not None:
        body["chat_template_kwargs"] = {"enable_thinking": thinking}
    if fixed:
        body["ignore_eos"] = True
    started = time.perf_counter()
    response = api(base, "/chat/completions", body, timeout)
    elapsed = time.perf_counter() - started
    choice = response["choices"][0]
    message = choice["message"]
    usage = response.get("usage", {})
    return {"seconds": elapsed, "content": message.get("content") or "",
            "reasoning": message.get("reasoning_content") or message.get("reasoning") or "",
            "finish_reason": choice.get("finish_reason"), "usage": usage}


def check(base, model, timeout):
    models = api(base, "/models", timeout=timeout)
    if model not in [m["id"] for m in models["data"]]:
        raise RuntimeError(f"/models 未列出服务名 {model}")
    functions = project_functions()
    prompt = '只输出一个 JSON 对象，不要 Markdown：{"action":"stay","角色":"卡布达","value":42}。'
    prepared = functions["prepare_prompt_for_model"](prompt, model)
    rows = []
    for name, text, thinking in [
        ("project_current_request", prepared, None),
        ("explicit_thinking_off", prepared, False),
        ("thinking_on_control", "计算 17*19，先推理再回答。", True),
    ]:
        try:
            row = chat(base, model, text, timeout, thinking, tokens=192)
            cleaned = functions["strip_qwen_think_tags"](row["content"])
            row["cleaned_content"] = cleaned
            row["reasoning_observed"] = bool(row["reasoning"].strip() or "<think>" in row["content"] or "</think>" in row["content"])
            if thinking is not True:
                try:
                    valid = json.loads(cleaned) == {"action": "stay", "角色": "卡布达", "value": 42}
                except (ValueError, TypeError):
                    valid = False
                row["pass"] = valid and not row["reasoning_observed"] and row["finish_reason"] == "stop"
            else:
                # A positive control detects ignored kwargs. A short cap may truncate reasoning.
                row["pass"] = row["reasoning_observed"]
        except Exception as exc:
            row = {"pass": False, "error": str(exc)}
        row["case"] = name
        rows.append(row)
        print(f"  {name}: {'PASS' if row['pass'] else 'FAIL/INCONCLUSIVE'}", flush=True)
    return {"model": model, "base_url": base, "project_prompt_suffix": prepared[len(prompt):],
            "pass": all(row["pass"] for row in rows), "cases": rows,
            "scope": "真实项目 prompt/清理函数 + HTTP；不运行完整仿真，不验证 embedding"}


def benchmark(base, model, args):
    prompt = ("居民卡布达今天在村中散步，随后与金龟次郎聊天。请依据日常生活场景继续写具体、自然的中文描述。" * 16)
    # One warmup for each measured concurrency. Warmup excluded from timing.
    rows = []
    for concurrency in (1, 4):
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            list(pool.map(lambda i: chat(base, model, f"预热{i}：" + prompt, args.timeout, tokens=32, fixed=True), range(concurrency)))
            started = time.perf_counter()
            records = list(pool.map(lambda i: chat(base, model, f"样本{i}：" + prompt, args.timeout,
                                                   tokens=128, fixed=True), range(args.requests)))
        wall = time.perf_counter() - started
        # Fixed generation compares compute throughput, not helpfulness or natural stop behavior.
        for record in records:
            if record["usage"].get("completion_tokens") != 128:
                raise RuntimeError("未生成固定 128 tokens，拒绝输出误导性吞吐结果")
            if record["reasoning"].strip() or "<think>" in record["content"]:
                raise RuntimeError("测速发现 reasoning，关闭思考未生效")
        latencies = sorted(r["seconds"] for r in records)
        row = {"concurrency": concurrency, "requests": len(records), "wall_seconds": wall,
               "output_tokens_per_second": sum(r["usage"]["completion_tokens"] for r in records) / wall,
               "requests_per_second": len(records) / wall,
               "median_latency_seconds": statistics.median(latencies),
               "max_latency_seconds": max(latencies),
               "mean_prompt_tokens": statistics.mean(r["usage"]["prompt_tokens"] for r in records)}
        rows.append(row)
        print(f"  并发 {concurrency}: {row['output_tokens_per_second']:.1f} output tok/s, "
              f"中位延迟 {row['median_latency_seconds']:.2f}s", flush=True)
    return rows


def embedding_check(args):
    """Six small batches for three replicas; no persistent index is read/written."""
    urls = args.embedding_urls.split(",")
    if len(urls) != 3:
        raise ValueError("embedding 检查要求三个逗号分隔端点")
    documents = ["卡布达昨晚失眠，今天感到疲倦。", "金龟次郎在商店购买了苹果。",
                 "蜻蜓队长建议卡布达每天散步并记录心情。", "村里的咖啡馆早上八点营业。"]
    query = "医生建议卡布达做什么？"
    instructed = "Instruct: Given a query, retrieve relevant memories that answer the query\nQuery: " + query
    all_rows, references = [], {}

    def cosine(a, b):
        na, nb = math.sqrt(sum(x*x for x in a)), math.sqrt(sum(x*x for x in b))
        if not na or not nb:
            raise ValueError("零向量")
        return sum(x*y for x, y in zip(a, b)) / (na * nb)

    for url in urls:
        for mode, q in [("plain_query", query), ("instructed_query", instructed)]:
            started = time.perf_counter()
            response = api(url, "/embeddings", {"model": args.model, "input": documents + [q]}, args.timeout)
            elapsed = time.perf_counter() - started
            data = sorted(response["data"], key=lambda item: item["index"])
            if [item["index"] for item in data] != list(range(5)):
                raise ValueError("embedding batch 返回数量/索引错误")
            vectors = [item["embedding"] for item in data]
            dim = len(vectors[0])
            if not dim or any(len(v) != dim or not all(math.isfinite(x) for x in v) for v in vectors):
                raise ValueError("embedding 维度不一致或存在 NaN/Inf")
            ref = references.setdefault(mode, vectors)
            if len(ref[0]) != dim:
                raise ValueError("副本间向量维度不同")
            agreement = min(cosine(a, b) for a, b in zip(ref, vectors))
            scores = [cosine(vectors[-1], v) for v in vectors[:-1]]
            top = max(range(4), key=scores.__getitem__)
            row = {"base_url": url, "mode": mode, "dimensions": dim, "seconds": elapsed,
                   "norms": [math.sqrt(sum(x*x for x in v)) for v in vectors],
                   "replica_min_cosine": agreement, "top_document": top, "scores": scores,
                   "pass": agreement > 0.999 and top == 2}
            all_rows.append(row)
            print(f"  {url} {mode}: dim={dim}, top={top}, replica cosine={agreement:.6f}", flush=True)
    return {"cases": all_rows, "pass": all(row["pass"] for row in all_rows),
            "scope": "仅接口、批量、有限向量、副本一致性及一个检索冒烟样例；不是质量/吞吐评测。更换模型必须重建旧索引。"}


def stop_owned():
    global ACTIVE
    if ACTIVE is None:
        return
    proc = ACTIVE
    ACTIVE = None
    # Only the isolated process group created by this tool, including vLLM workers.
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        proc.wait()
        return
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        proc.poll()
        try:
            os.killpg(proc.pid, 0)
        except ProcessLookupError:
            break
        time.sleep(0.25)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    proc.wait(timeout=10)


def gpu_info(gpu):
    text = subprocess.check_output(["nvidia-smi", "-i", gpu,
        "--query-gpu=name,memory.total,memory.used", "--format=csv,noheader,nounits"], text=True).strip()
    if "\n" in text:
        raise RuntimeError("只允许选择一张 GPU")
    if float(text.rsplit(",", 1)[1]) > 1000:
        raise RuntimeError(f"GPU {gpu} 不空闲：{text}。请先释放该卡；脚本不会停止现有服务。")
    return text


def compare(args, report, output):
    global ACTIVE
    version = subprocess.check_output([args.vllm, "--version"], text=True, stderr=subprocess.STDOUT).strip()
    versions = re.findall(r"\b(\d+)\.(\d+)\.(\d+)", version)
    if not versions or tuple(map(int, versions[-1])) < (0, 17, 0):
        raise RuntimeError(f"需要 vLLM >= 0.17.0；当前输出：{version}")
    report["vllm_version"] = version
    report["gpu"] = gpu_info(args.gpu)
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", args.port))
    for directory in (args.old_model, args.new_model):
        if not (Path(directory) / "config.json").is_file():
            raise RuntimeError(f"模型目录缺少 config.json：{directory}")
    base = f"http://127.0.0.1:{args.port}/v1"
    report["models"] = []
    for label, directory, name in [("old", args.old_model, "qwen3-8b-vllm"),
                                    ("new", args.new_model, "qwen3.5-9b-vllm")]:
        deadline = time.monotonic() + 30
        while True:
            try:
                gpu_info(args.gpu)
                break
            except RuntimeError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        command = [args.vllm, "serve", directory, "--host", "127.0.0.1", "--port", str(args.port),
                   "--served-model-name", name, "--dtype", "bfloat16", "--tensor-parallel-size", "1",
                   "--gpu-memory-utilization", "0.90", "--max-model-len", "4096",
                   "--max-num-seqs", "4", "--max-num-batched-tokens", "4096",
                   "--no-enable-prefix-caching", "--reasoning-parser", "qwen3",
                   "--default-chat-template-kwargs", '{"enable_thinking":false}',
                   "--generation-config", "vllm"]
        if label == "new":
            command.append("--language-model-only")
        log_path = output / f"{label}_server.log"
        entry = {"label": label, "model": name, "directory": directory, "command": command}
        report["models"].append(entry)
        print(f"启动 {name}，GPU={args.gpu}，日志 {log_path}", flush=True)
        with log_path.open("w") as logfile:
            ACTIVE = subprocess.Popen(command, env={**os.environ, "CUDA_VISIBLE_DEVICES": args.gpu},
                                      stdout=logfile, stderr=subprocess.STDOUT, start_new_session=True)
            try:
                deadline = time.monotonic() + args.startup_timeout
                while True:
                    if ACTIVE.poll() is not None:
                        raise RuntimeError(f"模型启动失败，请查看 {log_path}")
                    try:
                        ids = [m["id"] for m in api(base, "/models", timeout=2)["data"]]
                        if name in ids:
                            break
                    except Exception:
                        pass
                    if time.monotonic() >= deadline:
                        raise RuntimeError(f"启动超时，请查看 {log_path}")
                    time.sleep(2)
                entry["benchmark"] = benchmark(base, name, args)
                entry["integration"] = check(base, name, args.timeout)
            finally:
                stop_owned()
                (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    old, new = report["models"]
    report["new_over_old_throughput"] = {
        str(a["concurrency"]): b["output_tokens_per_second"] / a["output_tokens_per_second"]
        for a, b in zip(old["benchmark"], new["benchmark"])}
    print("新/旧吞吐比（>1 更快）:", report["new_over_old_throughput"], flush=True)
    report["pass"] = all(m["integration"]["pass"] for m in report["models"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["compare", "check", "embedding"])
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=18990)
    parser.add_argument("--old-model", default=os.environ.get("MODEL_ROOT", MODEL_ROOT) + "/Qwen3-8B")
    parser.add_argument("--new-model", default=os.environ.get("MODEL_ROOT", MODEL_ROOT) + "/Qwen3.5-9B")
    parser.add_argument("--vllm", default=os.environ.get("VLLM_BIN", "vllm"))
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3.5-9b-vllm")
    parser.add_argument("--embedding-urls", default="http://127.0.0.1:18001/v1,http://127.0.0.1:18004/v1,http://127.0.0.1:18005/v1")
    parser.add_argument("--timeout", type=float, default=45)
    parser.add_argument("--startup-timeout", type=float, default=600)
    parser.add_argument("--requests", type=int, default=6, help="每种并发的请求数，默认 6")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()
    if args.requests < 4 or args.timeout <= 0 or args.startup_timeout <= 0:
        parser.error("requests 必须 >=4，超时必须 >0")
    output = Path(args.output) if args.output else ROOT / ".vllm" / f"quick_probe_{time.strftime('%Y%m%d_%H%M%S')}_{os.getpid()}"
    output.mkdir(parents=True, exist_ok=True)
    report = {"settings": vars(args), "pass": False}
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(143))
    try:
        if args.mode == "compare":
            compare(args, report, output)
        elif args.mode == "embedding":
            report["embedding"] = embedding_check(args)
            report["pass"] = report["embedding"]["pass"]
        else:
            report["integration"] = check(args.base_url, args.model, args.timeout)
            report["pass"] = report["integration"]["pass"]
    except Exception as exc:
        report["error"] = str(exc)
        print(f"ERROR: {exc}", file=sys.stderr)
    finally:
        stop_owned()
        (output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        print(f"结果：{output / 'report.json'}", flush=True)
    return 0 if report["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
