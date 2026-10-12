#!/usr/bin/env python3
"""Run the Session × Domain evidence audit on one finished checkpoint."""
import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from humanlike_validation.session_domain_evidence_audit import compare, fingerprint, parse, prepare, summarize
from runshells.run_psi_bench_eval import add_judge_arguments, load_judge_settings


def write_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    os.replace(temp, path)


def write_jsonl(path, values):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in values), encoding="utf-8")
    os.replace(temp, path)


def make_call(routing, key, metadata, max_tokens):
    if not key:
        raise ValueError("Missing DeepSeek API key")
    from openai import OpenAI
    import threading
    from modules.model.endpoint_pool import next_endpoint
    local = threading.local()
    endpoints = metadata["endpoints"]

    def call(prompt):
        if not hasattr(local, "clients"):
            local.clients = {url: OpenAI(api_key=key, base_url=url, timeout=metadata["timeout_seconds"], max_retries=0)
                             for url in endpoints}
        url = next_endpoint(routing, endpoints)
        result = local.clients[url].chat.completions.create(
            model=metadata["model"], messages=[{"role": "user", "content": prompt}],
            temperature=0, response_format={"type": "json_object"},
            extra_body={"thinking": {"type": "disabled"}}, max_tokens=max_tokens)
        if not result.choices or result.choices[0].finish_reason == "length":
            raise ValueError("Truncated judge response")
        return result.choices[0].message.content.strip()
    return call


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--patient", default="卡布达")
    parser.add_argument("--doctor", default="蜻蜓队长")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "experiment_data")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--max-tokens", type=int, default=8192)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--retry-errors", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--show-prompts", type=int, default=0)
    add_judge_arguments(parser)
    args = parser.parse_args()
    if args.workers < 1 or args.attempts < 1 or args.max_tokens < 1024:
        parser.error("Invalid workers, attempts, or max-tokens")
    windows = prepare(args.run, args.patient, args.doctor)
    if args.dry_run or args.show_prompts:
        print(json.dumps({"windows": len(windows), "intra": sum(x["kind"] == "intra" for x in windows),
                          "inter": sum(x["kind"] == "inter" for x in windows),
                          "evidence_counts": [(x["id"], len(x["evidence"])) for x in windows]}, ensure_ascii=False))
        for x in windows[:args.show_prompts]:
            print(json.dumps({"id": x["id"], "evidence_prompt": x["evidence_prompt"],
                              "graph_prompt": x["graph_prompt"]}, ensure_ascii=False, indent=2))
        return 0
    args.backend = "deepseek_api"
    routing, key, metadata = load_judge_settings(args)
    call = make_call(routing, key, metadata, args.max_tokens)
    output = args.output_root / args.run.name / "state_transition_evidence_audit_v2_session_domain_deepseek_api"
    identity = fingerprint({"windows": windows, "metadata": metadata, "max_tokens": args.max_tokens})
    manifest_file, item_file = output / "manifest.json", output / "items.jsonl"
    if output.exists() and not args.resume:
        raise ValueError(f"Output already exists: {output}; use --resume")
    if args.resume and output.exists() and (not manifest_file.is_file() or json.loads(manifest_file.read_text())["identity"] != identity):
        raise ValueError("Resume identity mismatch")
    output.mkdir(parents=True, exist_ok=True)
    existing = {}
    if item_file.is_file():
        source_windows = {window["id"]: window for window in windows}
        for line in item_file.read_text(encoding="utf-8").splitlines():
            item = json.loads(line)
            if item["id"] in existing or item["id"] not in source_windows:
                raise ValueError("Duplicate or unexpected cached item")
            if item.get("status") == "error":
                for kind in ("evidence", "graph"):
                    if item.get(f"{kind}_ratings") is None and item.get(f"{kind}_raw"):
                        try:
                            item[f"{kind}_ratings"] = parse(item[f"{kind}_raw"], kind, source_windows[item["id"]])
                            item[f"{kind}_error"] = None
                        except (ValueError, KeyError, TypeError):
                            pass
                if item.get("evidence_ratings") is not None and item.get("graph_ratings") is not None:
                    item["status"] = "ok"
                    item["domain_comparison"] = compare(item)
            existing[item["id"]] = item
    manifest = {"schema_version": 2, "status": "running", "identity": identity,
                "model": metadata, "run": str(args.run.resolve()), "patient": args.patient,
                "doctor": args.doctor, "windows": len(windows)}
    write_json(manifest_file, manifest)

    def evaluate(window):
        result = dict(window)
        for kind, prefix in (("evidence", "evidence"), ("graph", "graph")):
            result[f"{prefix}_ratings"] = None
            result[f"{prefix}_raw"] = None
            result[f"{prefix}_error"] = None
            for attempt in range(args.attempts):
                try:
                    raw = call(window[f"{prefix}_prompt"])
                    result[f"{prefix}_raw"] = raw
                    result[f"{prefix}_ratings"] = parse(raw, kind, window)
                    break
                except Exception as exc:
                    result[f"{prefix}_error"] = f"{type(exc).__name__}: {exc}"
                    if attempt + 1 < args.attempts:
                        time.sleep(min(2 ** attempt * 2, 30))
            if result[f"{prefix}_ratings"] is not None:
                result[f"{prefix}_error"] = None
        result["status"] = "ok" if result["evidence_ratings"] is not None and result["graph_ratings"] is not None else "error"
        result["domain_comparison"] = compare(result)
        return result

    pending = [w for w in windows if w["id"] not in existing or
               (args.retry_errors and existing[w["id"]].get("status") != "ok")]
    done = {k: v for k, v in existing.items() if k not in {w["id"] for w in pending}}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(evaluate, w): w for w in pending}
        for future in as_completed(futures):
            row = future.result()
            done[row["id"]] = row
            write_jsonl(item_file, [done[w["id"]] for w in windows if w["id"] in done])
            print(f"{len(done)}/{len(windows)} {row['id']} {row['status']}", flush=True)
    ordered = [done[w["id"]] for w in windows]
    write_jsonl(item_file, ordered)
    summary = summarize(ordered)
    summary.update(run=str(args.run.resolve()), patient=args.patient, model=metadata["model"])
    write_json(output / "summary.json", summary)
    manifest.update(status="complete" if not summary["errors"] else "partial", errors=summary["errors"])
    write_json(manifest_file, manifest)
    print(json.dumps({"output": str(output), "summary": summary}, ensure_ascii=False, indent=2))
    return bool(summary["errors"])


if __name__ == "__main__":
    raise SystemExit(main())
