#!/usr/bin/env python3
"""Evaluate one saved checkpoint using a local OpenAI-compatible judge."""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanlike_validation.longitudinal_case_fidelity import (
    EXTRACT_PROMPT, evaluate, latest_snapshot, load_fact_catalog, load_patient_turns,
    no_case_fact_cues, read_json,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--snapshot", type=Path, help="Freeze one simulate-*.json; recommended for active archives")
    parser.add_argument("--facts", type=Path, default=ROOT / "humanlike_validation/lin_ruoning_longitudinal_facts.json")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--patient", default="林若宁")
    parser.add_argument("--doctor", default="蜻蜓队长")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3-8b-vllm")
    parser.add_argument("--api-key", default="EMPTY")
    parser.add_argument("--timeout", type=float, default=120)
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.snapshot:
        snapshot_path = args.snapshot
        if snapshot_path.parent.resolve() != args.run_dir.resolve() or not snapshot_path.name.startswith("simulate-"):
            parser.error("--snapshot must be a simulate-*.json in run_dir")
        snapshot = read_json(snapshot_path)
    else:
        snapshot_path, snapshot = latest_snapshot(args.run_dir)
    catalog, hashes = load_fact_catalog(args.facts, snapshot, args.patient)
    turns = load_patient_turns(args.run_dir, snapshot, args.patient, args.doctor)
    if args.dry_run:
        print(json.dumps({"snapshot": str(snapshot_path), "snapshot_time": snapshot["time"],
                          "facts": len(catalog["facts"]), "source_hashes": hashes,
                          "turns": len(turns), "contexts": {kind: sum(t["context"] == kind for t in turns)
                                                         for kind in ("cbt", "resident", "other")}},
                         ensure_ascii=False, indent=2))
        return
    def call(prompt):
        no_fact_fallback = (prompt.startswith(EXTRACT_PROMPT)
                            and no_case_fact_cues(json.loads(prompt[len(EXTRACT_PROMPT):])["patient"]))
        for max_tokens in (1400, 3200):
            body = json.dumps({"model": args.model, "temperature": 0, "max_tokens": max_tokens,
                               "messages": [{"role": "user", "content": prompt + "\n/nothink"}],
                               "response_format": {"type": "json_object"}}, ensure_ascii=False).encode()
            request = Request(args.base_url.rstrip("/") + "/chat/completions", data=body,
                              headers={"Content-Type": "application/json",
                                       "Authorization": "Bearer " + args.api_key}, method="POST")
            with build_opener(ProxyHandler({})).open(request, timeout=args.timeout) as response:
                data = json.load(response)
            if data.get("choices") and data["choices"][0].get("finish_reason") != "length":
                content = data["choices"][0]["message"].get("content") or ""
                if content:
                    return content
            if no_fact_fallback:
                raise ValueError("Empty or truncated local judge reply on no-case-fact turn")
        raise ValueError("Empty or truncated local judge reply after extended retry")

    summary = evaluate(args.run_dir, args.facts, args.output, call,
                       patient=args.patient, doctor=args.doctor, resume=args.resume,
                       judge_metadata={"model": args.model, "base_url": args.base_url,
                                       "temperature": 0, "max_tokens": 1400}, workers=args.workers,
                       snapshot_file=snapshot_path)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    if summary["extraction_or_judge_errors"]:
        raise SystemExit("Some turns were not evaluated; inspect errors.json and rerun with --resume")


if __name__ == "__main__":
    main()
