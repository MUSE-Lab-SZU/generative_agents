#!/usr/bin/env python3
"""Audit 0929-G1 V3 pre-state alignment, then run local-Qwen conformity."""
import argparse
import json
import sys
from pathlib import Path
from urllib.request import ProxyHandler, Request, build_opener

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from humanlike_validation.psyche_state_expression import (
    extract_all, judge_all, negative_control, operationalize, summarize, verify_all,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "humanlike_outputs/psyche_state_expression_0929_g1")
    parser.add_argument("--mode", choices=("audit", "prepare", "evaluate"), default="audit")
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default=None)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--negative-limit", type=int, default=30)
    args = parser.parse_args()
    rows, manifest = extract_all(ROOT, args.output)
    print(json.dumps({"phase": "alignment_audit", "runs": len(manifest["runs"]),
                      "patient_utterances": manifest["patient_utterances"],
                      "accepted_patient_utterances": manifest["accepted_patient_utterances"],
                      "aligned": manifest["aligned"], "per_run": manifest["runs"]}, ensure_ascii=False), flush=True)
    if args.mode == "audit":
        return
    opener = build_opener(ProxyHandler({}))
    base = args.base_url.rstrip("/")
    with opener.open(Request(base + "/models"), timeout=10) as response:
        models = json.load(response).get("data", [])
    available = [m.get("id") for m in models if "qwen" in str(m.get("id", "")).lower()]
    if not available:
        raise ValueError(f"No local Qwen model at {base}")
    model = args.model or available[0]
    if model not in available:
        raise ValueError(f"Requested model {model} not served at {base}; available={available}")

    def call(prompt):
        body = json.dumps({"model": model, "temperature": 0, "max_tokens": 600,
                           "messages": [{"role": "system", "content": "你是离线心理语言研究编码员。严格执行用户指定的 JSON schema。不要回显输入资料或解释过程。"},
                                        {"role": "user", "content": prompt + "\n/nothink"}],
                           "response_format": {"type": "json_object"}}, ensure_ascii=False).encode()
        request = Request(base + "/chat/completions", data=body,
                          headers={"Content-Type": "application/json", "Authorization": "Bearer EMPTY"}, method="POST")
        with opener.open(request, timeout=120) as response:
            answer = json.load(response)
        if not answer.get("choices") or answer["choices"][0].get("finish_reason") == "length":
            raise ValueError("Empty or truncated local judge response")
        return answer["choices"][0]["message"].get("content") or ""

    catalog = operationalize(rows, args.output, call, workers=args.workers)
    counts = {"operationalizable": sum(v.get("status") == "operationalizable" for v in catalog.values()),
              "state_not_operationalizable": sum(v.get("status") == "state_not_operationalizable" for v in catalog.values()),
              "errors": sum(bool(v.get("error")) for v in catalog.values())}
    manifest.update({"judge": {"base_url": base, "model": model, "temperature": 0, "max_tokens": 600},
                     "construct_catalog": counts, "status": "construct_audited"})
    (args.output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"phase": "construct_audit", "unique_states": len(catalog), **counts,
                      "operationalizable_messages": sum(r.get("operationalization", {}).get("status") == "operationalizable" for r in rows)}, ensure_ascii=False), flush=True)
    if args.mode == "prepare":
        return
    first_pass = judge_all(rows, args.output, call, workers=args.workers)
    errors = [r for r in first_pass if r.get("error")]
    print(json.dumps({"phase": "judge_first_pass", "results": len(first_pass), "errors": len(errors)}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit("Judge errors present; rerun to retry cached errors")
    verdicts = verify_all(first_pass, args.output, call, workers=args.workers)
    errors = [r for r in verdicts if r.get("error") or r.get("verification_error")]
    print(json.dumps({"phase": "quote_verification", "results": len(verdicts), "errors": len(errors)}, ensure_ascii=False), flush=True)
    if errors:
        raise SystemExit("Quote verification errors present; rerun to retry")
    controls = negative_control(rows, verdicts, args.output, call, workers=min(args.workers, 4), limit=args.negative_limit)
    summary = summarize(rows, verdicts, controls, args.output)
    print(json.dumps({"phase": "complete", "summary": summary}, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
