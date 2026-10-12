#!/usr/bin/env python3
"""Run the frozen 0929 LRN longitudinal case evaluation and aggregate results."""

import argparse
import csv
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
GROUPS = ("G1", "G2", "G3", "G4", "G5", "G6", "G7", "G9")
METRICS = ("stable_fact_contradiction_rate", "cross_session_contradiction_rate",
           "cross_context_contradiction_rate")
LABELS = ("supported", "plausible_extension", "unsupported", "contradiction")
PATTERN = re.compile(r"batch-0929-(\d\d)-Counsel-LRN-(G1|G2|G3|G4|G5|G6|G7|G9)-MOD--")


def save_csv(path, rows):
    if not rows:
        return
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def collect(output):
    runs = []
    for group in GROUPS:
        for number in range(1, 6):
            run = output / f"{group}_R{number:02d}"
            summary_path, manifest_path = run / "summary.json", run / "manifest.json"
            if not summary_path.exists() or not manifest_path.exists():
                continue
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            row = {"group": group, "repeat": number, "status": manifest["status"],
                   "snapshot": manifest["snapshot"], "snapshot_sha256": hashlib.sha256(
                       Path(manifest["snapshot"]).read_bytes()).hexdigest(),
                   "sessions": summary["sessions"], "patient_turns": summary["patient_turns"],
                   "cbt_turns": summary["turns_by_context"].get("cbt", 0),
                   "resident_turns": summary["turns_by_context"].get("resident", 0),
                   "other_turns": summary["turns_by_context"].get("other", 0),
                   "claims": summary["claims"], "errors": summary["extraction_or_judge_errors"],
                   "extraction_fallbacks": summary.get("extraction_fallbacks", 0)}
            row.update({label: summary["labels"][label] for label in LABELS})
            for metric in METRICS:
                row[metric + "_n"] = summary[metric]["numerator"]
                row[metric + "_d"] = summary[metric]["denominator"]
                row[metric] = summary[metric]["rate"]
            row["coverage_n"] = summary["stable_fact_coverage"]["numerator"]
            row["coverage_d"] = summary["stable_fact_coverage"]["denominator"]
            runs.append(row)
    save_csv(output / "run_summary.csv", runs)
    groups = []
    for group in GROUPS:
        subset = [row for row in runs if row["group"] == group]
        if not subset:
            continue
        row = {"group": group, "runs": len(subset),
               "complete_runs": sum(item["status"] == "complete" for item in subset),
               "sessions": sum(item["sessions"] for item in subset),
               "patient_turns": sum(item["patient_turns"] for item in subset),
               "cbt_turns": sum(item["cbt_turns"] for item in subset),
               "resident_turns": sum(item["resident_turns"] for item in subset),
               "other_turns": sum(item["other_turns"] for item in subset),
               "claims": sum(item["claims"] for item in subset),
               "errors": sum(item["errors"] for item in subset),
               "extraction_fallbacks": sum(item["extraction_fallbacks"] for item in subset),
               "zero_claim_runs": sum(item["claims"] == 0 for item in subset)}
        row.update({label: sum(item[label] for item in subset) for label in LABELS})
        for metric in METRICS:
            n = sum(item[metric + "_n"] for item in subset)
            d = sum(item[metric + "_d"] for item in subset)
            row[metric + "_n"], row[metric + "_d"] = n, d
            row[metric] = n / d if d else None
        row["coverage_n"] = sum(item["coverage_n"] for item in subset)
        row["coverage_d"] = sum(item["coverage_d"] for item in subset)
        row["coverage_rate"] = row["coverage_n"] / row["coverage_d"] if row["coverage_d"] else None
        row["plausible_extension_rate"] = row["plausible_extension"] / row["claims"] if row["claims"] else None
        row["unsupported_invention_rate"] = row["unsupported"] / row["claims"] if row["claims"] else None
        groups.append(row)
    save_csv(output / "group_summary.csv", groups)
    flags = []
    for row in runs:
        path = output / f"{row['group']}_R{row['repeat']:02d}" / "claims.jsonl"
        claims = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        for claim in claims:
            findings = []
            quote = claim["quote"]
            if quote not in claim["text"]:
                findings.append("quote_not_in_patient_text")
            if re.search(r"尚未|从未|没有|未曾|未发|没发|没有完成|没画完", claim["claim"]) and not re.search(
                    r"没|未|不曾", quote):
                findings.append("negative_claim_not_supported_by_short_quote")
            if claim["verdict"].get("citation_audit"):
                findings.append("judge_citation_adjusted")
            if findings:
                flags.append({"group": row["group"], "repeat": row["repeat"],
                              "claim_id": claim["id"], "flags": ";".join(findings),
                              "context": claim["context"], "label": claim["verdict"]["label"],
                              "claim": claim["claim"], "quote": quote, "source": claim["source"]})
    save_csv(output / "audit_flags.csv", flags)
    (output / "comparison.json").write_text(json.dumps({"runs": runs, "groups": groups,
                                                   "audit_flag_count": len(flags)},
                                                  ensure_ascii=False, indent=2), encoding="utf-8")
    return runs, groups


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:18000/v1")
    parser.add_argument("--model", default="qwen3-8b-vllm")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--recheck-complete", action="store_true",
                        help="Reapply current deterministic filters to all cached turns")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if not args.aggregate_only:
        paths = {}
        for path in (ROOT / "results/checkpoints").glob("batch-0929-*"):
            match = PATTERN.match(path.name)
            if not path.is_dir() or not match:
                continue
            key = (match.group(2), int(match.group(1)))
            if key in paths:
                raise ValueError(f"Duplicate run: {key}")
            paths[key] = path
        expected = {(group, number) for group in GROUPS for number in range(1, 6)}
        if set(paths) != expected:
            raise ValueError(f"0929 group/repeat mismatch: missing={expected-set(paths)}, extra={set(paths)-expected}")
        runner = ROOT / "runshells/humanlike_validation/run_longitudinal_case_fidelity.py"
        failed = []
        for group in GROUPS:
            for number in range(1, 6):
                run = paths[group, number]
                snapshot = sorted(run.glob("simulate-*.json"))[-1]
                destination = args.output / f"{group}_R{number:02d}"
                manifest = destination / "manifest.json"
                if (manifest.exists() and not args.recheck_complete
                        and json.loads(manifest.read_text(encoding="utf-8"))["status"] == "complete"):
                    print(f"[0929] skip complete {group} R{number:02d}", flush=True)
                    continue
                cmd = [sys.executable, str(runner), str(run), "--snapshot", str(snapshot),
                       "--output", str(destination), "--base-url", args.base_url,
                       "--model", args.model, "--workers", str(args.workers)]
                if manifest.exists():
                    cmd.append("--resume")
                print(f"[0929] start {group} R{number:02d}: {snapshot.name}", flush=True)
                with (args.output / f"{group}_R{number:02d}.log").open("a", encoding="utf-8") as log:
                    outcome = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT)
                print(f"[0929] done {group} R{number:02d}: exit={outcome.returncode}", flush=True)
                if outcome.returncode:
                    failed.append(f"{group}_R{number:02d}")
        if failed:
            print("Failed runs: " + ", ".join(failed), file=sys.stderr)
    runs, groups = collect(args.output)
    print(json.dumps({"summarized_runs": len(runs), "groups": len(groups),
                      "complete_runs": sum(row["status"] == "complete" for row in runs)},
                     ensure_ascii=False), flush=True)
    if not args.aggregate_only and (failed or len(runs) != 40 or any(row["status"] != "complete" for row in runs)):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
