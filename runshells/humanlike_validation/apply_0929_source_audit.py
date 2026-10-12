#!/usr/bin/env python3
"""Apply explicit source-text corrections without changing machine judgment files."""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from humanlike_validation.longitudinal_case_fidelity import (  # noqa: E402
    load_fact_catalog, load_patient_turns, read_json, summarize,
)
from run_longitudinal_0929_batch import GROUPS, LABELS, METRICS, save_csv  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    output = args.output
    overrides = read_json(output / "source_audit_overrides.json")["overrides"]
    by_key = {(item["run"], item["id"]): item for item in overrides}
    if len(by_key) != len(overrides):
        raise ValueError("Duplicate source audit override")
    applied = set()
    reviewed_rows, run_rows = [], []
    for group in GROUPS:
        for number in range(1, 6):
            name = f"{group}_R{number:02d}"
            run_output = output / name
            manifest = read_json(run_output / "manifest.json")
            if manifest["status"] != "complete":
                raise ValueError(f"Incomplete machine run: {name}")
            snapshot_path = Path(manifest["snapshot"])
            snapshot = read_json(snapshot_path)
            catalog, _ = load_fact_catalog(manifest["catalog"], snapshot, "林若宁")
            turns = load_patient_turns(snapshot_path.parent, snapshot, "林若宁", "蜻蜓队长")
            accepted = []
            for line in (run_output / "claims.jsonl").read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                key = (name, row["id"])
                override = by_key.get(key)
                if override is None:
                    row["source_audit"] = {"status": "retained"}
                    accepted.append(row)
                else:
                    applied.add(key)
                    row["source_audit"] = {"status": override["action"], "reason": override["reason"]}
                    if override["action"] == "exclude":
                        pass
                    elif override["action"] == "edit":
                        for field in ("claim", "quote"):
                            if field in override:
                                row["source_audit"]["machine_" + field] = row[field]
                                row[field] = override[field]
                        if row["quote"] not in row["text"]:
                            raise ValueError(f"Audited quote missing from source: {name} {row['id']}")
                        if "label" in override:
                            if override["label"] not in LABELS:
                                raise ValueError("Invalid audited label")
                            row["source_audit"]["machine_label"] = row["verdict"]["label"]
                            row["verdict"]["label"] = override["label"]
                        if "support_ids" in override:
                            row["source_audit"]["machine_support_ids"] = row["verdict"]["support_ids"]
                            row["verdict"]["support_ids"] = override["support_ids"]
                        accepted.append(row)
                    else:
                        raise ValueError(f"Unknown audit action: {override['action']}")
                reviewed_rows.append({"run": name, **row})
            summary = summarize(accepted, turns, catalog["facts"], errors=0)
            (run_output / "source_audited_summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
            item = {"group": group, "repeat": number, "run": name,
                    "patient_turns": summary["patient_turns"], "claims": summary["claims"],
                    "sessions": summary["sessions"],
                    "cbt_turns": summary["turns_by_context"].get("cbt", 0),
                    "resident_turns": summary["turns_by_context"].get("resident", 0)}
            item.update(summary["labels"])
            for metric in METRICS:
                item[metric + "_n"] = summary[metric]["numerator"]
                item[metric + "_d"] = summary[metric]["denominator"]
                item[metric] = summary[metric]["rate"]
            item["coverage_n"] = summary["stable_fact_coverage"]["numerator"]
            item["coverage_d"] = summary["stable_fact_coverage"]["denominator"]
            run_rows.append(item)
    if applied != set(by_key):
        raise ValueError(f"Unmatched audit overrides: {set(by_key)-applied}")
    (output / "source_audited_claims.jsonl").write_text(
        "".join(json.dumps(item, ensure_ascii=False) + "\n" for item in reviewed_rows), encoding="utf-8")
    save_csv(output / "source_audited_run_summary.csv", run_rows)
    groups = []
    for group in GROUPS:
        rows = [item for item in run_rows if item["group"] == group]
        item = {"group": group, "runs": len(rows), "patient_turns": sum(row["patient_turns"] for row in rows),
                "claims": sum(row["claims"] for row in rows),
                "zero_claim_runs": sum(row["claims"] == 0 for row in rows)}
        item.update({label: sum(row[label] for row in rows) for label in LABELS})
        for metric in METRICS:
            n, d = sum(row[metric + "_n"] for row in rows), sum(row[metric + "_d"] for row in rows)
            item[metric + "_n"], item[metric + "_d"] = n, d
            item[metric] = n / d if d else None
        item["coverage_n"] = sum(row["coverage_n"] for row in rows)
        item["coverage_d"] = sum(row["coverage_d"] for row in rows)
        groups.append(item)
    save_csv(output / "source_audited_group_summary.csv", groups)
    (output / "source_audited_comparison.json").write_text(
        json.dumps({"groups": groups, "runs": run_rows, "audit_actions": dict(Counter(
            item["action"] for item in overrides))}, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"runs": len(run_rows), "machine_claims": len(reviewed_rows),
                      "audited_claims": sum(item["claims"] for item in groups),
                      "overrides": len(overrides)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
