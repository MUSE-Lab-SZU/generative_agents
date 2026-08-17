"""One-shot orchestration for the persona treatment-response profile family."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis.persona_profile import build_persona_profile_analysis
from .charts.persona.treatment_response_profile import (
    plot_persona_treatment_response_profile,
)


PROFILE_FIGURE_TYPES = {"persona-profile"}


def _write_markdown(audit: dict[str, Any]) -> str:
    lines = [
        "# Persona treatment-response profile feasibility",
        "",
        f"- Status: `{audit['status']}`",
        f"- Z-score method: `{audit['zscore_method']}`",
        f"- Independent runs with complete change: {audit['independent_run_count']}",
        f"- Personas: {', '.join(audit['persona_order']) or 'none'}",
        f"- Conditions: {', '.join(audit['group_order']) or 'none'}",
        "",
    ]
    for panel, detail in audit["panels"].items():
        lines.extend(
            [
                f"## Panel {panel}",
                "",
                f"- Scale: {', '.join(detail['scale'])}",
                f"- Baseline / post: `{detail['baseline_timepoint']}` / `{detail['post_timepoint']}`",
                f"- Post selection: `{detail['post_selection']}`",
                f"- Complete changes: {detail['complete_change_rows']}",
                f"- Missing baseline/post pairs: {detail['missing_pair_count']}",
                "",
            ]
        )
    if audit["warnings"]:
        lines.extend(
            ["## Warnings", "", *[f"- {warning}" for warning in audit["warnings"]], ""]
        )
    lines.extend(
        [
            "## Statistical unit",
            "",
            "Baseline and post are paired within an independent `run_id` before persona × condition aggregation. "
            "Frozen-snapshot measurement repeats are not independent runs.",
            "",
        ]
    )
    return "\n".join(lines)


def render_persona_profile(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    method: str = "relative-profile",
    scales: list[str] | None = None,
    persona_order: list[str] | None = None,
    group_order: list[str] | None = None,
    panel_by: str = "scale",
    setting_order: list[str] | None = None,
    baseline_timepoint: str | None = None,
    post_timepoint: str | None = None,
    uncertainty: str = "none",
    title: str = "Persona treatment-response profile",
) -> dict[str, Any]:
    destination = out_dir / "persona_profile"
    destination.mkdir(parents=True, exist_ok=True)
    changes, summary, references, audit = build_persona_profile_analysis(
        frame,
        method=method,
        scales=scales,
        persona_order=persona_order,
        group_order=group_order,
        panel_by=panel_by,
        setting_order=setting_order,
        baseline_timepoint=baseline_timepoint,
        post_timepoint=post_timepoint,
    )
    outputs: dict[str, str] = {}
    canonical_path = destination / "persona_profile_long_data.csv"
    frame.to_csv(canonical_path, index=False)
    outputs["canonical_long_data"] = canonical_path.name
    for key, table, filename in (
        ("run_changes", changes, "persona_profile_run_changes.csv"),
        ("cell_summary", summary, "persona_profile_summary.csv"),
        ("zscore_references", references, "persona_profile_zscore_references.csv"),
    ):
        path = destination / filename
        table.to_csv(path, index=False)
        outputs[key] = path.name
    figure = None
    if audit["status"] != "unavailable":
        figure = plot_persona_treatment_response_profile(
            summary,
            destination,
            method=method,
            panel_order=audit["panel_order"],
            persona_order=audit["persona_order"],
            group_order=audit["group_order"],
            uncertainty=uncertainty,
            title=title,
        )
    if figure:
        outputs["figure"] = figure.name
    audit_path = destination / "persona_profile_feasibility.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    outputs["feasibility_json"] = audit_path.name
    report_path = destination / "persona_profile_feasibility.md"
    report_path.write_text(_write_markdown(audit), encoding="utf-8")
    outputs["feasibility_report"] = report_path.name
    manifest = {
        "schema": "experiment_eval_persona_profile_v1",
        "zscore_method": method,
        "uncertainty": uncertainty,
        "panel_by": panel_by,
        "outputs": outputs,
        "audit": audit,
    }
    manifest_path = destination / "persona_profile_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "out_dir": str(destination),
        "manifest": str(manifest_path),
        "outputs": outputs,
        "audit": audit,
    }
