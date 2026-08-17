"""One-shot orchestration for long-format symptom and persona paper figures."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from .analysis.persona_predictor import build_persona_shap_analysis
from .analysis.symptom_effects import build_model_effects, build_timepoint_effects
from .analysis.symptom_network import build_symptom_network_analysis
from .analysis.symptom_composite import build_symptom_composite_analysis
from .analysis.symptom_trajectory import build_symptom_trajectory_summary
from .charts.persona.shap import plot_persona_shap
from .charts.symptoms.effect_forest import plot_symptom_effect_forest
from .charts.symptoms.network import plot_symptom_network_panels
from .charts.symptoms.trajectory_item_change import (
    plot_symptom_trajectory_item_change,
)
from .charts.symptoms.trajectory_small_multiples import (
    plot_symptom_trajectory_small_multiples,
)
from .data.longitudinal import dataset_audit
from .schema import slugify


LONG_FIGURE_TYPES = {
    "persona-shap",
    "symptom-trajectory",
    "symptom-forest",
    "symptom-network",
    "symptom-composite",
}


def _write_frame(
    frame: pd.DataFrame, path: Path, outputs: dict[str, str], key: str
) -> None:
    if frame.empty:
        return
    frame.to_csv(path, index=False)
    outputs[key] = path.name


def _markdown_report(audit: dict[str, Any]) -> str:
    data = audit["data"]
    lines = [
        "# Long-format paper figure feasibility audit",
        "",
        f"- Independent outer runs: {data['independent_runs']}",
        f"- Personas: {', '.join(data['personas']) or 'none'}",
        f"- Groups: {', '.join(f'{key}={value}' for key, value in data['groups'].items()) or 'none'}",
        "",
    ]
    for family, result in audit["figures"].items():
        lines.extend(
            [f"## {family}", "", f"Status: `{result.get('status', 'unknown')}`", ""]
        )
        reasons = result.get("missing_conditions") or (
            [result["reason"]] if result.get("reason") else []
        )
        if reasons:
            lines.append("Missing conditions:")
            lines.append("")
            lines.extend(f"- {reason}" for reason in reasons)
            lines.append("")
        for warning in result.get("warnings") or []:
            lines.append(f"- Warning: {warning}")
        if result.get("warnings"):
            lines.append("")
    lines.extend(
        [
            "## Statistical unit",
            "",
            "Rows are summarized at the independent `run` level. Repeated scale generations from one frozen snapshot are not treated as independent samples.",
            "",
        ]
    )
    return "\n".join(lines)


def render_long_format_paper_figures(
    frame: pd.DataFrame,
    out_dir: Path,
    *,
    figure_types: list[str],
    scales: list[str],
    feature_frame: pd.DataFrame | None = None,
    cbt_group: str | None = None,
    control_group: str | None = None,
    interval: str = "ci95",
    forest_timepoints: list[str] | None = None,
    total_effect_reference: bool = False,
    fdr_alpha: float = 0.05,
    shap_target: str = "cbt_relative_benefit",
    shap_scale: str = "PHQ-9",
    shap_endpoint: str | None = None,
    shap_min_samples: int = 40,
    network_mode: str = "longitudinal",
    network_scale: str = "PHQ-9",
    network_groups: list[str] | None = None,
    network_timepoints: list[str] | None = None,
    network_min_n: int | None = None,
    network_ebic_gamma: float = 0.5,
    network_alpha_min_ratio: float = 0.01,
    network_alpha_count: int = 30,
    network_edge_threshold: float = 0.05,
) -> dict[str, Any]:
    requested = [value for value in figure_types if value in LONG_FIGURE_TYPES]
    out_dir.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, str] = {}
    audit: dict[str, Any] = {"data": dataset_audit(frame), "figures": {}}
    canonical_path = out_dir / "symptom_long_data.csv"
    frame.to_csv(canonical_path, index=False)
    outputs["canonical_long_data"] = canonical_path.name

    if "symptom-composite" in requested:
        if not cbt_group or not control_group:
            audit["figures"]["symptom-composite"] = {
                "status": "unavailable",
                "missing_conditions": [
                    "explicit --cbt-group and --control-group mapping"
                ],
            }
        elif len(scales) != 2:
            audit["figures"]["symptom-composite"] = {
                "status": "unavailable",
                "missing_conditions": ["exactly two --symptom-scale values"],
            }
        else:
            composite_trajectory, composite_changes, composite_audit = (
                build_symptom_composite_analysis(
                    frame,
                    scales=scales,
                    groups=[cbt_group, control_group],
                    common_personas_only=True,
                )
            )
            _write_frame(
                composite_trajectory,
                out_dir / "symptom_composite_trajectory_statistics.csv",
                outputs,
                "symptom_composite_trajectory_statistics",
            )
            _write_frame(
                composite_changes,
                out_dir / "symptom_composite_item_changes.csv",
                outputs,
                "symptom_composite_item_changes",
            )
            path = plot_symptom_trajectory_item_change(
                composite_trajectory,
                composite_changes,
                out_dir,
            )
            if path:
                outputs["symptom_composite_figure"] = path.name
            audit["figures"]["symptom-composite"] = composite_audit

    if "symptom-trajectory" in requested:
        family_audits = []
        for scale in scales:
            summary, scale_audit = build_symptom_trajectory_summary(frame, scale)
            family_audits.append(scale_audit)
            _write_frame(
                summary,
                out_dir / f"symptom_trajectory_statistics_{slugify(scale)}.csv",
                outputs,
                f"trajectory_statistics_{scale}",
            )
            path = plot_symptom_trajectory_small_multiples(
                summary, out_dir, scale=scale, interval=interval
            )
            if path:
                outputs[f"trajectory_figure_{scale}"] = path.name
        statuses = [entry["status"] for entry in family_audits]
        audit["figures"]["symptom-trajectory"] = {
            "status": (
                "unavailable"
                if not statuses or all(value == "unavailable" for value in statuses)
                else (
                    "available_with_warnings"
                    if any(value != "available" for value in statuses)
                    else "available"
                )
            ),
            "scales": family_audits,
            "warnings": [
                warning
                for entry in family_audits
                for warning in entry.get("warnings", [])
            ],
        }

    if "symptom-forest" in requested:
        if not cbt_group or not control_group:
            audit["figures"]["symptom-forest"] = {
                "status": "unavailable",
                "missing_conditions": [
                    "explicit --cbt-group and --control-group mapping"
                ],
            }
        else:
            scale_audits = []
            for scale in scales:
                model_rows, model_audit = build_model_effects(
                    frame,
                    scale=scale,
                    cbt_group=cbt_group,
                    control_group=control_group,
                    fdr_alpha=fdr_alpha,
                )
                d_rows, total_rows, d_audit = build_timepoint_effects(
                    frame,
                    scale=scale,
                    cbt_group=cbt_group,
                    control_group=control_group,
                    timepoints=forest_timepoints,
                    fdr_alpha=fdr_alpha,
                )
                _write_frame(
                    model_rows,
                    out_dir / f"symptom_model_effects_{slugify(scale)}.csv",
                    outputs,
                    f"model_effects_{scale}",
                )
                _write_frame(
                    d_rows,
                    out_dir / f"symptom_cohen_d_{slugify(scale)}.csv",
                    outputs,
                    f"cohen_d_{scale}",
                )
                _write_frame(
                    total_rows,
                    out_dir / f"total_score_cohen_d_{slugify(scale)}.csv",
                    outputs,
                    f"total_effects_{scale}",
                )
                path = plot_symptom_effect_forest(
                    model_rows,
                    d_rows,
                    out_dir,
                    scale=scale,
                    total_reference=total_effect_reference,
                )
                if path:
                    outputs[f"forest_figure_{scale}"] = path.name
                scale_audits.append(
                    {"scale": scale, "panel_a": model_audit, "panel_b": d_audit}
                )
            statuses = [
                panel[part]["status"]
                for panel in scale_audits
                for part in ("panel_a", "panel_b")
            ]
            warnings_list = [
                warning
                for panel in scale_audits
                for part in ("panel_a", "panel_b")
                for warning in panel[part].get("warnings", [])
            ]
            audit["figures"]["symptom-forest"] = {
                "status": (
                    "unavailable"
                    if not statuses or all(value == "unavailable" for value in statuses)
                    else (
                        "available_with_warnings"
                        if any(value != "available" for value in statuses)
                        else "available"
                    )
                ),
                "scales": scale_audits,
                "warnings": warnings_list,
            }

    if "symptom-network" in requested:
        resolved_network_groups = network_groups
        if (
            network_mode == "group-comparison"
            and not resolved_network_groups
            and cbt_group
            and control_group
        ):
            resolved_network_groups = [cbt_group, control_group]
        panels, edge_rows, network_audit = build_symptom_network_analysis(
            frame,
            scale=network_scale,
            mode=network_mode,
            groups=resolved_network_groups,
            timepoints=network_timepoints,
            min_n=network_min_n,
            ebic_gamma=network_ebic_gamma,
            alpha_min_ratio=network_alpha_min_ratio,
            n_alphas=network_alpha_count,
            edge_threshold=network_edge_threshold,
        )
        edge_path = out_dir / "symptom_network_edges.csv"
        edge_rows.to_csv(edge_path, index=False)
        outputs["symptom_network_edges"] = edge_path.name
        diagnostics_path = out_dir / "symptom_network_diagnostics.json"
        diagnostics_path.write_text(
            json.dumps(network_audit, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs["symptom_network_diagnostics"] = diagnostics_path.name
        network_paths = plot_symptom_network_panels(
            panels,
            out_dir,
            edge_threshold=network_edge_threshold,
        )
        outputs.update(
            {
                f"symptom_network_figure_{index}": path.name
                for index, path in enumerate(network_paths, start=1)
            }
        )
        audit["figures"]["symptom-network"] = network_audit

    if "persona-shap" in requested:
        analysis, shap_audit = build_persona_shap_analysis(
            frame,
            feature_frame,
            scale=shap_scale,
            cbt_group=cbt_group,
            control_group=control_group,
            target_column=shap_target,
            endpoint=shap_endpoint,
            min_samples=shap_min_samples,
        )
        audit["figures"]["persona-shap"] = shap_audit
        if analysis:
            stats = analysis["feature_stats"].drop(columns=["column_index"])
            _write_frame(
                stats,
                out_dir / "persona_shap_feature_statistics.csv",
                outputs,
                "persona_shap_statistics",
            )
            outputs["persona_shap_figure"] = plot_persona_shap(analysis, out_dir).name

    audit_path = out_dir / "paper_figure_feasibility.json"
    audit_path.write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    report_path = out_dir / "paper_figure_feasibility.md"
    report_path.write_text(_markdown_report(audit), encoding="utf-8")
    outputs["feasibility_json"] = audit_path.name
    outputs["feasibility_report"] = report_path.name
    manifest = {
        "schema": "experiment_eval_long_item_v1",
        "requested_figures": requested,
        "scales": scales,
        "cbt_group": cbt_group,
        "control_group": control_group,
        "symptom_network": {
            "mode": network_mode,
            "scale": network_scale,
            "groups": network_groups,
            "timepoints": network_timepoints,
            "min_n": network_min_n,
            "ebic_gamma": network_ebic_gamma,
            "alpha_min_ratio": network_alpha_min_ratio,
            "alpha_count": network_alpha_count,
            "edge_threshold": network_edge_threshold,
        },
        "outputs": outputs,
        "audit": audit,
    }
    manifest_path = out_dir / "paper_figure_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {
        "out_dir": str(out_dir),
        "outputs": outputs,
        "audit": audit,
        "manifest": str(manifest_path),
    }
