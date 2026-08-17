"""Render the compact PHQ-9/BDI-II scale-credibility figure suite."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .charts.kappa.item_forest import plot_item_heatmap, plot_item_heatmap_by_entity
from .charts.outcomes.cross_scale_convergence import (
    plot_cross_scale_concurrent_validity,
    plot_cross_scale_convergence,
)
from .charts.reliability.measurement_icc import (
    plot_measurement_error,
    plot_measurement_icc,
    plot_measurement_icc_by_entity,
)
from .cli_config import PROJECT_ROOT
from .loader import find_report_files, load_records
from .reports import write_scale_credibility_csvs
from .statistics import (
    cross_scale_concurrent_validity,
    cross_scale_convergence,
    measurement_error_summary,
    measurement_icc,
)
from .weighted_kappa import build_item_weighted_kappa_by_entity, build_weighted_kappa


def _aliases(values: list[str] | None) -> list[tuple[str, str]]:
    aliases: list[tuple[str, str]] = []
    for value in values or []:
        if "=" not in value:
            raise ValueError(f"Alias must be PATH_MATCH=REPEAT_ID: {value}")
        match, label = (part.strip() for part in value.split("=", 1))
        aliases.append((match, label))
    return aliases


def render_scale_credibility(
    records: list[Any],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    *,
    title_prefix: str,
    dataset_label: str,
) -> dict[str, Any]:
    selected_scales = [scale for scale in ("PHQ-9", "BDI-II") if scale in scales]
    out_dir.mkdir(parents=True, exist_ok=True)
    reliability = {scale: measurement_icc(records, labels, scale) for scale in selected_scales}
    measurement_error = {
        scale: measurement_error_summary(records, labels, scale) for scale in selected_scales
    }
    concurrent = cross_scale_concurrent_validity(records, labels)
    change = cross_scale_convergence(records, labels)
    kappa = build_weighted_kappa(records, labels, selected_scales)
    metrics = {
        "scales": selected_scales,
        "measurement_reliability": reliability,
        "measurement_error": measurement_error,
        "cross_scale_concurrent_validity": concurrent,
        "cross_scale_convergence": change,
    }
    csv_paths = write_scale_credibility_csvs(out_dir, metrics)
    kappa_path = out_dir / "weighted_kappa_item.csv"
    from .reports import _write_rows_csv

    _write_rows_csv(kappa_path, kappa.get("item") or [])
    candidates = {
        "icc_forest": plot_measurement_icc(records, labels, selected_scales, out_dir, title_prefix),
        "item_weighted_kappa_heatmap": plot_item_heatmap(kappa, out_dir, title_prefix),
        "measurement_error": plot_measurement_error(records, labels, selected_scales, out_dir, title_prefix),
        "concurrent_validity": plot_cross_scale_concurrent_validity(records, labels, out_dir, title_prefix),
        "change_agreement": plot_cross_scale_convergence(records, labels, out_dir, title_prefix),
    }
    generated = {key: path.name for key, path in candidates.items() if path is not None}
    skipped = {
        key: "required scale/repeat variation is unavailable"
        for key, path in candidates.items()
        if path is None
    }
    manifest = {
        "dataset_label": dataset_label,
        "n_outer_runs": len(records),
        "n_personas": len({record.kbd for record in records}),
        "groups": sorted({record.group for record in records}),
        "timepoints": labels,
        "scales": selected_scales,
        "statistical_units": {
            "measurement": "same frozen snapshot × scale; K generations are repeat measurements",
            "bootstrap_cluster": "independent outer simulation run",
            "concurrent_validity": "same persona/condition/timepoint aligned across scales",
            "change_agreement": "one paired change per independent outer run, using a common baseline and endpoint",
        },
        "definitions": {
            "icc": next((row.get("icc_definition") for row in reliability.values() if row.get("icc_definition")), None),
            "measurement_error": next((row.get("definition") for row in measurement_error.values() if row.get("definition")), None),
            "weighted_kappa": kappa.get("metadata"),
        },
        "generated_figures": generated,
        "generated_csv": {key: path.name for key, path in csv_paths.items()} | {"weighted_kappa_item": kappa_path.name},
        "skipped": skipped,
        "intentionally_not_generated": {
            "repeat_pair_kappa_heatmap": "item heatmap directly answers the requested item-stability question",
            "item_kappa_forest": "duplicates the requested compact item heatmap",
            "by_repeat_panels": "a single outer run is not a valid condition-comparison unit",
            "legacy_reliability_heatmaps": "repeat SD distribution plus ICC/SEM/MDC95 is the compact canonical set",
        },
    }
    manifest_path = out_dir / "scale_credibility_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "figures": generated, "csv": manifest["generated_csv"], "skipped": skipped}


def render_dimension_credibility(
    records: list[Any],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    *,
    entity_field: str,
    dimension_label: str,
    filename_suffix: str,
    dataset_label: str,
) -> dict[str, Any]:
    """Render entity-stratified ICC, item κ, and cross-scale figures."""
    from .reports import _write_rows_csv

    selected_scales = [scale for scale in ("PHQ-9", "BDI-II") if scale in scales]
    out_dir.mkdir(parents=True, exist_ok=True)
    icc_path, icc_rows = plot_measurement_icc_by_entity(
        records,
        labels,
        selected_scales,
        out_dir,
        dimension_label,
        entity_field=entity_field,
        dimension_label=dimension_label,
        filename_suffix=filename_suffix,
    )
    kappa = build_item_weighted_kappa_by_entity(
        records,
        labels,
        selected_scales,
        entity_field=entity_field,
    )
    kappa_path = plot_item_heatmap_by_entity(
        kappa,
        out_dir,
        dimension_label=dimension_label,
        filename_suffix=filename_suffix,
    )
    scatter_entity = "persona" if entity_field == "kbd" else "group"
    concurrent_path = plot_cross_scale_concurrent_validity(
        records,
        labels,
        out_dir,
        dimension_label,
        entity_field=scatter_entity,
        dimension_label=dimension_label,
        filename_suffix=filename_suffix,
        fit_lines=True,
    )
    change_path = plot_cross_scale_convergence(
        records,
        labels,
        out_dir,
        dimension_label,
        entity_field=scatter_entity,
        dimension_label=dimension_label,
        filename_suffix=filename_suffix,
        fit_lines=True,
    )
    entities = sorted({str(getattr(record, entity_field)) for record in records})
    correlation_rows: list[dict[str, Any]] = []
    for entity in entities:
        selected = [record for record in records if str(getattr(record, entity_field)) == entity]
        concurrent = cross_scale_concurrent_validity(selected, labels)
        change = cross_scale_convergence(selected, labels)
        correlation_rows.extend(
            [
                {
                    "analysis": "concurrent_validity",
                    "entity": entity,
                    "n": concurrent.get("n_aligned_snapshots"),
                    "spearman_rho": concurrent.get("spearman_rho"),
                    "ci95_lower": concurrent.get("ci95_lower"),
                    "ci95_upper": concurrent.get("ci95_upper"),
                    "p_value_note": concurrent.get("p_value_note"),
                    "direction_agreement_rate": None,
                },
                {
                    "analysis": "change_agreement",
                    "entity": entity,
                    "n": change.get("n_outer_runs"),
                    "spearman_rho": change.get("spearman_rho"),
                    "ci95_lower": change.get("ci95_lower"),
                    "ci95_upper": change.get("ci95_upper"),
                    "spearman_p": change.get("spearman_p"),
                    "direction_agreement_rate": change.get("direction_agreement_rate"),
                },
            ]
        )
    csv_paths = {
        "icc": _write_rows_csv(out_dir / f"measurement_icc_{filename_suffix}.csv", icc_rows),
        "kappa": _write_rows_csv(out_dir / f"weighted_kappa_item_{filename_suffix}.csv", kappa.get("item") or []),
        "correlations": _write_rows_csv(out_dir / f"cross_scale_grouped_statistics_{filename_suffix}.csv", correlation_rows),
    }
    figures = {
        key: path.name
        for key, path in {
            "icc_forest": icc_path,
            "item_kappa_heatmap": kappa_path,
            "concurrent_validity": concurrent_path,
            "change_agreement": change_path,
        }.items()
        if path is not None
    }
    manifest = {
        "dataset_label": dataset_label,
        "dimension": dimension_label,
        "entity_field": entity_field,
        "entities": entities,
        "n_outer_runs": len(records),
        "timepoints": labels,
        "scales": selected_scales,
        "figures": figures,
        "csv": {key: path.name for key, path in csv_paths.items()},
        "definitions": {
            "icc": "ICC(A,1)/ICC(2,1), two-way random absolute agreement; outer-run cluster bootstrap CI",
            "weighted_kappa": kappa.get("metadata"),
            "fit_lines": "descriptive ordinary least-squares fit within each displayed entity; not the Spearman estimator",
        },
    }
    manifest_path = out_dir / f"scale_credibility_manifest_{filename_suffix}.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"manifest": manifest_path, "figures": figures, "csv": manifest["csv"]}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reports-dir", type=Path)
    parser.add_argument("--report-file", type=Path, action="append")
    parser.add_argument("--recursive", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--repeat-alias", action="append", metavar="PATH_MATCH=REPEAT_ID")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--title-prefix", default="PHQ-9 / BDI-II scale credibility")
    parser.add_argument("--dataset-label", default="validated repeat summaries")
    parser.add_argument(
        "--dimension",
        choices=["condition", "persona"],
        help="Render entity-stratified figures instead of the legacy overall suite.",
    )
    parser.add_argument("--filename-suffix", help="Required with --dimension; e.g. kbd2_by_condition.")
    args = parser.parse_args(argv)
    if args.report_file:
        paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in args.report_file]
    elif args.reports_dir:
        reports_dir = args.reports_dir if args.reports_dir.is_absolute() else PROJECT_ROOT / args.reports_dir
        paths = find_report_files(reports_dir, args.recursive)
    else:
        parser.error("one of --reports-dir or --report-file is required")
    records, labels, scales = load_records(paths, repeat_aliases=_aliases(args.repeat_alias))
    out_dir = args.out_dir if args.out_dir.is_absolute() else PROJECT_ROOT / args.out_dir
    if args.dimension:
        if not args.filename_suffix:
            parser.error("--filename-suffix is required with --dimension")
        entity_field = "group" if args.dimension == "condition" else "kbd"
        result = render_dimension_credibility(
            records,
            labels,
            scales,
            out_dir,
            entity_field=entity_field,
            dimension_label=args.title_prefix,
            filename_suffix=args.filename_suffix,
            dataset_label=args.dataset_label,
        )
    else:
        result = render_scale_credibility(
            records,
            labels,
            scales,
            out_dir,
            title_prefix=args.title_prefix,
            dataset_label=args.dataset_label,
        )
    print(f"Generated {len(result['figures'])} scale-credibility figures in {out_dir}")
    print(result["manifest"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
