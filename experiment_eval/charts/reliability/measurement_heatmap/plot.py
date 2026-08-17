"""Frozen-snapshot repeat reliability figures."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import numpy as np

from ....schema import ExperimentRecord, label_display, slugify

from ....statistics import measurement_icc, stat_at

from ...shared.plotting import (
    heatmap_text_color as _text_color,
    plt,
    save_figure,
    wrapped as _wrapped,
)

MEASUREMENT_METRICS = {
    "sample_sd": "Sample SD",
    "ci95_width": "95% CI width",
    "category_pairwise_flip_rate": "Category pairwise flip rate",
}


def plot_measurement_reliability_heatmaps(
    records: list[ExperimentRecord],
    labels: list[str],
    scales: list[str],
    out_dir: Path,
    title_prefix: str,
) -> list[Path]:
    paths: list[Path] = []
    for scale in scales:
        for metric, metric_label in MEASUREMENT_METRICS.items():
            matrix = np.asarray(
                [
                    [
                        (
                            stat_at(record, scale, label, metric)
                            if stat_at(record, scale, label, metric) is not None
                            else np.nan
                        )
                        for label in labels
                    ]
                    for record in records
                ],
                dtype=float,
            )
            fig, ax = plt.subplots(
                figsize=(
                    max(7.0, 1.05 * len(labels) + 3.0),
                    max(3.5, 0.55 * len(records) + 2.0),
                ),
                constrained_layout=True,
            )
            cmap_name = (
                "viridis" if metric != "category_pairwise_flip_rate" else "magma"
            )
            finite = matrix[np.isfinite(matrix)]
            vmin = (
                0.0
                if metric == "category_pairwise_flip_rate"
                else (float(finite.min()) if finite.size else 0.0)
            )
            vmax = (
                1.0
                if metric == "category_pairwise_flip_rate"
                else (float(finite.max()) if finite.size else 1.0)
            )
            if math.isclose(vmin, vmax):
                vmax = vmin + 1.0
            im = ax.imshow(matrix, aspect="auto", cmap=cmap_name, vmin=vmin, vmax=vmax)
            ax.set_xticks(
                range(len(labels)), [label_display(label) for label in labels]
            )
            ax.set_yticks(
                range(len(records)), [record.plot_label for record in records]
            )
            ax.set_title(
                _wrapped(
                    f"{title_prefix}: {scale} measurement reliability — {metric_label}"
                ),
                fontsize=12,
            )
            for y, record in enumerate(records):
                for x, label in enumerate(labels):
                    value = matrix[y, x]
                    if not np.isfinite(value):
                        continue
                    modal = stat_at(record, scale, label, "modal_confidence")
                    text = f"{value:.2f}" + (
                        f"\nm={modal:.2f}" if modal is not None else ""
                    )
                    ax.text(
                        x,
                        y,
                        text,
                        ha="center",
                        va="center",
                        fontsize=7,
                        color=_text_color(im.cmap, im.norm, value),
                    )
            fig.colorbar(im, ax=ax, label=metric_label)
            ns = sorted({record.expected_repeats for record in records})
            fig.text(
                0.5,
                -0.015,
                f"n={','.join(map(str, ns))} repeats/cell; error=N/A; endpoint rule for derived metrics: "
                "POST > NOW > last available; not inferred follow-up. m=modal confidence; rows=independent outer experiments.",
                ha="center",
                fontsize=8,
            )
            path = (
                out_dir
                / f"presentation_04_{slugify(scale)}_{metric}_measurement_reliability_heatmap.png"
            )
            paths.append(save_figure(fig, path))
    return paths
