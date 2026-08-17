"""Fixed-layout multi-panel symptom-network figures."""

from __future__ import annotations

import math

from pathlib import Path

from typing import Any

import networkx as nx

import numpy as np

from matplotlib.lines import Line2D

from ....life_state import ITEM_LABELS_EN

from ....schema import slugify

from ...shared.plotting import plt, save_figure

POSITIVE_COLOR = "#d55e00"

NEGATIVE_COLOR = "#0072b2"


def _fixed_circular_layout(items: list[int]) -> dict[int, np.ndarray]:
    angles = np.linspace(
        math.pi / 2.0, math.pi / 2.0 - 2.0 * math.pi, len(items), endpoint=False
    )
    return {
        item: np.asarray([math.cos(angle), math.sin(angle)])
        for item, angle in zip(items, angles)
    }


def plot_symptom_network_panels(
    panels: list[dict[str, Any]],
    out_dir: Path,
    *,
    edge_threshold: float,
) -> list[Path]:
    """Render only complete figure sets whose every panel passed estimation."""
    paths: list[Path] = []
    for figure_key in dict.fromkeys(panel["figure_key"] for panel in panels):
        selected = [panel for panel in panels if panel["figure_key"] == figure_key]
        if not selected or any(
            panel.get("status") != "estimable" for panel in selected
        ):
            continue
        items = [int(value) for value in selected[0]["items"]]
        layout = _fixed_circular_layout(items)
        max_edge = max(
            (
                abs(float(np.asarray(panel["partial_correlation"])[i, j]))
                for panel in selected
                for i in range(len(items))
                for j in range(i + 1, len(items))
                if abs(float(np.asarray(panel["partial_correlation"])[i, j]))
                >= edge_threshold
            ),
            default=1.0,
        )
        if len(selected) == 4 and len({panel["group"] for panel in selected}) == 2:
            rows, columns = 2, 2
        else:
            rows, columns = 1, len(selected)
        fig, axes = plt.subplots(
            rows,
            columns,
            figsize=(4.25 * columns, 4.45 * rows + 1.0),
            squeeze=False,
            constrained_layout=True,
        )
        for ax, panel in zip(axes.ravel(), selected):
            partial = np.asarray(panel["partial_correlation"], dtype=float)
            graph = nx.Graph()
            graph.add_nodes_from(items)
            positive: list[tuple[int, int, float]] = []
            negative: list[tuple[int, int, float]] = []
            for left_index, left_item in enumerate(items):
                for right_index in range(left_index + 1, len(items)):
                    value = float(partial[left_index, right_index])
                    if abs(value) < edge_threshold:
                        continue
                    target = positive if value > 0 else negative
                    target.append((left_item, items[right_index], value))
            for edges, color in (
                (positive, POSITIVE_COLOR),
                (negative, NEGATIVE_COLOR),
            ):
                if not edges:
                    continue
                nx.draw_networkx_edges(
                    graph,
                    layout,
                    ax=ax,
                    edgelist=[(left, right) for left, right, _ in edges],
                    edge_color=color,
                    width=[0.6 + 4.8 * abs(value) / max_edge for _, _, value in edges],
                    alpha=[
                        0.28 + 0.65 * abs(value) / max_edge for _, _, value in edges
                    ],
                )
            nx.draw_networkx_nodes(
                graph,
                layout,
                ax=ax,
                node_color="white",
                edgecolors="#1f2937",
                linewidths=0.9,
                node_size=720 if len(items) <= 9 else 520,
            )
            nx.draw_networkx_labels(
                graph,
                layout,
                ax=ax,
                labels={item: f"I{item}" for item in items},
                font_size=8 if len(items) <= 9 else 6.5,
            )
            ax.set_title(
                f"{panel['panel_label']}\n$n$ = {panel['n']}",
                fontsize=10.5,
                fontweight="bold",
            )
            ax.set_xlim(-1.28, 1.28)
            ax.set_ylim(-1.28, 1.28)
            ax.set_aspect("equal")
            ax.axis("off")
        for ax in axes.ravel()[len(selected) :]:
            ax.axis("off")
        scale = str(selected[0]["scale"])
        labels = ITEM_LABELS_EN.get(scale, {})
        item_entries = [f"I{item} {labels.get(item, f'Item {item}')}" for item in items]
        split_at = math.ceil(len(item_entries) / 2)
        item_key = (
            "  ·  ".join(item_entries[:split_at])
            + "\n"
            + "  ·  ".join(item_entries[split_at:])
        )
        fig.suptitle(str(selected[0]["figure_title"]), fontsize=13.5, fontweight="bold")
        fig.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color=POSITIVE_COLOR,
                    linewidth=2.4,
                    label="Positive partial association",
                ),
                Line2D(
                    [0],
                    [0],
                    color=NEGATIVE_COLOR,
                    linewidth=2.4,
                    label="Negative partial association",
                ),
            ],
            loc="lower center",
            bbox_to_anchor=(0.5, 0.05),
            ncol=2,
            frameon=False,
        )
        fig.text(
            0.5,
            0.008,
            item_key,
            ha="center",
            va="top",
            fontsize=7.2,
            color="#475569",
            wrap=True,
        )
        path = out_dir / f"symptom_network_{slugify(scale)}_{figure_key}.png"
        paths.append(save_figure(fig, path))
    return paths
