"""Argument configuration for the main experiment-evaluation command."""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORTS_DIR = PROJECT_ROOT / "results" / "experiment_data" / "reports"
OUT_DIR = PROJECT_ROOT / "docs" / "experiment_evaluation"
RECURSIVE = False
TITLE_PREFIX = "In-silico experiment evaluation"
BATCH_MODE = "compare-all"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot KBD repeat summary charts from archived report JSON files.")
    parser.add_argument("--reports-dir", type=Path, default=None)
    parser.add_argument("--report-file", type=Path, action="append", default=None)
    parser.add_argument("--group-alias", action="append", default=None, metavar="PATH_MATCH=LABEL")
    parser.add_argument(
        "--repeat-alias",
        action="append",
        default=None,
        metavar="PATH_MATCH=REPEAT_ID",
        help="Remap an outer repeat by report-path substring; repeatable.",
    )
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument(
        "--experiment-data-root",
        type=Path,
        default=None,
        help="Optional raw experiment_data root used for conversation-dose/process metrics.",
    )
    parser.add_argument(
        "--checkpoints-root",
        type=Path,
        default=None,
        help="Optional checkpoint root used for CBT stage and prompt-completion metrics.",
    )
    parser.add_argument("--dataset-label", default=None, help="Dataset/protocol label written into every report.")
    parser.add_argument("--recursive", dest="recursive", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--title-prefix", default=None)
    parser.add_argument(
        "--batch-mode",
        choices=["single", "by-group", "by-repeat", "compare-all", "all"],
        default=None,
    )
    parser.add_argument("--kbd", action="append", default=None)
    parser.add_argument("--group", action="append", default=None)
    parser.add_argument("--repeat-id", action="append", default=None)
    parser.add_argument(
        "--combine-kbds",
        action="store_true",
        help="Combine selected KBD variants in the same chart batches instead of writing one subtree per KBD.",
    )
    parser.add_argument(
        "--report-mode",
        choices=["core", "lines-only", "presentation", "appendix", "all"],
        default="core",
        help=(
            "Generate a compact core set by default, or explicitly request "
            "lines-only, presentation, appendix, or the legacy all-chart expansion."
        ),
    )
    parser.add_argument(
        "--error-bar",
        choices=["ci95", "sd", "none"],
        default="ci95",
        help="Uncertainty shown on presentation trajectories (default: ci95).",
    )
    parser.add_argument(
        "--y-axis",
        choices=["full", "adaptive"],
        default="full",
        help="Presentation trajectory y-axis; full uses PHQ-9 0–27 and BDI-II 0–63.",
    )
    parser.add_argument("--show-repeat-points", action="store_true", help="Overlay raw repeat total scores on trajectories.")
    parser.add_argument(
        "--outer-summary",
        choices=["none", "mean-ci"],
        default="none",
        help="Optionally overlay outer-experiment means and t CIs; independent runs remain visible.",
    )
    parser.add_argument(
        "--core-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate paper-oriented contrast, convergence, ICC, waterfall, and process figures.",
    )
    parser.add_argument(
        "--paper-figures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable requested paper-figure families.",
    )
    parser.add_argument(
        "--paper-figure",
        action="append",
        choices=[
            "change-ci",
            "engagement",
            "persona-shap",
            "symptom-trajectory",
            "symptom-forest",
            "symptom-network",
            "symptom-composite",
            "persona-profile",
        ],
        default=None,
        help="Restrict paper figures; repeat to select multiple (default remains change-ci + engagement).",
    )
    parser.add_argument(
        "--paper-outcome",
        action="append",
        default=None,
        help="Outcome for change+CI figures; repeat for multiple outcomes (default: PHQ-9).",
    )
    parser.add_argument(
        "--change-panel-b",
        choices=["complete-case", "qc-passed", "replication"],
        default="complete-case",
        help="Sensitivity/replication subset used in panel B.",
    )
    parser.add_argument(
        "--change-estimator",
        choices=["mean", "ancova"],
        default="mean",
        help="Outer-run mean change or baseline-adjusted ANCOVA marginal change.",
    )
    parser.add_argument(
        "--change-significance-label",
        choices=["stars", "p-value", "both"],
        default="stars",
        help="Label displayed on significant comparison brackets.",
    )
    parser.add_argument(
        "--engagement-bin-hours",
        type=int,
        default=2,
        help="Fixed 24-hour bin width; must divide 24 (default: 2).",
    )
    parser.add_argument(
        "--engagement-secondary",
        choices=["turns", "duration-proxy"],
        default="turns",
        help="Panel-A complementary metric; duration-proxy is not true elapsed duration.",
    )
    parser.add_argument(
        "--long-data",
        type=Path,
        default=None,
        help="Canonical item-level CSV/JSON/JSONL: run, persona, group, timepoint, scale, item, item_score.",
    )
    parser.add_argument(
        "--persona-profile-data",
        type=Path,
        default=None,
        help=(
            "Optional run-level CSV/JSON/JSONL for persona profiles: "
            "run_id, persona_id, group, timepoint, scale, score."
        ),
    )
    parser.add_argument(
        "--profile-zscore",
        choices=["relative-profile", "common-reference", "standardized-change"],
        default="relative-profile",
        help="Declared standardization used by persona-profile (default: relative-profile).",
    )
    parser.add_argument(
        "--profile-scale",
        action="append",
        default=None,
        help="Scale included in persona-profile; repeat for stacked panels (default: all).",
    )
    parser.add_argument(
        "--persona-order",
        nargs="+",
        default=None,
        help="Fixed persona display order; also restricts the profile to these personas.",
    )
    parser.add_argument(
        "--profile-group-order",
        nargs="+",
        default=None,
        help="Condition display order; also restricts the profile to these groups.",
    )
    parser.add_argument(
        "--profile-panel-by",
        choices=["scale", "setting"],
        default="scale",
        help="Stack panels by scale or by optional setting/study column (default: scale).",
    )
    parser.add_argument(
        "--profile-setting-order",
        nargs="+",
        default=None,
        help="Fixed setting panel order when --profile-panel-by setting is used.",
    )
    parser.add_argument("--profile-baseline", default=None, help="Explicit baseline timepoint label.")
    parser.add_argument("--profile-post", default=None, help="Explicit post timepoint label.")
    parser.add_argument(
        "--profile-uncertainty",
        choices=["ci95-band", "ci95-bars", "se-bars", "none"],
        default="none",
        help="Uncertainty display for persona-profile cell means (default: none).",
    )
    parser.add_argument(
        "--feature-data",
        type=Path,
        default=None,
        help="Optional one-row-per-run wide baseline/persona/behavior feature table for persona-shap.",
    )
    parser.add_argument("--cbt-group", default=None, help="Exact group label treated as CBT in long-format analyses.")
    parser.add_argument("--control-group", default=None, help="Exact group label treated as control in long-format analyses.")
    parser.add_argument(
        "--symptom-scale",
        action="append",
        choices=["PHQ-9", "BDI-II"],
        default=None,
        help="Scale for symptom trajectory/forest; repeat for both (default: PHQ-9).",
    )
    parser.add_argument(
        "--symptom-interval",
        choices=["ci95", "se"],
        default="ci95",
        help="Ribbon for symptom small multiples (default: ci95).",
    )
    parser.add_argument(
        "--symptom-network-mode",
        choices=["longitudinal", "group-comparison"],
        default="longitudinal",
        help="One group across time or exactly two groups at baseline/final.",
    )
    parser.add_argument("--symptom-network-scale", choices=["PHQ-9", "BDI-II"], default="PHQ-9")
    parser.add_argument(
        "--symptom-network-group",
        action="append",
        default=None,
        help="Group to include; repeat twice for group-comparison. Longitudinal mode defaults to every group.",
    )
    parser.add_argument(
        "--symptom-network-timepoint",
        action="append",
        default=None,
        help="Timepoint to include in display order; defaults to baseline/middle/final.",
    )
    parser.add_argument(
        "--symptom-network-min-n",
        type=int,
        default=None,
        help="Independent-run rendering gate per panel; default max(30, 5 × node count).",
    )
    parser.add_argument("--symptom-network-ebic-gamma", type=float, default=0.5)
    parser.add_argument("--symptom-network-alpha-min-ratio", type=float, default=0.01)
    parser.add_argument("--symptom-network-alpha-count", type=int, default=30)
    parser.add_argument(
        "--symptom-network-edge-threshold",
        type=float,
        default=0.05,
        help="Minimum absolute regularized partial correlation drawn; all estimated pairs remain in CSV.",
    )
    parser.add_argument(
        "--forest-timepoint",
        action="append",
        default=None,
        help="Assessment shown in forest panel B; repeat for multiple (default: last three non-baseline points).",
    )
    parser.add_argument(
        "--total-effect-reference",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Draw each timepoint's total-score Cohen's d as a red reference line in forest panel B.",
    )
    parser.add_argument("--fdr-alpha", type=float, default=0.05, help="Benjamini-Hochberg threshold (default: 0.05).")
    parser.add_argument(
        "--shap-target",
        default="cbt_relative_benefit",
        help="Feature-table target; derived from matched controls when absent and arm labels are supplied.",
    )
    parser.add_argument("--shap-scale", choices=["PHQ-9", "BDI-II"], default="PHQ-9")
    parser.add_argument("--shap-endpoint", default=None)
    parser.add_argument(
        "--shap-min-samples",
        type=int,
        default=40,
        help="Minimum usable independent CBT runs before fitting SHAP (also requires at least 5×feature count).",
    )
    return parser.parse_args(argv)
