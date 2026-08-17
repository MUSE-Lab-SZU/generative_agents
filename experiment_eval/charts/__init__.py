"""Chart implementations grouped by family, with aliases for pre-grouping imports."""

from __future__ import annotations

import importlib
import sys

_LEGACY_CHART_ALIASES = {
    "trajectory_lines": "experiment_eval.charts.outcomes.trajectory_lines",
    "trajectory_ci": "experiment_eval.charts.outcomes.trajectory_ci",
    "endpoint_change_ci": "experiment_eval.charts.outcomes.endpoint_change_ci",
    "change_rebound": "experiment_eval.charts.outcomes.change_rebound",
    "group_time_contrasts": "experiment_eval.charts.outcomes.group_time_contrasts",
    "outer_contrast_forest": "experiment_eval.charts.outcomes.outer_contrast_forest",
    "cross_scale_convergence": "experiment_eval.charts.outcomes.cross_scale_convergence",
    "endpoint_waterfall": "experiment_eval.charts.outcomes.endpoint_waterfall",
    "change_ci": "experiment_eval.charts.outcomes.change_ci",
    "measurement_icc": "experiment_eval.charts.reliability.measurement_icc",
    "measurement_reliability": "experiment_eval.charts.reliability.measurement_heatmap",
    "process_delivery": "experiment_eval.charts.process.delivery",
    "engagement_process": "experiment_eval.charts.process.engagement",
    "appendix_delta_heatmap": "experiment_eval.charts.appendix.delta_heatmap",
    "appendix_delta_trajectory": "experiment_eval.charts.appendix.delta_trajectory",
    "appendix_endpoint_uncertainty": "experiment_eval.charts.appendix.endpoint_uncertainty",
    "appendix_final_delta": "experiment_eval.charts.appendix.final_delta",
    "appendix_trajectory": "experiment_eval.charts.appendix.trajectory",
    "appendix_volatility": "experiment_eval.charts.appendix.volatility",
    "persona_contrast_forest": "experiment_eval.charts.persona.contrast_forest",
    "persona_leave_one_out": "experiment_eval.charts.persona.leave_one_out",
    "persona_outcome_heatmap": "experiment_eval.charts.persona.outcome_heatmap",
    "persona_process_heatmap": "experiment_eval.charts.persona.process_heatmap",
    "persona_scale_trajectories": "experiment_eval.charts.persona.scale_trajectories",
    "persona_shap": "experiment_eval.charts.persona.shap",
    "persona_treatment_response_profile": "experiment_eval.charts.persona.treatment_response_profile",
    "persona_variance_partition": "experiment_eval.charts.persona.variance_partition",
    "complaint_evaluation_changes": "experiment_eval.charts.interpretive.complaint_evaluation_changes",
    "complaint_interval_alignment": "experiment_eval.charts.interpretive.complaint_interval_alignment",
    "complaint_run_alignment": "experiment_eval.charts.interpretive.complaint_run_alignment",
    "group_symptom_heatmap": "experiment_eval.charts.interpretive.group_symptom_heatmap",
    "item_life_state_change": "experiment_eval.charts.interpretive.item_life_state_change",
    "life_state_symptom_trajectory": "experiment_eval.charts.interpretive.life_state_symptom_trajectory",
    "stratified_item_heatmap": "experiment_eval.charts.interpretive.stratified_item_heatmap",
    "stratified_replicate_agreement": "experiment_eval.charts.interpretive.stratified_replicate_agreement",
    "stratified_run_symptom_heatmap": "experiment_eval.charts.interpretive.stratified_run_symptom_heatmap",
    "symptom_effect_forest": "experiment_eval.charts.symptoms.effect_forest",
    "symptom_network": "experiment_eval.charts.symptoms.network",
    "symptom_trajectory_small_multiples": "experiment_eval.charts.symptoms.trajectory_small_multiples",
    "symptom_trajectory_item_change": "experiment_eval.charts.symptoms.trajectory_item_change",
    "entity_kappa_heatmap": "experiment_eval.charts.kappa.entity_heatmap",
    "weighted_kappa_item_forest": "experiment_eval.charts.kappa.item_forest",
    "weighted_kappa_repeat_pair": "experiment_eval.charts.kappa.repeat_pair_heatmap",
    "weighted_kappa_stratum_heatmap": "experiment_eval.charts.kappa.stratum_heatmap",
    "faceted_boxplot_figure2": "experiment_eval.charts.faceted.figure2",
    "faceted_boxplot_figure4": "experiment_eval.charts.faceted.figure4",
}

for _alias, _target in _LEGACY_CHART_ALIASES.items():
    _module = importlib.import_module(_target)
    sys.modules[f"{__name__}.{_alias}"] = _module
    globals()[_alias] = _module

del _alias, _module, _target
