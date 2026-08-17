"""Historical visualization namespace backed by grouped compatibility facades."""

from __future__ import annotations

import importlib
import sys

_ALIAS_TARGETS = {
    "appendix_plots": "experiment_eval.compat.visualization.core.appendix_plots",
    "core_plots": "experiment_eval.compat.visualization.core.core_plots",
    "outcome_plots": "experiment_eval.compat.visualization.core.outcome_plots",
    "process_plots": "experiment_eval.compat.visualization.core.process_plots",
    "reliability_plots": "experiment_eval.compat.visualization.core.reliability_plots",
    "scale_plots": "experiment_eval.compat.visualization.core.scale_plots",
    "trajectory_plots": "experiment_eval.compat.visualization.core.trajectory_plots",
    "cross_persona_plots": "experiment_eval.compat.visualization.persona.cross_persona_plots",
    "persona_comparison_plots": "experiment_eval.compat.visualization.persona.persona_comparison_plots",
    "persona_heatmap_plots": "experiment_eval.compat.visualization.persona.persona_heatmap_plots",
    "persona_profile_plots": "experiment_eval.compat.visualization.persona.persona_profile_plots",
    "persona_shap_plots": "experiment_eval.compat.visualization.persona.persona_shap_plots",
    "persona_trajectory_plots": "experiment_eval.compat.visualization.persona.persona_trajectory_plots",
    "case_study_plots": "experiment_eval.compat.visualization.interpretive.case_study_plots",
    "interpretive_plots": "experiment_eval.compat.visualization.interpretive.interpretive_plots",
    "item_level_plots": "experiment_eval.compat.visualization.interpretive.item_level_plots",
    "stratified_interpretive_plots": "experiment_eval.compat.visualization.interpretive.stratified_interpretive_plots",
    "weighted_kappa_plots": "experiment_eval.compat.visualization.interpretive.weighted_kappa_plots",
    "change_ci_plots": "experiment_eval.compat.visualization.paper.change_ci_plots",
    "engagement_process_plots": "experiment_eval.compat.visualization.paper.engagement_process_plots",
    "symptom_effect_forest_plots": "experiment_eval.compat.visualization.paper.symptom_effect_forest_plots",
    "symptom_network_plots": "experiment_eval.compat.visualization.paper.symptom_network_plots",
    "symptom_trajectory_plots": "experiment_eval.compat.visualization.paper.symptom_trajectory_plots",
    "common": "experiment_eval.compat.visualization.shared.common",
}

for _alias, _target in _ALIAS_TARGETS.items():
    _module = importlib.import_module(_target)
    sys.modules[f"{__name__}.{_alias}"] = _module
    globals()[_alias] = _module

del _alias, _module, _target
