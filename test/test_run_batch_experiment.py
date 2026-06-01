from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RUNSHELLS_DIR = REPO_ROOT / "runshells"
for path in (REPO_ROOT, RUNSHELLS_DIR):
    path_str = str(path)
    if path_str not in sys.path:
        sys.path.insert(0, path_str)

import run_batch_experiment as rbexp



def _condition_names(selector: str) -> list[str]:
    return [condition.name for condition in rbexp.resolve_conditions(selector)]



def test_resolve_conditions_exact_match() -> None:
    assert _condition_names("Counsel-G1-MILD") == ["Counsel-G1-MILD"]



def test_resolve_conditions_single_group_all_severities() -> None:
    assert _condition_names("Counsel-G1-ALL") == [
        "Counsel-G1-MILD",
        "Counsel-G1-MOD",
        "Counsel-G1-SEV",
    ]



def test_resolve_conditions_all_groups_single_severity() -> None:
    assert _condition_names("Counsel-ALL-MOD") == [
        "Counsel-G1-MOD",
        "Counsel-G2-MOD",
        "Counsel-G3-MOD",
        "Counsel-G5-MOD",
    ]



def test_resolve_conditions_accepts_full_severity_name() -> None:
    assert _condition_names("Counsel-ALL-MODERATE") == [
        "Counsel-G1-MOD",
        "Counsel-G2-MOD",
        "Counsel-G3-MOD",
        "Counsel-G5-MOD",
    ]
