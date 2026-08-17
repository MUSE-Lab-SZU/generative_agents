"""Shared plotting infrastructure and compatibility aliases."""

from __future__ import annotations

import sys

from ...data import faceted as faceted_data
from .plotting import plt, save_figure

sys.modules[f"{__name__}.faceted_data"] = faceted_data

__all__ = ["plt", "save_figure"]
