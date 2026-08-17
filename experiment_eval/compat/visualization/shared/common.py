"""Compatibility facade for shared chart infrastructure."""

from ....charts.shared import plotting as _impl

for _name in dir(_impl):
    if not _name.startswith("__"):
        globals()[_name] = getattr(_impl, _name)

del _name
