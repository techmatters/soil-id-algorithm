# Copyright © 2025 Technology Matters
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published
# by the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program. If not, see https://www.gnu.org/licenses/.

"""
Explain / trace layer for Soil ID scoring.

When ranking is asked to explain itself, the ranking functions write intermediate
values into a `Recorder` (a plain container; a no-op cost when unused). This module
turns a filled `Recorder` into a structured, path-agnostic JSON trace that shows,
per candidate, exactly how each score was computed — location, per-depth-band
horizon Gower detail (including features present on only one side), color, US site
score, rule overrides, and the final combine.

The trace is data only; a separate renderer (scripts/) turns it into an HTML
report. Design so the same schema serves US and global, and could later feed a
product "why this soil" view.

Nothing here re-implements the scoring math: the horizon detail is assembled from
the *actual* `gower_distances(..., return_details=True)` output captured during
ranking, so it can't drift from the algorithm.
"""

from dataclasses import dataclass, field
from typing import Any, Optional

TRACE_VERSION = "1"


@dataclass
class Recorder:
    """
    Collector the ranking code writes into when explaining. All fields default to
    empty; a ranking function only fills what it computes. Keys in the per-candidate
    dicts are the candidate's component id / name so US and global can populate
    them independently.
    """

    region: str = ""  # "US" | "GLOBAL"
    site: dict = field(default_factory=dict)  # lat, lon
    inputs: dict = field(default_factory=dict)  # user horizons + site data
    # per-candidate raw captures, keyed by a stable candidate key (cokey/compname)
    location: dict = field(default_factory=dict)
    horizon: dict = field(default_factory=dict)
    color: dict = field(default_factory=dict)
    site_score: dict = field(default_factory=dict)
    scores: dict = field(default_factory=dict)  # horizon_score, properties_score, combined
    overrides: dict = field(default_factory=dict)
    order: list = field(default_factory=list)  # candidate keys in final rank order

    def enabled(self) -> bool:
        return True


def _status(user: Optional[float], cand: Optional[float]) -> str:
    u = user is not None
    c = cand is not None
    if u and c:
        return "both"
    if u and not c:
        return "user_only"
    if c and not u:
        return "candidate_only"
    return "none"


def _num(x: Any) -> Optional[float]:
    """Coerce to a JSON-friendly float, mapping NaN/None -> None."""
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if f != f else round(f, 4)  # f != f  → NaN


# NOTE: assembly helpers (Recorder -> trace dict) are added incrementally as each
# ranking path is instrumented; see build_trace().


def build_trace(recorder: Recorder) -> dict:
    """Assemble the final JSON trace from a filled Recorder. (WIP: fields added as
    each path is instrumented.)"""
    return {
        "version": TRACE_VERSION,
        "site": recorder.site,
        "region": recorder.region,
        "inputs": recorder.inputs,
        "candidates": [],  # populated by path-specific assembly (added next)
    }
