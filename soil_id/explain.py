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

import math
from dataclasses import dataclass, field
from typing import Any, Optional

TRACE_VERSION = "1"

# Distance-decay coefficient for the global location score (global_soil.py). Used
# only to *display* the decay multiplier in the trace; the score itself (cond_prob)
# comes from the ranking code.
GLOBAL_EXP_COEFF = -0.00036888


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


def _decay_multiplier(distance: Optional[float], region: str) -> Optional[float]:
    """The distance-decay factor shown in the location breakdown (display only)."""
    if distance is None:
        return None
    if region == "GLOBAL":
        return round(max(math.exp(GLOBAL_EXP_COEFF * distance), 0.25), 4)
    return None  # US decay is captured directly by the US path (added later)


def _horizon_segments(slices: list, candidate: str) -> list:
    """
    Consolidate the captured per-cm slices into depth bands for one candidate.

    Consecutive depths with identical (user value, candidate value) across every
    feature collapse into a single band. Each feature carries both sides' values
    (None where absent), the normalized difference, and a status so the report can
    show whether it was compared, skipped, or a no-soil gap.
    """
    bands: list = []
    prev_key = None
    for s in slices:
        compnames = s["compnames"]
        if candidate not in compnames:
            continue
        ci = compnames.index(candidate)
        pi = compnames.index("sample_pedon") if "sample_pedon" in compnames else 0
        cols, vals = s["columns"], s["values"]
        denom = s.get("denom")
        depth = s["depth"]

        features = []
        for k, col in enumerate(cols):
            uv, cv = _num(vals[pi][k]), _num(vals[ci][k])
            rng = None if not denom else round(float(denom[k]), 2)
            norm_diff = None
            if uv is not None and cv is not None and rng:
                norm_diff = round(abs(uv - cv) / rng, 4)
            features.append(
                {
                    "name": col,
                    "user": uv,
                    "candidate": cv,
                    "range": rng,
                    "norm_diff": norm_diff,
                    "status": _status(uv, cv),
                }
            )

        dist = s["dist_from_pedon"][ci]
        band = {
            "top": depth,
            "bottom": depth + 1,
            "depth_weight": 0.2 if depth < 20 else 1.0,
            "features": features,
            "slice_distance": _num(dist),
        }
        key = tuple((f["name"], f["user"], f["candidate"]) for f in features)
        if prev_key == key and bands:
            bands[-1]["bottom"] = depth + 1  # extend the current band
        else:
            bands.append(band)
            prev_key = key
    return bands


def _candidate_trace(name: str, recorder: Recorder) -> dict:
    loc = recorder.location.get(name, {})
    scores = recorder.scores.get(name, {})
    components = []

    # Location (both paths)
    dist = _num(loc.get("distance_m"))
    components.append(
        {
            "type": "location",
            "distance_m": dist,
            "share_pct": _num(loc.get("share_pct")),
            "exp_coeff": GLOBAL_EXP_COEFF if recorder.region == "GLOBAL" else None,
            "decay_multiplier": _decay_multiplier(dist, recorder.region),
            "score": _num(loc.get("cond_prob")),
        }
    )

    # Horizon (both paths)
    components.append(
        {
            "type": "horizon",
            "segments": _horizon_segments(recorder.horizon.get("slices", []), name),
            "score": _num(scores.get("horizon_score")),
        }
    )

    # Color (global only, as a separate track)
    if recorder.color:
        components.append(
            {
                "type": "color",
                "delta_e": recorder.color.get("delta_e"),
                "score": _num(recorder.color.get("similarity", {}).get(name)),
                "weight": recorder.color.get("weight"),
            }
        )

    return {
        "name": name,
        "component_id": loc.get("cokey"),
        "combined_score": _num(scores.get("combined_score")),
        "properties_score": _num(scores.get("properties_score")),
        "score_components": components,
        "overrides": [recorder.overrides[name]] if name in recorder.overrides else [],
    }


def build_trace(recorder: Recorder) -> dict:
    """Assemble the final JSON trace from a filled Recorder."""
    candidates = [_candidate_trace(name, recorder) for name in recorder.order]
    for rank, cand in enumerate(
        sorted(
            candidates, key=lambda c: (c["combined_score"] is None, -(c["combined_score"] or 0))
        ),
        start=1,
    ):
        cand["rank"] = rank
    candidates.sort(key=lambda c: c["rank"])
    return {
        "version": TRACE_VERSION,
        "site": recorder.site,
        "region": recorder.region,
        "inputs": recorder.inputs,
        "candidates": candidates,
    }
