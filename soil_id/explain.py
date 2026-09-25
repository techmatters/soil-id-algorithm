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

# Distance-decay coefficients per data source (used only to *display* the decay
# multiplier + formula in the trace; the score itself comes from the ranking
# code). The coefficient is steeper for finer-resolution sources — SSURGO map
# units are small so distance discriminates strongly; STATSGO/HWSD2 are coarse.
GLOBAL_EXP_COEFF = -0.00036888  # HWSD2 (global)
US_EXP_COEFF = {"SSURGO": -0.008, "STATSGO": -0.0002772}


def _exp_coeff(region: str, data_source: Optional[str]) -> Optional[float]:
    if region == "GLOBAL":
        return GLOBAL_EXP_COEFF
    if region == "US":
        return US_EXP_COEFF.get(data_source)
    return None


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


def _compared_by_depth(recorder: Recorder, candidate: str) -> dict:
    """From the per-slice captures (the compared window), index user values, the
    per-feature denominator, and the pedon->candidate distance by depth."""
    out = {}
    for s in recorder.horizon.get("slices", []):
        names = s["compnames"]
        if candidate not in names:
            continue
        pi = names.index("sample_pedon") if "sample_pedon" in names else 0
        ci = names.index(candidate)
        cols, vals, denom = s["columns"], s["values"], (s.get("denom") or [])
        out[s["depth"]] = {
            "user": {c: _num(vals[pi][k]) for k, c in enumerate(cols)},
            "denom": {c: (denom[k] if k < len(denom) else None) for k, c in enumerate(cols)},
            "dist": s["dist_from_pedon"][ci],
        }
    return out


def _horizon_segments(recorder: Recorder, candidate: str) -> list:
    """
    Build depth bands for one candidate over the FULL range (0 .. max(pit,
    candidate depth)), consolidating consecutive identical depths. Candidate values
    come from its full profile; the user side is `None` outside the depths you
    recorded; distance/Δ are only present where the algorithm actually compared
    (your recorded window). One-sided data shows as `None` with a status.
    """
    columns = recorder.horizon.get("columns", [])
    cand_full = recorder.horizon.get("candidate_full", {}).get(candidate, {})
    # candidate_full keys may be int (in-process) or str (after a JSON round-trip)
    cand_at = {int(d): v for d, v in cand_full.items()}
    # The pit's full recorded profile (per depth -> {col: value}), so the user
    # column always shows what was entered — even where it wasn't compared.
    pedon_at = {int(d): v for d, v in recorder.horizon.get("pedon_full", {}).items()}
    compared = _compared_by_depth(recorder, candidate)

    cand_depths = [d for d, v in cand_at.items() if any(x is not None for x in v)]
    depths = set(compared) | set(cand_depths) | set(pedon_at)
    if not depths:
        return []
    max_d = max(depths) + 1

    bands: list = []
    prev_key = None
    for d in range(max_d):
        comp = compared.get(d)
        cvals = cand_at.get(d)
        prow = pedon_at.get(d)
        features = []
        for k, col in enumerate(columns):
            # User value from the pit's recorded profile; fall back to the compared
            # slice (older traces without pedon_full).
            uv = prow.get(col) if prow else (comp["user"].get(col) if comp else None)
            cv = _num(cvals[k]) if cvals is not None and k < len(cvals) else None
            rng, nd = None, None
            if comp and comp["denom"].get(col):
                rng = round(float(comp["denom"][col]), 2)
                if uv is not None and cv is not None:
                    nd = round(abs(uv - cv) / rng, 4)
            features.append(
                {
                    "name": col,
                    "user": uv,
                    "candidate": cv,
                    "range": rng,
                    "norm_diff": nd,
                    "status": _status(uv, cv),
                }
            )
        band = {
            "top": d,
            "bottom": d + 1,
            "depth_weight": 0.2 if d < 20 else 1.0,
            "compared": comp is not None,
            "features": features,
            "slice_distance": _num(comp["dist"]) if comp else None,
        }
        # Break bands at the depth-weight boundary (20 cm) too, so the report shows
        # 0-20 @ ×0.2 and 20-.. @ ×1.0 separately even when values are identical.
        key = (comp is not None, band["depth_weight"]) + tuple(
            (f["name"], f["user"], f["candidate"]) for f in features
        )
        if prev_key == key and bands:
            bands[-1]["bottom"] = d + 1
        else:
            bands.append(band)
            prev_key = key
    return bands


def _candidate_trace(name: str, recorder: Recorder) -> dict:
    loc = recorder.location.get(name, {})
    scores = recorder.scores.get(name, {})
    components = []

    # Location (both paths). The real flow (process_distance_scores, Fan et al.):
    # per (component, map-unit) distance_score = share × decay; summed over a
    # component's map units (sum_distance_score); cond_prob = that ÷ the grand
    # total over all components. decay_multiplier is reconstructed for display only.
    dist = _num(loc.get("distance_m"))
    share = _num(loc.get("share_pct"))
    data_source = loc.get("data_source") or ("HWSD2" if recorder.region == "GLOBAL" else None)

    # Reconstruct the decay factor + the distance where it hits the 0.25 floor,
    # using the per-source coefficient (display only).
    coeff = _exp_coeff(recorder.region, data_source)
    decay, floor_m = None, None
    if coeff:
        if dist is not None:
            decay = round(max(math.exp(coeff * dist), 0.25), 4)
        floor_m = round(math.log(0.25) / coeff)  # distance where exp == 0.25

    # Grand total = Σ comp_distance_score over the distinct component groups (the
    # cond_prob denominator).
    totals = {}
    for other in recorder.location.values():
        g = other.get("compname_grp")
        if g is not None and g not in totals:
            totals[g] = _num(other.get("sum_distance_score")) or 0.0
    total = round(sum(totals.values()), 4) if totals else None

    # This series' visible map-unit occurrences (the candidates sharing its group),
    # deduped by map unit. Their distance_scores are what sum toward
    # comp_distance_score — though some occurrences are filtered before ranking, so
    # they may not fully add up (the renderer shows any remainder).
    grp = loc.get("compname_grp")
    occ, seen_mukeys = [], set()
    for other_name, other in recorder.location.items():
        if grp is not None and other.get("compname_grp") == grp:
            mukey = other.get("mukey")
            if mukey in seen_mukeys:
                continue
            seen_mukeys.add(mukey)
            occ.append({"name": other_name, "distance_score": _num(other.get("distance_score"))})

    components.append(
        {
            "type": "location",
            "distance_m": dist,
            "share_pct": share,
            "data_source": data_source,
            "exp_coeff": coeff,
            "decay_multiplier": decay,
            "floor_m": floor_m,  # distance (m) at which decay reaches the 0.25 floor
            # this map-unit instance's decay×share
            "distance_score": _num(loc.get("distance_score")),
            # the component's total across all its map units (cond_prob numerator)
            "comp_distance_score": _num(loc.get("sum_distance_score")),
            "occurrences": occ,
            "total_distance_score": total,
            "score": _num(loc.get("cond_prob")),
        }
    )

    # Horizon (both paths)
    components.append(
        {
            "type": "horizon",
            "segments": _horizon_segments(recorder, name),
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

    # Site (US only, as a separate track): slope / elevation / depth-to-bedrock.
    if recorder.site_score:
        ss = recorder.site_score
        cand = ss.get("candidates", {}).get(name)
        if cand is not None:
            denom = ss.get("denom", {})
            weights = ss.get("weights", {})
            feats = []
            for fn in ss.get("features", []):
                uv = _num(ss.get("pedon", {}).get(fn))
                cv = _num(cand.get("values", {}).get(fn))
                rng = denom.get(fn)
                nd = (
                    round(abs(uv - cv) / rng, 4)
                    if uv is not None and cv is not None and rng
                    else None
                )
                feats.append(
                    {
                        "name": fn,
                        "user": uv,
                        "candidate": cv,
                        "norm_diff": nd,
                        # Gower feature weight in the site distance (slope 1.0,
                        # elev 0.5, depth-to-bedrock 1.5).
                        "weight": _num(weights.get(fn)),
                        "status": _status(uv, cv),
                    }
                )
            components.append(
                {
                    "type": "site",
                    "features": feats,
                    "weight": ss.get("site_wt"),
                    "score": _num(cand.get("score")),
                }
            )

    return {
        "name": name,
        "component_id": loc.get("cokey"),
        "compname_grp": loc.get("compname_grp"),
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

    # The application shows one entry per component group (the best-scoring one).
    # Mark each candidate's app rank; a lower-scoring duplicate of the same series
    # points at the representative that stands in for it.
    reps: dict = {}  # compname_grp -> (representative name, app rank)
    app_n = 0
    for cand in candidates:  # rank order
        grp = cand.get("compname_grp") or cand["name"]
        if grp not in reps:
            app_n += 1
            reps[grp] = (cand["name"], app_n)
            cand["app_rank"], cand["app_repr"] = app_n, None
        else:
            cand["app_rank"] = None
            cand["app_repr"] = {"name": reps[grp][0], "app_rank": reps[grp][1]}

    return {
        "version": TRACE_VERSION,
        "site": recorder.site,
        "region": recorder.region,
        "inputs": recorder.inputs,
        "candidates": candidates,
    }
