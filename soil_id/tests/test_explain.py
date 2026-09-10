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
Unit tests for the explain/trace layer (soil_id/explain.py) and the HTML
renderer (scripts/render_soil_explain.py).

These are pure/synthetic — they build a Recorder by hand and never touch the
DB or live APIs — so they pin the trace schema and renderer behaviour
independently of the ranking snapshots.
"""

import math

import pytest

from scripts.render_soil_explain import render_html
from soil_id.explain import (
    Recorder,
    _decay_multiplier,
    _num,
    _status,
    build_trace,
)


def _global_recorder():
    """Two candidates: A deep + well matched, B shallow. User recorded 0-2 cm of
    sand + clay only (no rock fragments), so rfv is a 'not entered' column."""
    rec = Recorder()
    rec.region = "GLOBAL"
    rec.site = {"lat": 0.15, "lon": 35.9}
    rec.inputs = {
        "horizons": [{"top": 0, "bottom": 2, "texture": "Clay", "rfv": None, "lab": None}],
        "effective_bedrock_cm": 2.0,
    }
    rec.order = ["A soil", "B soil"]
    rec.location = {
        "A soil": {"distance_m": 0.0, "share_pct": 60.0, "cond_prob": 0.6, "cokey": "1"},
        "B soil": {"distance_m": 100.0, "share_pct": 40.0, "cond_prob": 0.4, "cokey": "2"},
    }
    rec.scores = {
        "A soil": {"horizon_score": 0.80, "properties_score": 0.80, "combined_score": 0.70},
        "B soil": {"horizon_score": 0.30, "properties_score": 0.30, "combined_score": 0.35},
    }
    rec.horizon = {
        "columns": ["sandpct_intpl", "claypct_intpl", "rfv_intpl"],
        "dis_max": 1.0,
        "candidate_full": {
            "A soil": {0: [20.0, 50.0, 10.0], 1: [20.0, 50.0, 10.0]},
            "B soil": {0: [40.0, 20.0, 5.0]},  # shallow: no soil at depth 1
        },
        "slices": [
            {
                "depth": 0,
                "compnames": ["sample_pedon", "A soil", "B soil"],
                "columns": ["sandpct_intpl", "claypct_intpl"],
                "values": [[22.0, 52.0], [20.0, 50.0], [40.0, 20.0]],
                "slice_min": [20.0, 20.0],
                "denom": [82.0, 65.0],
                "dist_from_pedon": [0.0, 0.05, 0.6],
            },
            {
                "depth": 1,
                "compnames": ["sample_pedon", "A soil", "B soil"],
                "columns": ["sandpct_intpl", "claypct_intpl"],
                "values": [[22.0, 52.0], [20.0, 50.0], [40.0, 20.0]],
                "slice_min": [20.0, 20.0],
                "denom": [82.0, 65.0],
                "dist_from_pedon": [0.0, 0.05, 1.0],  # B penalized: no soil here
            },
        ],
    }
    return rec


def _us_recorder():
    """A US-style recorder: color lives in the horizon features (l/a/b) and there
    is a separate site component (slope/elev)."""
    rec = _global_recorder()
    rec.region = "US"
    rec.site_score = {
        "features": ["slope_r", "elev_r"],
        "weights": {"slope_r": 1.0, "elev_r": 0.5},
        "site_wt": 0.5,
        "denom": {"slope_r": 12.0, "elev_r": 200.0},
        "pedon": {"slope_r": 3.0, "elev_r": 950.0},
        "candidates": {
            "A soil": {"values": {"slope_r": 2.0, "elev_r": 900.0}, "score": 0.41},
            "B soil": {"values": {"slope_r": 8.0, "elev_r": 700.0}, "score": 0.12},
        },
    }
    return rec


# --------------------------------------------------------------------------- #
# Pure helpers
# --------------------------------------------------------------------------- #
def test_status_matrix():
    assert _status(1.0, 2.0) == "both"
    assert _status(1.0, None) == "user_only"
    assert _status(None, 2.0) == "candidate_only"
    assert _status(None, None) == "none"


def test_num_coercion():
    assert _num(None) is None
    assert _num(float("nan")) is None
    assert _num("1.23456") == 1.2346
    assert _num(3) == 3.0


def test_decay_multiplier_global_only():
    # global decay is exp(coeff*distance) floored at 0.25
    d = _decay_multiplier(1000.0, "GLOBAL")
    assert d == pytest.approx(max(math.exp(-0.00036888 * 1000.0), 0.25), abs=1e-4)
    assert _decay_multiplier(10_000_000.0, "GLOBAL") == 0.25  # floor
    assert _decay_multiplier(100.0, "US") is None  # US decay not shown here
    assert _decay_multiplier(None, "GLOBAL") is None


# --------------------------------------------------------------------------- #
# build_trace
# --------------------------------------------------------------------------- #
def test_build_trace_ranks_by_combined_score():
    trace = build_trace(_global_recorder())
    assert trace["region"] == "GLOBAL"
    names = [c["name"] for c in trace["candidates"]]
    ranks = [c["rank"] for c in trace["candidates"]]
    assert names == ["A soil", "B soil"]  # higher combined_score first
    assert ranks == [1, 2]


def test_build_trace_none_score_sorts_last():
    rec = _global_recorder()
    rec.scores["A soil"]["combined_score"] = None
    trace = build_trace(rec)
    assert trace["candidates"][-1]["name"] == "A soil"
    assert trace["candidates"][-1]["rank"] == 2


def test_location_component_decay_and_score():
    trace = build_trace(_global_recorder())
    a = trace["candidates"][0]
    loc = next(c for c in a["score_components"] if c["type"] == "location")
    assert loc["distance_m"] == 0.0
    assert loc["decay_multiplier"] == 1.0  # exp(0) = 1
    # location_score = decay * share/100
    assert loc["location_score"] == pytest.approx(0.6, abs=1e-4)
    assert loc["score"] == pytest.approx(0.6, abs=1e-4)  # cond_prob


def test_horizon_segments_penalize_shallow_candidate():
    """B has no soil at depth 1 -> that band must carry the post-infill distance
    (1.0), not be dropped."""
    trace = build_trace(_global_recorder())
    b = next(c for c in trace["candidates"] if c["name"] == "B soil")
    hz = next(c for c in b["score_components"] if c["type"] == "horizon")
    dists = [s["slice_distance"] for s in hz["segments"] if s.get("slice_distance") is not None]
    assert max(dists) == pytest.approx(1.0)


def test_horizon_shows_not_entered_property():
    """rfv was never entered by the user: it must still appear as a feature with a
    candidate value but no user value / no Δ."""
    trace = build_trace(_global_recorder())
    a = next(c for c in trace["candidates"] if c["name"] == "A soil")
    hz = next(c for c in a["score_components"] if c["type"] == "horizon")
    rfv_feats = [f for s in hz["segments"] for f in s["features"] if f["name"] == "rfv_intpl"]
    assert rfv_feats, "rfv column should be present even though not entered"
    # candidate has values, user never does, and there is no normalized diff
    assert any(f["candidate"] is not None for f in rfv_feats)
    assert all(f["user"] is None for f in rfv_feats)
    assert all(f["norm_diff"] is None for f in rfv_feats)


def test_us_site_component_present():
    trace = build_trace(_us_recorder())
    a = next(c for c in trace["candidates"] if c["name"] == "A soil")
    site = next(c for c in a["score_components"] if c["type"] == "site")
    assert site["score"] == pytest.approx(0.41)
    slope = next(f for f in site["features"] if f["name"] == "slope_r")
    assert slope["user"] == 3.0 and slope["candidate"] == 2.0
    assert slope["norm_diff"] == pytest.approx(abs(3.0 - 2.0) / 12.0, abs=1e-4)


# --------------------------------------------------------------------------- #
# Renderer smoke tests
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("rec_fn", [_global_recorder, _us_recorder])
def test_render_html_smoke(rec_fn):
    html = render_html(build_trace(rec_fn()))
    assert isinstance(html, str)
    assert html.startswith("<!doctype")
    # both candidates rendered
    assert "A soil" in html and "B soil" in html
    # core sections present
    assert "Location" in html and "horizon" in html.lower()
    # rfv shown despite not being entered
    assert "not entered" in html
    # the region title made it into the document
    assert rec_fn().region in html


def test_render_html_us_has_site_section():
    assert "Site" in render_html(build_trace(_us_recorder()))
