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
End-to-end snapshot of the explain trace produced by the real US ranking path
(against cached API fixtures), plus a smoke check that the HTML renderer
consumes that trace without error. Refresh with --snapshot-update (via
`make regen_snapshots`).
"""

from syrupy.extensions.json import JSONSnapshotExtension

from soil_id.explain import Recorder
from soil_id.render_explain import render_html
from soil_id.us_soil import list_soils, rank_soils

LOCATION = {"lon": -101.9733687, "lat": 33.81246789}


def test_explain_trace_snapshot(snapshot, api_fixtures):
    with api_fixtures(LOCATION["lon"], LOCATION["lat"]):
        list_result = list_soils(LOCATION["lon"], LOCATION["lat"])
        recorder = Recorder()
        rank_result = rank_soils(
            LOCATION["lon"],
            LOCATION["lat"],
            list_result,
            soilHorizon=["Clay loam", "Clay", "Sandy clay loam"],
            topDepth=[0, 30, 60],
            bottomDepth=[30, 60, 100],
            rfvDepth=["0-1%", "1-15%", "1-15%"],
            lab_Color=[[41.24, 2.54, 21.17]] * 3,
            pSlope="3",
            pElev=None,  # api_fixtures mocks get_elev_data -> deterministic elevation
            bedrock=None,
            cracks=False,
            explain=recorder,
        )

    trace = rank_result["explanation"]

    # The trace must cover every ranked candidate and carry location + horizon
    # (and, since slope+elevation are both present, site) components.
    assert trace["region"] == "US"
    assert trace["candidates"], "expected at least one candidate"
    types = {c["type"] for cand in trace["candidates"] for c in cand["score_components"]}
    assert {"location", "horizon", "site"} <= types

    # The renderer must consume the real trace end-to-end without raising.
    html = render_html(trace)
    assert html.startswith("<!doctype")

    assert snapshot.with_defaults(extension_class=JSONSnapshotExtension) == trace
