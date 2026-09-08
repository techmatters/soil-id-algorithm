# Copyright © 2024 Technology Matters
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

import logging
import time

import pytest
from syrupy.extensions.json import JSONSnapshotExtension

from soil_id.db import get_datastore_connection
from soil_id.global_soil import list_soils_global, rank_soils_global

test_locations = [
    {"lon": -1.4631, "lat": 7.3318},
    {"lat": -10.950086, "lon": 17.573093},
    {"lat": 34.5, "lon": 69.16667},
    {"lat": -10.07856, "lon": 15.107436},
    {
        "lat": -19.13333,
        "lon": 145.5125,
        "data": {
            "bottomDepth": [10, 24],
            "lab_Color": [
                [81.36696398, 2.011595682, 13.47178439],
                [81.36696398, 2.011595682, 13.47178439],
            ],
            "rfvDepth": [None, None],
            "soilHorizon": ["Sandy loam", "Loam"],
            "topDepth": [0, 10],
        },
    },
    {
        "lat": 48.71667,
        "lon": 126.13333,
        "data": {
            "bottomDepth": [20],
            "lab_Color": [[51.58348661, 4.985592123, 11.10506759]],
            "rfvDepth": [None],
            "soilHorizon": ["Loamy sand"],
            "topDepth": [0],
        },
    },
    {
        "lat": 37.33333,
        "lon": -5.4,
        "data": {
            "bottomDepth": [25, 160],
            "lab_Color": [
                [41.22423435, 1.447890286, 6.167240052],
                [51.59649652, 4.791128549, 18.92743224],
            ],
            "rfvDepth": [None, None],
            "soilHorizon": ["Clay", "Clay"],
            "topDepth": [0, 125],
        },
    },
    {
        "lat": -1.75,
        "lon": 13.6,
        "data": {
            "bottomDepth": [10, 30],
            "lab_Color": [
                [30.77416274, 5.568356326, 18.03952892],
                [41.23714543, 7.282579218, 25.96353458],
            ],
            "rfvDepth": [None, None],
            "soilHorizon": ["Clay", "Clay"],
            "topDepth": [0, 10],
        },
    },
    {
        "lat": 8.48333,
        "lon": 76.95,
        "data": {
            "bottomDepth": [9, 25, 52],
            "lab_Color": [
                [61.6838179, 11.454856, 19.93103357],
                [61.68224615, 17.08123986, 30.77963923],
                [51.60588072, 9.821763719, 38.77054648],
            ],
            "rfvDepth": [42.0, 41.0, 56.0],
            "soilHorizon": ["Clay", "Clay", "Clay"],
            "topDepth": [0, 9, 25],
        },
    },
    {
        "lat": 30.38333,
        "lon": 35.53333,
        "data": {
            "bottomDepth": [5],
            "lab_Color": [[71.62679033, 5.183026621, 25.83506164]],
            "rfvDepth": [50.0],
            "soilHorizon": ["Loam"],
            "topDepth": [0],
        },
    },
    {
        "lat": -2.06972,
        "lon": 37.29,
        "data": {
            "bottomDepth": [80, 125, 140],
            "lab_Color": [
                [30.73434089, 19.87611528, 21.31856213],
                [30.73434089, 19.87611528, 21.31856213],
                [30.73434089, 19.87611528, 21.31856213],
            ],
            "rfvDepth": [None, None, None],
            "soilHorizon": ["Clay", "Clay", "Silty clay loam"],
            "topDepth": [37, 80, 125],
        },
    },
    {
        "lat": 32.11667,
        "lon": 20.08333,
        "data": {
            "bottomDepth": [20, 47, 120],
            "lab_Color": [
                [81.35859545, 3.799237833, 11.54430451],
                [81.35859545, 3.799237833, 11.54430451],
                [81.35859545, 3.799237833, 11.54430451],
            ],
            "rfvDepth": [None, None, None],
            "soilHorizon": ["Sand", "Sand", "Sand"],
            "topDepth": [0, 20, 47],
        },
    },
    {
        "lat": -24.53333,
        "lon": 33.36667,
        "data": {
            "bottomDepth": [5, 10],
            "lab_Color": [
                [51.59117331, 3.180150056, 12.67936276],
                [71.60636516, 1.125498577, 6.932398776],
            ],
            "rfvDepth": [None, None],
            "soilHorizon": ["Sandy loam", "Loamy sand"],
            "topDepth": [0, 5],
        },
    },
    {
        "lat": 15.73333,
        "lon": 120.31667,
        "data": {
            "bottomDepth": [10, 23, 38],
            "lab_Color": [
                [41.22423435, 1.447890286, 6.167240052],
                [41.22423435, 1.447890286, 6.167240052],
                [51.5981893, 1.26834264, 13.74773572],
            ],
            "rfvDepth": [None, None, None],
            "soilHorizon": ["Silty clay loam", "Silty clay loam", "Silt loam"],
            "topDepth": [0, 10, 23],
        },
    },
]

test_params = []
for idx, coords in enumerate(test_locations):
    test_params.append(pytest.param(coords, id=f"{coords['lat']},{coords['lon']}"))


@pytest.mark.parametrize("location", test_params)
def test_soil_location(location, snapshot):
    if "data" in location:
        data = location["data"]
    else:
        data = {
            "soilHorizon": ["Loam"],
            "topDepth": [0],
            "bottomDepth": [15],
            "rfvDepth": [20],
            "lab_Color": [[41.23035939, 3.623018224, 13.27654356]],
        }

    with get_datastore_connection() as connection:
        logging.info(f"Testing {location['lon']}, {location['lat']}")
        start_time = time.perf_counter()
        list_soils_result = list_soils_global(connection, location["lon"], location["lat"])
        logging.info(f"...time: {(time.perf_counter() - start_time):.2f}s")
        rank_result = rank_soils_global(
            connection,
            location["lon"],
            location["lat"],
            list_output_data=list_soils_result,
            **data,
            bedrock=None,
            cracks=None,
        )

        assert snapshot.with_defaults(extension_class=JSONSnapshotExtension) == {
            "list": list_soils_result.soil_list_json,
            "rank": rank_result,
        }


def test_shallow_soil_profile_not_padded():
    """
    Regression test for #378: each component's displayed profile depth must match
    its OWN soil depth (capped at the 120 cm display limit), not another
    component's. Before the cokey-alignment fix, the per-component profile lists
    were paired with components positionally, so a shallow Lithic Leptosol (soil
    to ~20 cm) was displayed with a deeper soil's full 0-120 cm profile.

    Invariant asserted for every returned soil: max(bottom_depth) == min(soilDepth, 120).
    """
    # Kenya point from #378: a nearby Lithic Leptosol ends at 20 cm and several
    # deep (200 cm) components are also returned, so the two orderings differ —
    # exactly the condition that triggered the misalignment.
    with get_datastore_connection() as connection:
        result = list_soils_global(connection, lon=35.87959, lat=0.14594)

    soils = result.soil_list_json["soilList"]
    assert soils, "expected at least one soil at this location"

    saw_shallow = False
    for soil in soils:
        soil_depth = soil["site"]["siteData"]["soilDepth"]
        depths = [v for v in soil["bottom_depth"].values() if isinstance(v, (int, float))]
        assert depths, f"no numeric bottom_depth for {soil['id']['component']}"
        displayed_max = max(depths)
        expected = min(soil_depth, 120)
        assert displayed_max == expected, (
            f"{soil['id']['component']}: displayed profile to {displayed_max} cm but "
            f"soilDepth={soil_depth} (expected max {expected}) — profile mis-aligned (#378)"
        )
        if soil_depth < 120:
            saw_shallow = True

    assert saw_shallow, "expected at least one shallow (<120 cm) soil to guard the regression"
