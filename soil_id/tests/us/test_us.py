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

from soil_id.us_soil import list_soils, rank_soils

test_locations = [
    {"lon": -121.5111084, "lat": 45.6508331},
    {"lon": -101.9733687, "lat": 33.81246789},
    {"lon": -121.0347381, "lat": 45.88932423},
    {"lon": -85.50621214, "lat": 39.26009312},
    {"lon": -94.31005777, "lat": 42.63413723},
    {"lon": -99.55016693, "lat": 37.48216451},
    {"lon": -157.2767099, "lat": 62.32776717},
    {"lon": -156.4422738, "lat": 63.52666854},
    {"lon": -119.4596489, "lat": 43.06450312},
    {"lon": -69.28246582, "lat": 47.21392200},
    {"lon": -158.4018264, "lat": 60.42282639},
    {"lon": -121.8166, "lat": 48.6956},
    {"lat": 34.92816, "lon": -114.80764},  # NOTCOM
    {"lat": 35.599180, "lon": -120.491439},  # previous crash: no objects to concatenate
    {"lon": -122.084000, "lat": 37.422000},  # missing LCC
    {"lat": 42.494912, "lon": -123.064531},  # crash: could not broadcast input array
    {"lat": 40.79861, "lon": -112.35477},  # crash: str object has no attribute rank_data_csv
]


def test_soil_location():
    # Dummy Soil Profile Data (replicating the structure provided)
    soilHorizon = ["LOAM"] * 7
    horizonDepth = [1, 10, 20, 50, 70, 100, 120]
    rfvDepth = ["0-1%"] * 7
    lab_Color = [[41.24, 2.54, 21.17]] * 7
    bedrock = None
    pSlope = "15"
    pElev = None
    cracks = False

    for item in test_locations:
        logging.info(f"Testing {item['lon']}, {item['lat']}")
        start_time = time.perf_counter()
        list_soils_result = list_soils(item["lon"], item["lat"])
        logging.info(f"...time: {(time.perf_counter() - start_time):.2f}s")
        rank_soils(
            item["lon"],
            item["lat"],
            list_soils_result,
            soilHorizon,
            horizonDepth,
            rfvDepth,
            lab_Color,
            pSlope,
            pElev,
            bedrock,
            cracks,
        )


def test_empty_rank():
    SoilListOutputData = list_soils(test_locations[0]["lon"], test_locations[0]["lat"])
    rank_soils(
        test_locations[0]["lon"],
        test_locations[0]["lat"],
        SoilListOutputData,
        soilHorizon=[],
        horizonDepth=[],
        rfvDepth=[],
        lab_Color=[],
        pSlope=None,
        pElev=None,
        bedrock=None,
        cracks=None,
    )
