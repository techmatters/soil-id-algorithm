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

from soil_id.db import get_datastore_connection
from soil_id.global_soil import list_soils_global, rank_soils_global

test_locations = [
    {"lon": -1.4631, "lat": 7.3318},
    {"lat": -10.950086, "lon": 17.573093},
    {"lat": 34.5, "lon": 69.16667},
    {"lat": -10.07856, "lon": 15.107436},
]

@pytest.mark.parametrize("location", test_locations)
def test_soil_location(location):
    with get_datastore_connection() as connection:
        logging.info(f"Testing {location['lon']}, {location['lat']}")
        start_time = time.perf_counter()
        list_soils_result = list_soils_global(connection, location["lon"], location["lat"])
        logging.info(f"...time: {(time.perf_counter()-start_time):.2f}s")
        rank_soils_global(
            connection,
            location["lon"],
            location["lat"],
            list_output_data=list_soils_result,
            soilHorizon=["Loam"],
            topDepth=[0],
            bottomDepth=[15],
            rfvDepth=[20],
            lab_Color=[[41.23035939, 3.623018224, 13.27654356]],
            bedrock=None,
            cracks=None,
        )
