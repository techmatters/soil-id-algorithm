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

from soil_id.global_soil import list_soils_global, rank_soils_global, sg_list

test_locations = [
    {"lon": -1.4631, "lat": 7.3318},
]


def test_soil_location():

    for item in test_locations:
        logging.info(f"Testing {item['lon']}, {item['lat']}")
        start_time = time.perf_counter()
        list_soils_result = list_soils_global(item["lon"], item["lat"])
        logging.info(f"...time: {(time.perf_counter()-start_time):.2f}s")
        rank_soils_result = rank_soils_global(
            item["lon"],
            item["lat"],
            list_output_data=list_soils_result,
            soilHorizon=[],
            horizonDepth=[],
            rfvDepth=[],
            lab_Color=[],
            bedrock=None,
            cracks=None,
        )
        sg_soils_result = sg_list(item["lon"], item["lat"])
        return (list_soils_result, rank_soils_result, sg_soils_result)
