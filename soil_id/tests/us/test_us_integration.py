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

import pytest

from soil_id.tests.us.test_us import test_params
from soil_id.us_soil import list_soils, rank_soils


@pytest.mark.integration
@pytest.mark.parametrize("location", test_params)
def test_soil_location_integration(location):
    """Runs full algorithm against live APIs. Just verifies no crash."""
    lon, lat = location["lon"], location["lat"]
    list_result = list_soils(lon, lat)
    rank_soils(
        lon,
        lat,
        list_result,
        soilHorizon=["LOAM"] * 7,
        topDepth=[0, 1, 10, 20, 50, 70, 100],
        bottomDepth=[1, 10, 20, 50, 70, 100, 120],
        rfvDepth=["0-1%"] * 7,
        lab_Color=[[41.24, 2.54, 21.17]] * 7,
        pSlope="15",
        pElev=None,
        bedrock=None,
        cracks=False,
    )
