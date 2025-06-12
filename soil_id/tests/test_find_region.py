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

from soil_id.utils import find_region_for_location
import pytest

test_locations = [
    (38.968984, -103.625974, 'US'),
    (-6.708697, -69.306646, 'Global'),
    (-3.521766, -136.995712, 'Global')
]

@pytest.mark.parametrize("location", test_locations)
def test_find_region(location):
    lat, lon, region = location    

    assert find_region_for_location(lat=lat, lon=lon) == region
