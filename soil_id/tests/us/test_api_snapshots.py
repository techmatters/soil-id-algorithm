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
from syrupy.extensions.json import JSONSnapshotExtension

from soil_id.services import get_elev_data, get_soilweb_data
from soil_id.tests.us.test_us import test_params


@pytest.mark.api_snapshot
@pytest.mark.parametrize("location", test_params)
def test_api_snapshot(location, snapshot):
    """
    Fetch live API responses and compare against stored Syrupy snapshots.
    Run with --snapshot-update to refresh stored responses.
    """
    lon, lat = location["lon"], location["lat"]
    assert snapshot.with_defaults(extension_class=JSONSnapshotExtension) == {
        "soilweb": get_soilweb_data(lon, lat),
        "elev": get_elev_data(lon, lat),
    }
