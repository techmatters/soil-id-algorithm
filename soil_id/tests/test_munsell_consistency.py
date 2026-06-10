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

"""Cross-platform consistency test for Munsell -> CIELAB conversion.

The fixture (fixtures/munsellTestData.json) is generated from the
terraso-mobile-client using the npm ``munsell`` library (regenerate there with
``npm run generate-munsell-test-data``). We check that this library's CSV-lookup
conversion stays close to the mobile client's values so the two platforms agree
on soil colors.

The two implementations use different source data and methods:
- this library: direct lookup in soil_id/data/LandPKS_munsell_rgb_lab.csv
- mobile: the npm ``munsell`` library, with interpolation and Bradford
  chromatic adaptation

A deliberately generous tolerance is used because the source data and methods
differ; the goal is to catch gross errors (wrong hue decoding, swapped
channels), not to enforce exact agreement between two different datasets.
"""

import json
import os

import pytest

from soil_id.color import munsell_to_lab
from soil_id.config import MUNSELL_COLOR_REF, MUNSELL_REF

FIXTURE_PATH = os.path.join(os.path.dirname(__file__), "fixtures", "munsellTestData.json")

# Maximum allowed per-channel LAB difference between this library and the client.
LAB_TOLERANCE = 6.0


def _load_entries():
    with open(FIXTURE_PATH) as f:
        return json.load(f)["entries"]


@pytest.mark.parametrize("entry", _load_entries(), ids=lambda e: e["munsell"])
def test_munsell_to_lab_matches_mobile_client(entry):
    # The fixture's munsell string carries the hue token directly:
    # "7.5YR 5/4" -> "7.5YR", and the neutral "N 2.5" -> "N".
    hue_str = entry["munsell"].split()[0]
    result = munsell_to_lab(
        MUNSELL_COLOR_REF, MUNSELL_REF, [hue_str, entry["value"], entry["chroma"]]
    )

    assert result is not None, (
        f"lookup returned None for {entry['munsell']} "
        f"(hue={hue_str}, value={entry['value']}, chroma={entry['chroma']})"
    )

    expected = entry["lab"]
    actual_l, actual_a, actual_b = result
    assert abs(actual_l - expected["L"]) <= LAB_TOLERANCE, (
        f"L channel: lib={actual_l:.2f}, client={expected['L']:.2f}"
    )
    assert abs(actual_a - expected["A"]) <= LAB_TOLERANCE, (
        f"A channel: lib={actual_a:.2f}, client={expected['A']:.2f}"
    )
    assert abs(actual_b - expected["B"]) <= LAB_TOLERANCE, (
        f"B channel: lib={actual_b:.2f}, client={expected['B']:.2f}"
    )
