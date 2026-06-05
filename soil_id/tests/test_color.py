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

from soil_id.color import munsell_to_lab
from soil_id.config import MUNSELL_COLOR_REF, MUNSELL_REF


def test_munsell_to_lab_known_color():
    lab = munsell_to_lab(MUNSELL_COLOR_REF, MUNSELL_REF, ["7.5YR", 5, 4])
    assert lab == pytest.approx([51.62524644, 9.878405711, 22.66377176])


def test_munsell_to_lab_neutral():
    lab = munsell_to_lab(MUNSELL_COLOR_REF, MUNSELL_REF, ["N", 8, 0])
    assert lab == pytest.approx([82.04578167, 0.0, 0.0])


def test_munsell_to_lab_off_table_returns_none():
    # A physically implausible (out-of-gamut) color is not in the table.
    assert munsell_to_lab(MUNSELL_COLOR_REF, MUNSELL_REF, ["10PB", 7, 33]) is None
