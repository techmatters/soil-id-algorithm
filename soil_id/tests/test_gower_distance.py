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

import numpy as np

from soil_id.utils import gower_distances


def test_fixed_ranges_dampen_clustered_slices():
    """
    #377: when candidates are tightly clustered, the un-ranged normalization
    divides by the slice's own (tiny) min-max range, so a small real difference is
    amplified to a near-maximal distance (the "rubber ruler"). Passing fixed
    theoretical_ranges floors the denominator (max(slice_range, 0.1*span)), so the
    same small difference stays a small distance.
    """
    # rows: [pedon, candidate A, candidate B]; one numeric feature, all near 10.
    X = np.array([[10.5], [10.0], [11.0]])

    d_unranged = gower_distances(X)[1, 2]  # A vs B, data-derived denom (=1)
    d_ranged = gower_distances(X, theoretical_ranges=[80.0])[1, 2]  # denom floored to 8

    assert d_unranged > 0.9, f"expected rubber-ruler amplification, got {d_unranged}"
    assert d_ranged < 0.2, f"expected fixed-range damping, got {d_ranged}"


def test_fixed_range_floor_only_lifts_small_ranges():
    """
    The floor is max(slice_range, 0.1*span): when the slice already spans more than
    10% of the theoretical range, the fixed range must not change anything.
    """
    X = np.array([[50.0], [10.0], [90.0]])  # slice_range = 80 >> 0.1*80 = 8
    d_unranged = gower_distances(X)[1, 2]
    d_ranged = gower_distances(X, theoretical_ranges=[80.0])[1, 2]
    assert abs(d_unranged - d_ranged) < 1e-9
