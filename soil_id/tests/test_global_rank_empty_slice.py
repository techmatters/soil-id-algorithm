# Copyright © 2026 Technology Matters
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

"""Regression tests for the empty-feature depth-slice guard in global rank.

rank_soils_global builds a per-depth-slice feature matrix and feeds it to
gower_distances. When a depth slice has zero usable feature columns (e.g. a gap in
the user's horizons, or horizon data that doesn't reach the slice depth), the raw
gower path passes a shape ``(n, 0)`` array into SimpleImputer and raises
"Found array with 0 feature(s)…", failing the whole ranking. ``_slice_gower_distance``
guards that case. These tests cover the guard directly so they need no database or
network — unlike the live ``test_global_integration`` path.
"""

import numpy as np
import pandas as pd
import pytest

from soil_id.global_soil import _slice_gower_distance
from soil_id.utils import gower_distances


def test_zero_feature_slice_returns_all_nan_matrix():
    # 11 components, but no feature columns — the exact shape from the Sentry crash.
    slice_mat = pd.DataFrame(index=range(11))
    assert slice_mat.shape == (11, 0)

    result = _slice_gower_distance(slice_mat)

    # An (n, n) all-NaN matrix: same shape gower would have returned, and the
    # downstream masked-average/NaN-infill steps treat it as "no information".
    assert result.shape == (11, 11)
    assert np.isnan(result).all()


def test_zero_feature_slice_would_otherwise_crash_gower():
    # Characterization: the raw gower path crashes on a 0-feature array. This is
    # exactly what the guard above prevents; if gower ever stops raising here, the
    # guard's rationale should be revisited.
    slice_mat = pd.DataFrame(index=range(11))
    with pytest.raises(ValueError):
        gower_distances(slice_mat)


def test_nonempty_slice_passes_through_to_gower():
    # With at least one feature column, the helper must be a transparent passthrough.
    slice_mat = pd.DataFrame({"sand": [10.0, 20.0, 30.0], "clay": [5.0, 15.0, 25.0]})

    result = _slice_gower_distance(slice_mat)
    expected = gower_distances(slice_mat)

    assert np.array_equal(result, expected, equal_nan=True)
