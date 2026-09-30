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
    When candidates are tightly clustered, the un-ranged normalization divides by the
    slice's own (tiny) min-max range, so a small real difference is amplified to a
    near-maximal distance (the "rubber ruler"). Passing fixed theoretical_ranges
    normalizes by the fixed span instead, so the same small difference stays small.
    """
    # rows: [pedon, candidate A, candidate B]; one numeric feature, all near 10.
    X = np.array([[10.5], [10.0], [11.0]])

    d_unranged = gower_distances(X)[1, 2]  # A vs B, data-derived denom (=1)
    d_ranged = gower_distances(X, theoretical_ranges=[80.0])[1, 2]  # denom = fixed span 80

    assert d_unranged > 0.9, f"expected rubber-ruler amplification, got {d_unranged}"
    assert d_ranged < 0.2, f"expected fixed-range damping, got {d_ranged}"


def test_fixed_range_denominator_is_independent_of_slice_spread():
    """
    Fixed-range normalization divides by the theoretical span, NOT the per-slice data
    spread — so a pit/candidate distance is identical regardless of what OTHER
    candidates are in the slice. (Supersedes the old #377 floor max(slice_range,
    0.1*span), which still tracked the slice spread whenever it exceeded the floor.)
    """
    span = 80.0
    # Same pit (0) and candidate-of-interest (16); the third row changes the spread.
    tight = np.array([[0.0], [16.0], [20.0]])  # slice spread 20
    wide = np.array([[0.0], [16.0], [80.0]])  # slice spread 80
    d_tight = gower_distances(tight, theoretical_ranges=[span])[0, 1]
    d_wide = gower_distances(wide, theoretical_ranges=[span])[0, 1]
    # denom is the fixed span in both -> |0-16|/80 = 0.2, unaffected by the third row.
    assert abs(d_tight - 0.2) < 1e-9, d_tight
    assert abs(d_wide - 0.2) < 1e-9, d_wide
    assert abs(d_tight - d_wide) < 1e-9


def test_complete_data_distance_unchanged():
    """
    #377 part 2 invariant: with no missing data, the NaN-aware distance is exactly
    the old weighted mean of |normalized diff| — so complete-data results (and
    their snapshots) don't move; only missing-data pairs change.
    """
    X = np.array([[0.0, 10.0], [1.0, 20.0], [0.5, 30.0]])
    D = gower_distances(X)
    assert D.shape == (3, 3)
    assert np.allclose(np.diag(D), 0.0)
    assert np.allclose(D, D.T)
    assert not np.isnan(D).any()
    # feat0 range 1 -> denom 1; feat1 range 20 -> denom 20; equal weights.
    # D[0,1] = mean(|0-1|, |0-0.5|) = mean(1.0, 0.5) = 0.75
    assert abs(D[0, 1] - 0.75) < 1e-6


def test_missing_feature_skipped_not_imputed():
    """A feature missing on one side is skipped; distance uses the present ones."""
    X = np.array([[0.0, 10.0], [1.0, np.nan]])
    D = gower_distances(X, theoretical_ranges=[1.0, 100.0])
    # only feature 0 is shared; normalized diff there is 1.0
    assert not np.isnan(D[0, 1])
    assert abs(D[0, 1] - 1.0) < 1e-6


def test_all_missing_row_is_nan_for_penalty():
    """
    A row with no data at all (a component with no soil at this depth) yields a NaN
    distance to a pedon that has data — which the callers' infill turns into the
    max-dissimilarity penalty (instead of the old mean-imputed near-match).
    """
    X = np.array([[10.0], [np.nan]])
    D = gower_distances(X)
    assert D[0, 0] == 0.0
    assert np.isnan(D[0, 1])
    assert np.isnan(D[1, 0])
