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

"""Regression tests for process_distance_scores top-12 group selection (#398)."""

import pandas as pd

from soil_id.utils import process_distance_scores


def _make_input():
    """13 low-probability filler groups (names sort first) + one high-probability
    group whose name sorts LAST. With >12 groups, the alphabetically-first-12 bug
    would drop the high-probability group solely because of its name."""
    rows = []
    for i in range(13):
        rows.append(
            {
                "compname": f"aaa{i:02d}",
                "mukey": f"mu{i}",
                "cokey": f"ck{i}",
                "comppct_r": 5,
                "distance": 900.0,
            }
        )
    rows.append(
        {"compname": "zzz", "mukey": "muZ", "cokey": "ckZ", "comppct_r": 100, "distance": 10.0}
    )
    return pd.DataFrame(rows)


def test_top12_selects_highest_cond_prob_not_alphabetical():
    out = process_distance_scores(_make_input(), ExpCoeff=-0.002)
    kept = set(out["compname"])
    assert len(kept) == 12
    # The high-probability group must survive the top-12 cut despite sorting last.
    assert "zzz" in kept, f"highest-cond_prob group was dropped; kept={sorted(kept)}"
    # It should indeed be the most probable group.
    assert out.loc[out.compname == "zzz", "cond_prob"].iloc[0] == out["cond_prob"].max()


def test_min_dist_written_back():
    # The per-group min_dist transform must reach the returned frame (#398).
    out = process_distance_scores(_make_input(), ExpCoeff=-0.002)
    assert "min_dist" in out.columns
    for _, g in out.groupby("compname"):
        assert g["min_dist"].iloc[0] == g["distance"].min()
