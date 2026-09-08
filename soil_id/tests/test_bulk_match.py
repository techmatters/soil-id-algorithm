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

from soil_id.tests.bulk_match import (
    CRASH,
    MISSING,
    best_rank,
    names_match,
    normalize,
    recall_summary,
    score_record,
)


def test_normalize():
    assert normalize("  Typic  Haplustolls ") == "typic haplustolls"
    assert normalize(None) == ""
    assert normalize("nan") == ""


def test_names_match_trailing_s():
    assert names_match("Leptosol", "Leptosols")
    assert names_match("Lithic Leptosols", "lithic leptosol")
    assert not names_match("Cambisols", "Leptosols")


def test_best_rank_first_match_and_alternatives():
    ranked = ["Chromic Cambisols", "Lithic Leptosols", "Eutric Fluvisols"]
    assert best_rank(ranked, ["Lithic Leptosols"]) == 2
    assert best_rank(ranked, ["Nitisols"]) is None
    # any acceptable alternative counts
    assert best_rank(ranked, ["Nitisols", "Eutric Fluvisols"]) == 3


def test_lenient_last_word():
    ranked = ["Chromic Cambisols"]
    assert best_rank(ranked, ["Vertic Cambisols"], lenient=False) is None
    assert best_rank(ranked, ["Vertic Cambisols"], lenient=True) == 1


def test_score_record():
    hit = {
        "pedon_name": "Lithic Leptosols",
        "rank_result": {"soilRank": [{"component": "Lithic Leptosols"}, {"component": "X"}]},
    }
    miss = {"pedon_name": "Nitisols", "rank_result": {"soilRank": [{"component": "Cambisols"}]}}
    crash = {"pedon_name": "X"}
    assert score_record(hit)[0] == 1
    assert score_record(miss)[0] == MISSING
    assert score_record(crash)[0] == CRASH
    # extra_truths (e.g. US th_taxonname history) let an alternative name match
    assert score_record(miss, extra_truths=["Cambisols"])[0] == 1


def test_recall_summary():
    s = recall_summary([1, 2, 3, MISSING, CRASH])
    assert s["total"] == 5
    assert abs(s["recall@1"] - 1 / 5) < 1e-9
    assert abs(s["recall@3"] - 3 / 5) < 1e-9
    assert abs(s["found"] - 3 / 5) < 1e-9
    assert abs(s["missing"] - 1 / 5) < 1e-9
    assert abs(s["crash"] - 1 / 5) < 1e-9
