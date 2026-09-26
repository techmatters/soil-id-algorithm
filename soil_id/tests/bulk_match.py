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

"""
Shared scoring helpers for the bulk accuracy harnesses.

The bulk tests feed each real pedon's measured horizons into the ranking and ask:
at what rank does the algorithm place the pedon's known soil ("ground truth")?
Historically the match was an exact string compare of the algorithm's component
name against a single ``taxonname``, which badly undercounts hits (component is
often a series name at a different taxonomic level, and pedons carry several
historical classifications). These helpers centralise a more forgiving,
normalized match and compute recall@k so results are comparable across runs.

Nothing here changes algorithm output — it only scores stored results.
"""

import re


def normalize(name):
    """Lowercase, trim, and collapse internal whitespace. '' for missing/NaN."""
    if name is None:
        return ""
    s = str(name).strip().lower()
    if not s or s == "nan":
        return ""
    return " ".join(s.split())


def _variants(name, lenient):
    """
    Comparable forms of a name: exact + trailing-'s' tolerance, plus the form with
    a trailing disambiguation digit removed (list_soils appends one to duplicate
    component names, e.g. "Calcaric cambisols2" -> "Calcaric cambisols"), and the
    last word when lenient.
    """
    n = normalize(name)
    if not n:
        return set()
    forms = set()
    for base in {n, re.sub(r"\d+$", "", n).strip()}:
        if not base:
            continue
        forms |= {base, base.rstrip("s"), base + "s"}
        if lenient:
            forms.add(re.sub(r"\d+$", "", base.split()[-1]))
    return {f for f in forms if f}


def names_match(candidate, truth, lenient=False):
    return bool(_variants(candidate, lenient) & _variants(truth, lenient))


def best_rank(ranked_components, truths, lenient=False):
    """
    1-based rank of the first ranked component matching any acceptable ground-truth
    name, or None if none match ("missing").
    """
    truth_forms = set()
    for t in truths:
        truth_forms |= _variants(t, lenient)
    if not truth_forms:
        return None
    for i, comp in enumerate(ranked_components):
        if _variants(comp, lenient) & truth_forms:
            return i + 1
    return None


# Ranks that count crashes/misses distinctly from a numeric position.
CRASH = "crash"
MISSING = "missing"


def score_record(record, extra_truths=None, ranked_key="rank_result", lenient=False):
    """
    Score one bulk-result record: returns (rank, ranked_components).

    rank is a 1-based int, MISSING (truth not in the list), or CRASH (no ranking).
    Ground truth is the record's ``pedon_name`` plus any ``extra_truths`` (e.g. the
    US th_taxonname_* history keyed by pedon_key).
    """
    ranking = record.get(ranked_key)
    if not ranking or "soilRank" not in ranking:
        return CRASH, []
    ranked = [m.get("component", "") for m in ranking["soilRank"]]
    truths = [record.get("pedon_name")] + list(extra_truths or [])
    rank = best_rank(ranked, truths, lenient=lenient)
    return (rank if rank is not None else MISSING), ranked


def recall_summary(ranks, ks=(1, 3, 5)):
    """
    From an iterable of ranks (ints / MISSING / CRASH) return a dict with total,
    recall@k (fraction with a numeric rank <= k), missing and crash fractions.
    """
    ranks = list(ranks)
    total = len(ranks)
    numeric = [r for r in ranks if isinstance(r, int)]
    out = {"total": total}
    if total:
        for k in ks:
            out[f"recall@{k}"] = sum(1 for r in numeric if r <= k) / total
        out["found"] = len(numeric) / total
        out["missing"] = sum(1 for r in ranks if r == MISSING) / total
        out["crash"] = sum(1 for r in ranks if r == CRASH) / total
    return out
