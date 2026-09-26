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
Paired before/after comparison of two bulk-result files (same pedon set).

The absolute recall numbers from a single run are noisy (the ground-truth match is
crude), but a PAIRED delta is trustworthy: the match bias is constant across the
two runs, so a shift in recall@k measures whether a change helped or hurt. This is
the tool to validate an algorithm change against ground truth.

    python -m soil_id.tests.compare_bulk_test_results before.jsonl after.jsonl \\
        [--dataset us] [--lenient]

--dataset us augments the ground truth with the th_taxonname_* classification
history from US_SoilID_KSSL_LPKS_Testing.csv. --lenient also matches on the last
word of the taxon name.
"""

import argparse
import json
import os

from soil_id.tests.bulk_match import CRASH, MISSING, recall_summary, score_record


def load_us_extra_truths():
    """pedon_key -> [th_taxonname_1..5] (non-empty) from the US testing CSV."""
    import pandas

    csv = os.path.join(os.path.dirname(__file__), "us", "US_SoilID_KSSL_LPKS_Testing.csv")
    df = pandas.read_csv(csv)
    cols = [c for c in df.columns if c.startswith("th_taxonname_")]
    extra = {}
    for pedon_key, group in df.groupby("pedon_key"):
        names = set()
        for c in cols:
            names.update(str(v) for v in group[c].dropna().unique())
        extra[pedon_key] = [n for n in names if n and n != "nan"]
    return extra


def score_file(path, extra_truths, lenient):
    with open(path) as f:
        records = [json.loads(line) for line in f]
    scored = {}
    for r in records:
        pedon_key = r.get("pedon_key")
        rank, _ = score_record(r, extra_truths.get(pedon_key, []), lenient=lenient)
        scored[pedon_key] = rank
    return scored


def _fmt(summary, ks):
    parts = [f"n={summary['total']}"]
    for k in ks:
        parts.append(f"recall@{k}={summary.get(f'recall@{k}', 0):.3f}")
    parts.append(f"found={summary.get('found', 0):.3f}")
    parts.append(f"missing={summary.get('missing', 0):.3f}")
    parts.append(f"crash={summary.get('crash', 0):.3f}")
    return "  ".join(parts)


def main():
    parser = argparse.ArgumentParser("compare_bulk_test_results")
    parser.add_argument("before")
    parser.add_argument("after")
    parser.add_argument("--dataset", choices=["us", "global"], default="global")
    parser.add_argument("--lenient", action="store_true")
    args = parser.parse_args()

    extra = load_us_extra_truths() if args.dataset == "us" else {}
    before = score_file(args.before, extra, args.lenient)
    after = score_file(args.after, extra, args.lenient)

    ks = (1, 3, 5)
    print(f"# Bulk comparison ({args.dataset}, {'lenient' if args.lenient else 'strict'} match)\n")
    print(f"BEFORE  {_fmt(recall_summary(before.values(), ks), ks)}")
    print(f"AFTER   {_fmt(recall_summary(after.values(), ks), ks)}")

    common = sorted(set(before) & set(after), key=lambda k: str(k))
    print(f"\n# Paired movement over {len(common)} shared pedons:\n")

    def rank_val(r):
        # numeric rank -> itself; MISSING/CRASH -> large sentinel so "found" is an improvement
        return r if isinstance(r, int) else 10**6

    improved = worsened = unchanged = newly_found = newly_missing = 0
    examples = {"improved": [], "worsened": []}
    for key in common:
        b, a = before[key], after[key]
        bv, av = rank_val(b), rank_val(a)
        b_found, a_found = isinstance(b, int), isinstance(a, int)
        if not b_found and a_found:
            newly_found += 1
        if b_found and not a_found:
            newly_missing += 1
        if av < bv:
            improved += 1
            if len(examples["improved"]) < 5:
                examples["improved"].append((key, b, a))
        elif av > bv:
            worsened += 1
            if len(examples["worsened"]) < 5:
                examples["worsened"].append((key, b, a))
        else:
            unchanged += 1

    print(f"  improved (better rank): {improved}")
    print(f"  worsened (worse rank):  {worsened}")
    print(f"  unchanged:              {unchanged}")
    print(f"  newly found (was {MISSING}/{CRASH}): {newly_found}")
    print(f"  newly missing:          {newly_missing}")
    for kind in ("improved", "worsened"):
        if examples[kind]:
            print(f"\n  example {kind} (pedon_key: before -> after):")
            for key, b, a in examples[kind]:
                print(f"    {key}: {b} -> {a}")


if __name__ == "__main__":
    main()
