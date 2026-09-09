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

import argparse
import json
import os

import pandas

from soil_id.tests.bulk_match import recall_summary, score_record

parser = argparse.ArgumentParser("process_bulk_test_results")
parser.add_argument("file", type=argparse.FileType())
parser.add_argument(
    "--lenient",
    action="store_true",
    help="also match on the last word of the taxon name (looser, less noisy)",
)
args = parser.parse_args()

# Ground-truth augmentation: the historical classifications (th_taxonname_1..5)
# for each pedon, so the match isn't limited to a single taxonname.
_csv = os.path.join(os.path.dirname(__file__), "US_SoilID_KSSL_LPKS_Testing.csv")
_gt = pandas.read_csv(_csv)
_th_cols = [c for c in _gt.columns if c.startswith("th_taxonname_")]
extra_truths = {}
for pedon_key, group in _gt.groupby("pedon_key"):
    names = set()
    for c in _th_cols:
        names.update(str(v) for v in group[c].dropna().unique())
    extra_truths[pedon_key] = [n for n in names if n and n != "nan"]

result_dicts = [json.loads(line) for line in args.file.readlines()]

for record in result_dicts:
    rank, ranked = score_record(
        record, extra_truths.get(record.get("pedon_key"), []), lenient=args.lenient
    )
    record["result"] = rank
    record["all_soils"] = json.dumps(ranked)

df = pandas.DataFrame.from_records(result_dicts)

print(f"# Total results: {len(df)}\n")

# Headline accuracy: recall@k against ground truth (+ historical names).
summary = recall_summary(df["result"].tolist())
print("# Accuracy (recall@k vs ground truth):\n")
for k in (1, 3, 5):
    print(f"  recall@{k}: {summary.get(f'recall@{k}', 0) * 100:.1f}%")
print(f"  found (any rank): {summary.get('found', 0) * 100:.1f}%")
print(f"  missing:          {summary.get('missing', 0) * 100:.1f}%")
print(f"  crash:            {summary.get('crash', 0) * 100:.1f}%")

print("\n# Result proportions (rank position / missing / crash):\n")
print(df.groupby(by=["result"]).count()["pedon_key"] / len(df) * 100)

if len(df) < 11:
    print("\n# Execution times:\n")
    print(df["execution_time_s"].to_list())
else:
    print("\n# Execution time quantiles:\n")
    print("50th percentile execution time:", df["execution_time_s"].quantile(0.50))
    print("90th percentile execution time:", df["execution_time_s"].quantile(0.90))
    print("99th percentile execution time:", df["execution_time_s"].quantile(0.99))
    print("99.9th percentile execution time:", df["execution_time_s"].quantile(0.999))

result_groups = df.groupby(by=["result"])
if "crash" in result_groups.groups:
    crashes = result_groups.get_group(("crash",))
    counts = df.value_counts(subset="traceback").sort_values(ascending=False)

    print(f"\n# Unique crash tracebacks ({len(counts)} unique, {len(crashes)} total):\n")

    for idx, (traceback, count) in enumerate(counts.to_dict().items()):
        example = crashes.loc[crashes["traceback"] == traceback].iloc[0]
        print(
            f"Traceback #{idx + 1}, occurred {count} times. Example pedon: "
            f"{example['pedon_key']}, lat: {example['lat']}, lon: {example['lon']}"
        )
        lines = traceback.splitlines()
        indented_lines = ["  " + line for line in lines]
        print("\n".join(indented_lines) + "\n")

df[["pedon_key", "pedon_name", "lat", "lon", "result", "all_soils"]].to_csv(
    "us_algorithm_results.csv", index=False
)
