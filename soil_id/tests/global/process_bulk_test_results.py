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

import argparse
import json

import pandas

from soil_id.tests.bulk_match import recall_summary, score_record

parser = argparse.ArgumentParser("process_bulk_test_results")
parser.add_argument("file", type=argparse.FileType())
args = parser.parse_args()

result_dicts = [json.loads(line) for line in args.file.readlines()]

for record in result_dicts:
    if record.get("result") == "unknown":
        continue
    # strict = normalized exact match; lenient = also match on the taxon's last word
    # (e.g. "Lithic Leptosols" -> "Leptosols"), the looser WRB-group-level signal.
    strict, ranked = score_record(record, lenient=False)
    lenient, _ = score_record(record, lenient=True)
    record["result"] = strict
    record["secondary_result"] = lenient
    record["all_soils"] = json.dumps(ranked)

df = pandas.DataFrame.from_records(result_dicts)

print(f"# Total results: {len(df)}\n")

strict_summary = recall_summary(df["result"].tolist())
lenient_summary = recall_summary(df["secondary_result"].tolist())
print("# Accuracy (recall@k vs ground truth):\n")
for label, summary in (("strict", strict_summary), ("lenient/last-word", lenient_summary)):
    line = "  ".join(f"recall@{k}={summary.get(f'recall@{k}', 0) * 100:.1f}%" for k in (1, 3, 5))
    print(f"  {label:18} {line}  found={summary.get('found', 0) * 100:.1f}%")

print("\n# Result proportions (strict, rank / missing / crash):\n")
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

df[["pedon_key", "pedon_name", "lat", "lon", "result", "secondary_result", "all_soils"]].to_csv(
    "global_algorithm_results.csv", index=False
)
