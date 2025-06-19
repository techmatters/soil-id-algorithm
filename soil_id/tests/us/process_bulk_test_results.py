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

import pandas

parser = argparse.ArgumentParser("process_bulk_test_results")
parser.add_argument("file", type=argparse.FileType())
args = parser.parse_args()

result_lines = args.file.readlines()
result_dicts = [json.loads(line) for line in result_lines]

for result_record in result_dicts:
    if "rank_result" in result_record:
        matches = result_record["rank_result"]["soilRank"]
        index = [
            i
            for i, match in enumerate(matches)
            if match["component"].lower() == result_record["pedon_name"].lower()
            or match["component"].lower() == (result_record["pedon_name"].lower() + "s")
        ]
        if len(index) == 0:
            result_record["result"] = "missing"
        else:
            result_record["result"] = index[0] + 1

        result_record["all_soils"] = json.dumps([match["component"] for match in matches])
    else:
        result_record["result"] = "crash"


df = pandas.DataFrame.from_records(result_dicts)

result_groups = df.groupby(by=["result"])

print(f"# Total results: {len(df)}\n")
print("# Result proportions:\n")
print(result_groups.count()["pedon_key"] / len(df) * 100)

if len(df) < 11:
    print("\n# Execution times:\n")
    print(df["execution_time_s"].to_list())
else:
    print("\n# Execution time quantiles:\n")
    print("50th percentile execution time:", df["execution_time_s"].quantile(0.50))
    print("90th percentile execution time:", df["execution_time_s"].quantile(0.90))
    print("99th percentile execution time:", df["execution_time_s"].quantile(0.99))
    print("99.9th percentile execution time:", df["execution_time_s"].quantile(0.999))


if "crash" in result_groups.groups:
    crashes = result_groups.get_group(("crash",))
    counts = df.value_counts(subset="traceback").sort_values(ascending=False)

    print(f"\n# Unique crash tracebacks ({len(counts)} unique, {len(crashes)} total):\n")

    for idx, (traceback, count) in enumerate(counts.to_dict().items()):
        example = crashes.loc[crashes["traceback"] == traceback].iloc[0]
        print(
            f"Traceback #{idx + 1}, occurred {count} times. Example pedon: {example['pedon_key']}, lat: {example['lat']}, lon: {example['lon']}"
        )
        lines = traceback.splitlines()
        indented_lines = ["  " + line for line in lines]
        print("\n".join(indented_lines) + "\n")

df[['pedon_key', 'pedon_name', 'lat', 'lon', 'result', 'all_soils']].to_csv('us_algorithm_results.csv', index=False)
