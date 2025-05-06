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

parser = argparse.ArgumentParser("process_bulk_test_results")
parser.add_argument("file", type=argparse.FileType())
args = parser.parse_args()

result_lines = args.file.readlines()
result_dicts = [json.loads(line) for line in result_lines]

for result_record in result_dicts:
    if "result" in result_record and result_record["result"] == "unknown":
        continue

    if "rank_result" in result_record and "soilRank" in result_record["rank_result"]:
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

        def last_word(s):
            return s.split()[-1].lower()
        secondary_index = [
            i
            for i, match in enumerate(matches)
            if last_word(match["component"]) == last_word(result_record["pedon_name"])
            or last_word(match["component"]) == (last_word(result_record["pedon_name"]) + "s")
        ]
    
        if len(secondary_index) == 0:
            result_record["secondary_result"] = "missing"
        else:
            result_record["secondary_result"] = secondary_index[0] + 1

        result_record["all_soils"] = json.dumps([match["component"] for match in matches])
    else:
        result_record["result"] = "crash"


df = pandas.DataFrame.from_records(result_dicts)

result_groups = df.groupby(by=["result"])
secondary_result_groups = df.groupby(by=["secondary_result"])

print(f"# Total results: {len(df)}\n")
print("# Result proportions:\n")
print(result_groups.count()["pedon_key"] / len(df) * 100)

print("# Secondary result proportions:\n")
print(secondary_result_groups.count()["pedon_key"] / (len(df) - df["secondary_result"].isnull().sum()) * 100)

if len(df) < 11:
    print("\n# Execution times:\n")
    print(df["execution_time_s"].to_list())
else:
    print("\n# Execution time deciles:\n")
    print(pandas.qcut(df["execution_time_s"], q=10, retbins=True)[1])


if "crash" in result_groups.groups:
    crashes = result_groups.get_group(("crash",))
    # counts = df.value_counts(subset="traceback").sort_values(ascending=False)

    # print(f"\n# Unique crash tracebacks ({len(counts)} unique, {len(crashes)} total):\n")

    # for idx, (traceback, count) in enumerate(counts.to_dict().items()):
    #     example = crashes.loc[crashes["traceback"] == traceback].iloc[0]
    #     print(
    #         f"Traceback #{idx + 1}, occurred {count} times. Example pedon: {example['pedon_key']}, lat: {example['lat']}, lon: {example['lon']}"
    #     )
    #     lines = traceback.splitlines()
    #     indented_lines = ["  " + line for line in lines]
    #     print("\n".join(indented_lines) + "\n")