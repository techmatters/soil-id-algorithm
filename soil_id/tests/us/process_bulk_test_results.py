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
df = pandas.DataFrame.from_records(result_dicts)

print("# Result proportions:\n")
print(df.groupby(by=["result"]).count()["pedon_key"] / len(df) * 100)
print("\n# Execution time deciles:\n")
print(pandas.qcut(df["execution_time_s"], q=10, retbins=True)[1])


crashes = df.groupby(by=["result"]).get_group(("crash",))
tracebacks = crashes["traceback"].unique().tolist()

print(f"\n# Unique crash tracebacks ({len(tracebacks)} unique, {len(crashes)} total):")

for t in tracebacks:
    print(t)
