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
