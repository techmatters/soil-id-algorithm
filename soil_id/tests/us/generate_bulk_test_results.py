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

import datetime
import json
import os
import time
import traceback

import pandas

from soil_id.us_soil import list_soils, rank_soils

test_data_df = pandas.read_csv(
    os.path.join(os.path.dirname(__file__), "US_SoilID_KSSL_LPKS_Testing.csv")
)
pedons = test_data_df.groupby(by=["pedon_key"])

current_time_filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
result_file_name = os.path.join(
    os.path.dirname(__file__),
    f"bulk_test_result_{current_time_filename}.jsonl",
)

print(f"logging results to {result_file_name}")
with open(result_file_name, "w") as result_file:
    result_agg = {}

    for pedon_key, pedon in pedons:
        result_record = {"pedon_key": pedon_key[0], "pedon_name": pedon["taxonname"].values[0]}

        start_time = time.perf_counter()
        try:
            lat = pedon["y"].values[0]
            lon = pedon["x"].values[0]

            list_result = list_soils(lat=lat, lon=lon)

            result_record["rank_result"] = rank_soils(
                lat=lat,
                lon=lon,
                list_output_data=list_result,
                horizonDepth=pedon["hzdepb"].values.tolist(),
                soilHorizon=pedon["textclass"].values.tolist(),
                rfvDepth=pedon["fragvoltot"].values.tolist(),
                lab_Color=pedon[["L", "a", "b"]].values.tolist(),
                pSlope=pedon["slope"].values[0],
                pElev=None,
                bedrock=None,
                cracks=None,
            )
        except Exception:
            result_record["traceback"] = traceback.format_exc()
        result_record["execution_time_s"] = time.perf_counter() - start_time

        if "rank_result" in result_record:
            matches = result_record["rank_result"]["soilRank"]
            index = [
                i
                for i, match in enumerate(matches)
                if match["component"].lower() == result_record["pedon_name"].lower()
            ]
            if len(index) == 0:
                result_record["result"] = "missing"
            else:
                result_record["result"] = index[0] + 1
        else:
            result_record["result"] = "crash"

        if result_record["result"] not in result_agg:
            result_agg[result_record["result"]] = 0
        result_agg[result_record["result"]] = result_agg[result_record["result"]] + 1

        result_file.write(json.dumps(result_record) + "\n")
