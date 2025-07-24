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

import datetime
import json
import os
import time
import traceback
import requests

import pandas
import numpy


def get_rfv(cf):
    if 0 <= cf < 2:
        return "0-1%"
    elif 2 <= cf < 16:
        return "1-15%"
    elif 16 <= cf < 36:
        return "15-35%"
    elif 36 <= cf < 61:
        return "35-60%"
    elif 61 <= cf <= 100:
        return ">60%"


def call_legacy(lat, lon, pedon):
    requests.get(f"https://api.landpotential.org/soilidlist?longitude={lon}&latitude={lat}")

    base = f"https://api.landpotential.org/soilidrank?longitude={lon}&latitude={lat}"
    depths = "&".join(
        [
            f"soilHorizon{idx + 1}_Depth={botdep}"
            for idx, botdep in enumerate(pedon["BOTDEP"].values.tolist())
        ]
    )
    rfv = "&".join(
        [
            f"soilHorizon{idx + 1}_RFV={get_rfv(rfv)}"
            for idx, rfv in enumerate(pedon["RFV"].values.tolist())
            if not numpy.isnan(rfv)
        ]
    )
    texture = "&".join(
        [
            f"soilHorizon{idx + 1}={textClass.upper()}"
            for idx, textClass in enumerate(pedon["textClass"].values.tolist())
        ]
    )
    color = "&".join(
        [
            f"soilHorizon{idx + 1}_LAB={','.join([f'{v}' for v in lab])}"
            for idx, lab in enumerate(pedon[["L", "A", "B"]].values.tolist())
        ]
    )
    req = "&".join([base, depths, rfv, texture, color])

    print(req)

    resp = requests.get(req)

    return resp.json()


test_data_df = pandas.read_csv(
    os.path.join(os.path.dirname(__file__), "../global/global_pedon_test_dataset.csv")
)
pedons = test_data_df.groupby(by=["ID"])

current_time_filename = datetime.datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
result_file_name = os.path.join(
    os.path.dirname(__file__),
    f"bulk_test_result_{current_time_filename}.jsonl",
)

print(f"logging results to {result_file_name}")
# buffering=1 is line buffering
with open(result_file_name, "w", buffering=1) as result_file:
    result_agg = {}

    for pedon_key, pedon in pedons:
        lat = pedon["Y_LatDD"].values[0]
        lon = pedon["X_LonDD"].values[0]
        result_record = {
            "pedon_key": pedon_key[0],
            "pedon_name": pedon["Description_fao90"].values[0],
            "lat": lat,
            "lon": lon,
        }

        if not isinstance(result_record["pedon_name"], str):
            result_record["result"] = "unknown"
        else:
            start_time = time.perf_counter()
            try:
                result_record["rank_result"] = call_legacy(lat, lon, pedon)
            except Exception:
                result_record["traceback"] = traceback.format_exc()
            result_record["execution_time_s"] = time.perf_counter() - start_time

        result_file.write(json.dumps(result_record) + "\n")
