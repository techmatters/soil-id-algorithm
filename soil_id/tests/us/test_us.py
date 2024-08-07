import logging
import time

from soil_id.us_soil import list_soils, rank_soils

test_locations = [
    {"lon": -101.9733687, "lat": 33.81246789},
    {"lon": -121.0347381, "lat": 45.88932423},
    {"lon": -85.50621214, "lat": 39.26009312},
    {"lon": -94.31005777, "lat": 42.63413723},
    {"lon": -99.55016693, "lat": 37.48216451},
    {"lon": -157.2767099, "lat": 62.32776717},
    {"lon": -156.4422738, "lat": 63.52666854},
    {"lon": -119.4596489, "lat": 43.06450312},
    {"lon": -69.28246582, "lat": 47.21392200},
    {"lon": -158.4018264, "lat": 60.42282639},
    {"lon": -121.8166, "lat": 48.6956},
]


def test_soil_location():
    # Dummy Soil Profile Data (replicating the structure provided)
    soilHorizon = ["LOAM"] * 7
    horizonDepth = [1, 10, 20, 50, 70, 100, 120]
    rfvDepth = ["0-1%"] * 7
    lab_Color = [[41.24, 2.54, 21.17]] * 7
    bedrock = None
    pSlope = "15"
    pElev = None
    cracks = False

    for item in test_locations:
        logging.info(f"Testing {item['lon']}, {item['lat']}")
        start_time = time.perf_counter()
        list_soils_result = list_soils(item["lon"], item["lat"])
        logging.info(f"...time: {(time.perf_counter()-start_time):.2f}s")
        rank_soils_result = rank_soils(
            item["lon"],
            item["lat"],
            list_soils_result,
            soilHorizon,
            horizonDepth,
            rfvDepth,
            lab_Color,
            pSlope,
            pElev,
            bedrock,
            cracks,
        )
        print(list_soils_result)
        print(rank_soils_result)


# def test_empty_rank():
#     SoilListOutputData = list_soils(test_locations[0]["lon"], test_locations[0]["lat"])
#     rank_soils(
#         lon=test_locations[0]["lon"],
#         lat=test_locations[0]["lat"],
#         SoilListOutputData,
#         soilHorizon=[],
#         horizonDepth=[],
#         rfvDepth=[],
#         lab_Color=[],
#         pSlope=None,
#         pElev=None,
#         bedrock=None,
#         cracks=None,
#     )


# #######################################################################
# import pandas as pd
# import pytest
# import os

# from soil_id.us_soil import list_soils, rank_soils
# def save_dataframe_to_csv(df, filename):
#     df.to_csv(filename, index=False)

# test_locations = [
#         {"lon": -101.9733687, "lat": 33.81246789},
#         # {"lon": -121.0347381, "lat": 45.88932423},
#         # {"lon": -85.50621214, "lat": 39.26009312},
#         # {"lon": -94.31005777, "lat": 42.63413723},
#         # {"lon": -99.55016693, "lat": 37.48216451},
#         # {"lon": -157.2767099, "lat": 62.32776717},
#         # {"lon": -156.4422738, "lat": 63.52666854},
#         # {"lon": -119.4596489, "lat": 43.06450312},
#         # {"lon": -69.28246582, "lat": 47.21392200},
#         # {"lon": -158.4018264, "lat": 60.42282639},
#         # {"lon": -121.8166, "lat": 48.6956},
#         #{"lat": 40.79861, "lon": -112.35477},
#         # {"lat": 35.599180, "lon": -120.491439},
#     ]

# # Create a sample DataFrame
# for item in test_locations:
#     list_soils_result = list_soils(item["lon"], item["lat"])
# # df = pd.DataFrame(list_soils_result)
# df = list_soils_result
# # Dummy Soil Profile Data (replicating the structure provided)
# soilHorizon = ["LOAM"] * 7
# horizonDepth = [1, 10, 20, 50, 70, 100, 120]
# rfvDepth = ["0-1%"] * 7
# lab_Color = [[41.24, 2.54, 21.17]] * 7
# bedrock = None
# pSlope = "15"
# pElev = None
# cracks = False

# rank_soils_result = rank_soils(
#     item["lon"],
#     item["lat"],
#     soilHorizon,
#     horizonDepth,
#     rfvDepth,
#     lab_Color,
#     pSlope,
#     pElev,
#     bedrock,
#     cracks,
#     list_soils_result,
# )
# # Save the DataFrame to a CSV file
# filename = '/mnt/c/Users/jmaynard/Documents/output.csv'
# save_dataframe_to_csv(rank_soils_result, filename)

# # Check if the file is created (additional test logic can be added here)
# assert os.path.exists(filename)


# if __name__ == "__main__":
#     pytest.main()
