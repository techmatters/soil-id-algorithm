import logging
import time

import soil_id.config
from soil_id.us_soil import list_soils, rank_soils


def test_true():
    assert True


def test_soil_location():
    test_locations = [
        {"lon": -101.9733687, "lat": 33.81246789, "plot_id": 3001},
        {"lon": -121.0347381, "lat": 45.88932423, "plot_id": 3002},
        {"lon": -85.50621214, "lat": 39.26009312, "plot_id": 3003},
        {"lon": -94.31005777, "lat": 42.63413723, "plot_id": 3004},
        {"lon": -99.55016693, "lat": 37.48216451, "plot_id": 3005},
        {"lon": -157.2767099, "lat": 62.32776717, "plot_id": 3006},
        {"lon": -156.4422738, "lat": 63.52666854, "plot_id": 3007},
        {"lon": -119.4596489, "lat": 43.06450312, "plot_id": 3008},
        {"lon": -69.28246582, "lat": 47.21392200, "plot_id": 3009},
        {"lon": -158.4018264, "lat": 60.42282639, "plot_id": 3010},
        {"lon": -121.8166, "lat": 48.6956, "plot_id": 3011},
    ]

    # Dummy Soil Profile Data (replicating the structure provided)
    soilHorizon = ["LOAM"] * 7
    horizonDepth = [1, 10, 20, 50, 70, 100, 120]
    rfvDepth = ["0-1%"] * 7
    lab_Color = [[41.24, 2.54, 21.17]] * 7
    bedrock = None
    pSlope = "15"
    pElev = None
    cracks = False
    site_calc = False

    for item in test_locations:
        logging.info(f"Testing {item['lon']}, {item['lat']}, {item['plot_id']}")
        start_time = time.perf_counter()
        result_list = list_soils(item["lon"], item["lat"], None, site_calc)
        logging.info(f"...time: {(time.perf_counter()-start_time):.2f}s")
        if site_calc:
            result_rank = rank_soils(
                item["lon"],
                item["lat"],
                soilHorizon,
                horizonDepth,
                rfvDepth,
                lab_Color,
                pSlope,
                pElev,
                bedrock,
                cracks,
                plot_id=None,
            )
        print(result_list)
        if site_calc:
            print(result_rank)

    soil_id.config.TEMP_DIR.cleanup()
