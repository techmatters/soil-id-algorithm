import config
from us_soil import getSoilLocationBasedUS, rankPredictionUS


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
        {"lon": -69.28246582, "lat": 47.213922, "plot_id": 3009},
        {"lon": -158.4018264, "lat": 60.42282639, "plot_id": 3010},
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
    for item in test_locations:
        print(f"Testing {item['lon']},{item['lat']},{item['plot_id']}")
        result_list = getSoilLocationBasedUS(item["lon"], item["lat"], None)
        result_rank = rankPredictionUS(
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
        print(result_rank)

    config.TEMP_DIR.cleanup()
