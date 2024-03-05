from us_soil import getSoilLocationBasedUS


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

    for item in test_locations:
        print(f"Testing {item['lon']},{item['lat']},{item['plot_id']}")
        result = getSoilLocationBasedUS(item["lon"], item["lat"], None)
