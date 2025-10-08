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

import logging
import re

import pandas as pd
import requests


def get_elev_data(lon, lat):
    """
    Fetch elevation data from the National Map's Elevation Point Query Service.

    Args:
        lon (float): Longitude of the location for which to fetch elevation data.
        lat (float): Latitude of the location for which to fetch elevation data.

    Returns:
        dict: A dictionary containing the elevation data or error information if the request fails.
    """

    base_url = "https://epqs.nationalmap.gov/v1/json"
    params = {
        "x": lon,  # Longitude of the location
        "y": lat,  # Latitude of the location
        "wkid": 4326,  # Well-known ID for geographic coordinate systems
        "units": "Meters",  # Output units for elevation
        "includeDate": False,  # Option to include the date in the response
    }
    result = None

    try:
        response = requests.get(base_url, params=params, timeout=2)
        logging.info(f"{round(round(response.elapsed.total_seconds(), 2), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()
    except requests.ConnectionError as err:
        logging.error(f"Elevation: failed to connect: {err}")
    except requests.Timeout:
        logging.error("Elevation: timed out")
    except requests.RequestException as err:
        logging.error(f"Elevation: error: {err}")

    return result


def get_soil_series_data(mucompdata_pd, OSD_compkind):
    series_name = [
        re.sub("[0-9]+", "", compname).strip()
        for compname in mucompdata_pd["compname"]
        if compname in OSD_compkind
    ]

    if not series_name:
        logging.error("Soil series data: empty series")
        logging.debug(f"mucompdata: {mucompdata_pd}")
        logging.debug(f"OSD: {OSD_compkind}")
        return None

    base_url = "https://casoilresource.lawr.ucdavis.edu/api/soil-series.php"
    params = {"q": "site_hz", "s": series_name}
    result = {}

    try:
        response = requests.get(base_url, params=params, timeout=3)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)} seconds: {base_url}")
        response.raise_for_status()
        result = response.json()
    except requests.ConnectionError as err:
        logging.error(f"Soil series data: failed to connect: {err}")
    except requests.Timeout:
        logging.error("Soil series data: timed out")
    except requests.RequestException as err:
        logging.error(f"Soil series data: request error: {err}")

    return result


def get_soilgrids_property_data(lon, lat):
    # SoilGrids250
    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"
    params = [
        ("lon", lon),
        ("lat", lat),
        ("property", "cfvo"),
        ("property", "cec"),
        ("property", "clay"),
        ("property", "phh2o"),
        ("property", "sand"),
        ("value", "mean"),
    ]
    result = None

    try:
        response = requests.get(base_url, params=params, timeout=160)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError as err:
        logging.error(f"Soilgrids properties: failed to connect: {err}")
    except requests.Timeout:
        logging.error("Soilgrids properties: timed out")
    except requests.RequestException as err:
        logging.error(f"Soilgrids properties: error: {err}")

    return result if result is not None else {"status": "unavailable"}


def get_soilgrids_classification_data(lon, lat):
    # Fetch SG wRB Taxonomy
    base_url = "https://rest.isric.org/soilgrids/v2.0/classification/query"
    params = [("lon", lon), ("lat", lat), ("number_classes", 3)]
    result = None

    try:
        response = requests.get(base_url, params=params, timeout=160)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError as err:
        logging.error(f"Soilgrids classification: failed to connect: {err}")
    except requests.Timeout:
        logging.error("Soilgrids classification: timed out")
    except requests.RequestException as err:
        logging.error(f"Soilgrids classification: error: {err}")

    return result


def get_soilweb_data(lon, lat):
    """
    Fetch SSURGO data from the SoilWeb API for a specified longitude and latitude.

    Args:
    lon (float): Longitude of the location for which to fetch soil data.
    lat (float): Latitude of the location for which to fetch soil data.

    Returns:
    dict: A dictionary containing soil data or error information if the request fails.
    """
    base_url = "https://casoilresource.lawr.ucdavis.edu/api/landPKS.php"
    params = {
        "q": "spn",  # Query type - static for this function's purpose
        "lon": lon,
        "lat": lat,
        "r": 1000,  # Radius (in TODO: units)
    }
    result = None

    try:
        response = requests.get(base_url, params=params, timeout=8)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError as err:
        logging.error(f"SoilWeb: failed to connect: {err}")
    except requests.Timeout:
        logging.error("SoilWeb: timed out")
    except requests.RequestException as err:
        logging.error(f"SoilWeb: error: {err}")

    return (
        result
        if result is not None
        else {
            "ESD": False,
            "OSD_morph": False,
            "OSD_narrative": False,
            "hz": False,
            "spn": False,
        }
    )


# SDA = Soil Data Access
def sda_return(propQry):
    """
    Queries data from the USDA's Soil Data Mart (SDM) Tabular Service and returns
    it as a pandas DataFrame.
    """
    base_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
    request_data = {"format": "JSON+COLUMNNAME", "query": propQry}
    result = None

    try:
        response = requests.post(base_url, json=request_data, timeout=6)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

        # If dictionary key "Table" is found, normalize the data and return as DataFrame
        result = pd.json_normalize(result) if "Table" in result else None

    except requests.ConnectionError as err:
        logging.error(f"USDA service: failed to connect: {err}")
    except requests.Timeout:
        logging.error("USDA service: timed out")
    except requests.RequestException as err:
        logging.error(f"USDA service: error: {err}")

    return result
