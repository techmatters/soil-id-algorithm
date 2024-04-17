import json
import re

import logging
import numpy as np
import pandas as pd
import requests
from db import save_soilgrids_output
from pandas import json_normalize


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

    try:
        response = requests.get(base_url, params=params, timeout=2)
        logging.info(f"{round(round(response.elapsed.total_seconds(), 2), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()
    except requests.ConnectionError:
        logging.error("Elevation: failed to connect")
        result = None
    except requests.Timeout:
        logging.error("Elevation: timed out")
        result = None
    except requests.RequestException as err:
        logging.error(f"Elevation: error: {err}")
        result = None

    return result


def get_esd_data(ecositeID, esd_geo, ESDcompdata_pd):
    base_url = "https://edit.jornada.nmsu.edu/services/downloads/esd/%s/class-list.json" % (esd_geo)

    try:
        response = requests.get(base_url, timeout=4)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()

        result = response.json()

        ESD_list_pd = json_normalize(result["ecoclasses"])[["id", "legacyId"]]
        esd_url = []

        if isinstance(result, list):
            esd_url.append("")
        else:
            for i in range(len(ecositeID)):
                if (
                    ecositeID[i] in ESD_list_pd["id"].tolist()
                    or ecositeID[i] in ESD_list_pd["legacyId"].tolist()
                ):
                    ecosite_edit_id = ESD_list_pd[
                        ESD_list_pd.apply(
                            lambda r: r.str.contains(ecositeID[i], case=False).any(),
                            axis=1,
                        )
                    ]["id"].values[0]
                    ES_URL_t = (
                        f"https://edit.jornada.nmsu.edu/catalogs/esd/{esd_geo}/{ecosite_edit_id}"
                    )
                    esd_url.append(ES_URL_t)
                else:
                    esd_url.append("")

        ESDcompdata_pd = ESDcompdata_pd.assign(esd_url=esd_url)

    except requests.ConnectionError:
        logging.error("ESD: failed to connect")
        ESDcompdata_pd["esd_url"] = pd.Series(np.repeat("", len(ecositeID))).values
    except requests.Timeout:
        logging.error("ESD: timed out")
        ESDcompdata_pd["esd_url"] = pd.Series(np.repeat("", len(ecositeID))).values
    except requests.RequestException as err:
        logging.error(f"ESD: error: {err}")
        ESDcompdata_pd["esd_url"] = pd.Series(np.repeat("", len(ecositeID))).values

    return ESDcompdata_pd


def get_soil_series_data(mucompdata_pd, OSD_compkind):
    series_name = [
        re.sub("[0-9]+", "", compname)
        for compname in mucompdata_pd["compname"]
        if compname in OSD_compkind
    ]

    params = {"q": "site_hz", "s": series_name}
    base_url = "https://casoilresource.lawr.ucdavis.edu/api/soil-series.php"

    try:
        response = requests.get(base_url, params=params, timeout=3)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError:
        logging.error("Soil series data: failed to connect")
        result = None
    except requests.Timeout:
        logging.error("Soil series data: timed out")
        result = None
    except requests.RequestException as err:
        logging.error(f"Soil series data: error: {err}")
        result = None

    return result


def get_soilgrids_property_data(lon, lat, plot_id):
    # SoilGrids250
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

    base_url = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    try:
        response = requests.get(base_url, params=params, timeout=160)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError:
        logging.error("Soilgrids properties: failed to connect")
        result = None
    except requests.Timeout:
        logging.error("Soilgrids properties: timed out")
        result = None
    except requests.RequestException as err:
        logging.error(f"Soilgrids properties: error: {err}")
        if plot_id is not None:
            # Assuming the function `save_soilgrids_output` exists elsewhere in the code
            save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
        result = None

    return result if result is not None else {"status": "unavailable"}


def get_soilgrids_classification_data(lon, lat, plot_id):
    # Fetch SG wRB Taxonomy
    params = [("lon", lon), ("lat", lat), ("number_classes", 3)]
    base_url = "https://rest.isric.org/soilgrids/v2.0/classification/query"

    try:
        response = requests.get(base_url, params=params, timeout=160)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError:
        logging.error("Soilgrids classification: failed to connect")
        result = None
    except requests.Timeout:
        logging.error("Soilgrids classification: timed out")
        result = None
    except requests.RequestException as err:
        logging.error(f"Soilgrids classification: error: {err}")
        if plot_id is not None:
            # Assuming the function `save_soilgrids_output` exists elsewhere in the code
            save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
        result = None

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
    base_url = "https://soilmap2-1.lawr.ucdavis.edu/dylan/soilweb/api/landPKS.php"

    params = {
        "q": "spn",  # Query type - static for this function's purpose
        "lon": lon,
        "lat": lat,
        "r": 1000,  # Radius (in TODO: units)
    }

    try:
        response = requests.get(base_url, params=params, timeout=8)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

    except requests.ConnectionError:
        logging.error("SoilWeb: failed to connect")
        result = None
    except requests.Timeout:
        logging.error("SoilWeb: timed out")
        result = None
    except requests.RequestException as err:
        logging.error(f"SoilWeb: error: {err}")
        result = None

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


def sda_return(propQry):
    """
    Queries data from the USDA's Soil Data Mart (SDM) Tabular Service and returns
    it as a pandas DataFrame.
    """
    base_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
    request_data = {"format": "JSON+COLUMNNAME", "query": propQry}

    try:
        response = requests.post(base_url, json=request_data, timeout=6)
        logging.info(f"{round(response.elapsed.total_seconds(), 2)}: {base_url}")
        response.raise_for_status()
        result = response.json()

        # If dictionary key "Table" is found, normalize the data and return as DataFrame
        return pd.json_normalize(result) if "Table" in result else None

    except requests.ConnectionError:
        logging.error("USDA service: failed to connect")
        return None
    except requests.Timeout:
        logging.error("USDA service: timed out")
        return None
    except requests.RequestException as err:
        logging.error(f"USDA service: error: {err}")
        return None
