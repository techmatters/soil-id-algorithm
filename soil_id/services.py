import json
import re

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
    # Define the base URL for the Elevation Point Query Service API
    elev_url = "https://epqs.nationalmap.gov/v1/json"

    # Parameters for the API request
    params = {
        "x": lon,  # Longitude of the location
        "y": lat,  # Latitude of the location
        "wkid": 4326,  # Well-known ID for geographic coordinate systems
        "units": "Meters",  # Output units for elevation
        "includeDate": False,  # Option to include the date in the response
    }

    try:
        # Perform the GET request with specified parameters and a timeout
        response = requests.get(elev_url, params=params, timeout=2)
        # Decode the JSON response into a dictionary
        result = response.json()
    except requests.RequestException as e:
        # Log any request-related errors and return a predefined error dictionary
        print(f"Error fetching elevation data: {e}")
        result = {"error": "Failed to fetch elevation data"}

    return result


def get_esd_data(ecositeID, esd_geo, ESDcompdata_pd):
    class_url = "https://edit.jornada.nmsu.edu/services/downloads/esd/%s/class-list.json" % (
        esd_geo
    )

    try:
        response = requests.get(class_url, timeout=4)
        response.raise_for_status()

        ESD_list = response.json()

        ESD_list_pd = json_normalize(ESD_list["ecoclasses"])[["id", "legacyId"]]
        esd_url = []

        if isinstance(ESD_list, list):
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

    except requests.exceptions.RequestException as err:
        ESDcompdata_pd["esd_url"] = pd.Series(np.repeat("", len(ecositeID))).values
        print("An error occurred:", err)

    return ESDcompdata_pd


def get_soil_series_data(mucompdata_pd, OSD_compkind):
    series_name = [
        re.sub("[0-9]+", "", compname)
        for compname in mucompdata_pd["compname"]
        if compname in OSD_compkind
    ]

    params = {"q": "site_hz", "s": series_name}

    series_url = "https://casoilresource.lawr.ucdavis.edu/api/soil-series.php"
    response = requests.get(series_url, params=params, timeout=3)
    result = response.json()

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

    sg_api = "https://rest.isric.org/soilgrids/v2.0/properties/query"

    try:
        response = requests.get(sg_api, params=params, timeout=160)
        response.raise_for_status()
        sg_out = response.json()

    except requests.RequestException:
        if plot_id is not None:
            save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
        sg_out = {"status": "unavailable"}

    return sg_out


def get_soilgrids_classification_data(lon, lat, plot_id):
    # Fetch SG wRB Taxonomy
    params = [("lon", lon), ("lat", lat), ("number_classes", 3)]
    sg_api = "https://rest.isric.org/soilgrids/v2.0/classification/query"

    try:
        response = requests.get(sg_api, params=params, timeout=160)
        response.raise_for_status()
        sg_tax = response.json()

    except requests.RequestException:
        if plot_id is not None:
            # Assuming the function `save_soilgrids_output` exists elsewhere in the code
            save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
        sg_tax = None

    return sg_tax


def get_soilweb_data(lon, lat):
    """
    Fetch SSURGO data from the SoilWeb API for a specified longitude and latitude.

    Args:
    lon (float): Longitude of the location for which to fetch soil data.
    lat (float): Latitude of the location for which to fetch soil data.

    Returns:
    dict: A dictionary containing soil data or error information if the request fails.
    """
    # Base URL for the SoilWeb API - moved here to keep it close to its usage
    soilweb_url = "https://soilmap2-1.lawr.ucdavis.edu/dylan/soilweb/api/landPKS.php"

    # Parameters for the API request
    params = {
        "q": "spn",  # Query type - static for this function's purpose
        "lon": lon,  # Longitude parameter
        "lat": lat,  # Latitude parameter
        "r": 1000,  # Radius parameter - static for this function's purpose
    }

    try:
        # Perform the GET request with a timeout to prevent hanging
        response = requests.get(soilweb_url, params=params, timeout=8)
        # Decode JSON response into a dictionary
        response_data = response.json()
    except requests.RequestException as e:
        # Log any request-related errors and return a predefined error dictionary
        print(f"Error fetching data from SoilWeb: {e}")
        response_data = {
            "ESD": False,
            "OSD_morph": False,
            "OSD_narrative": False,
            "hz": False,
            "spn": False,
        }

    return response_data


def sda_return(propQry):
    """
    Queries data from the USDA's Soil Data Mart (SDM) Tabular Service and returns
    it as a pandas DataFrame.
    """
    base_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
    request_data = {"format": "JSON+COLUMNNAME", "query": propQry}

    try:
        # Send POST request using the requests library
        response = requests.post(base_url, json=request_data, timeout=6)
        response.raise_for_status()  # Raise an exception for HTTP errors

        # Convert the returned JSON into a Python dictionary
        qData = response.json()

        # If dictionary key "Table" is found, normalize the data and return as DataFrame
        if "Table" in qData:
            qDataPD = pd.json_normalize(qData)
            return qDataPD
        else:
            return None

    except requests.ConnectionError:
        print("Failed to connect to the USDA service.")
        return None
    except requests.Timeout:
        print("Request to USDA service timed out.")
        return None
    except requests.RequestException as err:
        print(f"An error occurred: {err}")
        return None
