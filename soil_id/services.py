import json
import re
import urllib

import numpy as np
import pandas as pd
import requests
from db import save_soilgrids_output
from pandas import json_normalize


def get_elev_data(lon, lat):
    params = {
        "x": lon,
        "y": lat,
        "units": "Meters",
        "output": "json",
    }

    elev_url = "https://nationalmap.gov/epqs/pqs.php"

    # Fetch data from the URL
    response = requests.get(elev_url, params=params, timeout=2)
    result = response.json()

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
    # Load in SSURGO data from SoilWeb
    # current production API
    # soilweb base url "https://casoilresource.lawr.ucdavis.edu/api/landPKS.php"

    # testing API
    params = urllib.parse.urlencode([("q", "spn"), ("lon", lon), ("lat", lat), ("r", 1000)])
    soilweb_url = f"https://soilmap2-1.lawr.ucdavis.edu/dylan/soilweb/api/landPKS.php?{params}"
    try:
        response = requests.get(soilweb_url, timeout=8)
        out = response.json()
    except requests.RequestException as e:
        print(f"Error fetching data from SoilWeb: {e}")
        out = {
            "ESD": False,
            "OSD_morph": False,
            "OSD_narrative": False,
            "hz": False,
            "spn": False,
        }

    return out


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
