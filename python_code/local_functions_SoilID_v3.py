#####################################################################################################
#                                       Database and API Functions                                  #
#####################################################################################################
# Standard libraries
import collections
import csv
import json
import math
import os
import random
import re
import struct
import sys
import requests

# Third-party libraries
import colour
import geopandas as gpd
import MySQLdb
import numpy as np
import pandas as pd
import scipy.stats
from osgeo import gdal, ogr
from scipy.interpolate import CubicSpline
from scipy.spatial import distance
from scipy.stats import norm
from scipy.sparse import issparse
import shapely
from shapely.geometry import Point, Polygon, shape, LinearRing
from sklearn.metrics import pairwise
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.utils import validation
from numpy.linalg import cholesky
from flask import current_app

from scipy.linalg import sqrtm
from skbio.stats.composition import ilr
from skbio.stats.composition import ilr_inv

def getDataStore_Connection():
    """
    Establish a connection to the datastore using app configurations.

    Returns:
        Connection object if successful, otherwise exits the program.
    """
    try:
        conn = MySQLdb.connect(
            host=current_app.config['CLOUDSQL_IP'],
            user=current_app.config['CLOUDSQL_USER'],
            passwd=current_app.config['CLOUDSQL_PASSWORD'],
            db='apex'
        )
        return conn
    except Exception as err:
        print(err)
        sys.exit(str(err))

# def getDataStore_Connection():
#     try:
#         HOST = current_app.config['CLOUDSQL_IP']
#         USER = current_app.config['CLOUDSQL_USER']
#         PASSWORD = current_app.config['CLOUDSQL_PASSWORD']
#         DATABASE = 'apex'
#         conn = MySQLdb.connect(host=HOST, user=USER, passwd=PASSWORD, db=DATABASE)
#         return conn
#     except Exception, as err:
#         print err
#         sys.exit(str(err))
        
        
def save_model_output(plot_id, model_version, result_blob, soilIDRank_output_pd, mucompdata_cond_prob):
    """
    Save the output of the model to the 'landpks_soil_model' table.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO landpks_soil_model
        (plot_id, model_version, result_blob, soilIDRank_output_pd, mucompdata_cond_prob)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(sql, (plot_id, model_version, result_blob, soilIDRank_output_pd, mucompdata_cond_prob))
        conn.commit()

    except Exception as err:
        print(err)
        conn.rollback()
        return None
    finally:
        conn.close()

def save_soilgrids_output(plot_id, model_version, soilgrids_blob):
    """
    Save the output of the soil grids to the 'landpks_soilgrids_model' table.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO landpks_soilgrids_model
        (plot_id, model_version, soilgrids_blob)
        VALUES (%s, %s, %s)
        """
        cur.execute(sql, (plot_id, model_version, soilgrids_blob))
        conn.commit()

    except Exception as err:
        print(err)
        conn.rollback()
        return None
    finally:
        conn.close()

def save_rank_output(record_id, model_version, rank_blob):
    """
    Update the rank of the soil model in the 'landpks_soil_model' table.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        sql = """
        UPDATE landpks_soil_model
        SET soilrank = %s
        WHERE ID = %s AND model_version = %s
        """
        cur.execute(sql, (rank_blob, record_id, model_version))
        conn.commit()

    except Exception as err:
        print(err)
        conn.rollback()
        return None
    finally:
        conn.close()

def load_model_output(plot_id):
    """
    Load model output based on plot ID and model version.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()
        model_version = 2
        sql = f"SELECT ID, result_blob, soilIDRank_output_pd, mucompdata_cond_prob FROM  landpks_soil_model WHERE plot_id = {plot_id} AND model_version = {model_version} order by ID desc LIMIT 1"
        cur.execute(sql)
        results = cur.fetchall()
        for row in results:
            model_run = [row[0], row[1], row[2], row[3]]
        return model_run
    except Exception as err:
        print(err)
        return None
    finally:
        conn.close()

def get_WISE30sec_data(MUGLB_NEW_Select):
    """
    Retrieve WISE 30 second data based on selected MUGLB_NEW values.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()
        placeholders = ", ".join(["%s"] * len(MUGLB_NEW_Select))
        sql = f"SELECT MUGLB_NEW, COMPID, id, MU_GLOBAL, NEWSUID, SCID, PROP, CLAF,  PRID, Layer, TopDep, BotDep,  CFRAG,  SDTO,  STPC,  CLPC, CECS, PHAQ, ELCO, SU_name, FAO_SYS FROM  wise_soil_data WHERE MUGLB_NEW IN ({placeholders})"
        cur.execute(sql, MUGLB_NEW_Select)
        results = cur.fetchall()
        data = pd.DataFrame(results, columns=['MUGLB_NEW',  'COMPID', 'id', 'MU_GLOBAL', 'NEWSUID', 'SCID', 'PROP', 'CLAF',  'PRID', 'Layer', 'TopDep', 'BotDep',  'CFRAG',  'SDTO',  'STPC',  'CLPC', 'CECS', 'PHAQ', 'ELCO', 'SU_name', 'FAO_SYS'])
        return data
    except Exception as err:
        print(err)
        return None
    finally:
        conn.close()

def extract_WISE_data(lon, lat, file_path, layer_name=None, buffer_size=0.5):
    # Create LPKS point
    point = Point(lon, lat)
    point.crs = {'init': 'epsg:4326'}

    # Create bounding box to clip HWSD data around the point
    s = gpd.GeoSeries(point)
    bounding_box = s.buffer(buffer_size).total_bounds
    box_geom = shapely.geometry.box(*bounding_box)

    # Load HWSD data from the provided file_path
    hwsd = gpd.read_file(file_path, bbox=box_geom.bounds, driver="GPKG")

    # Filter data to consider unique map units
    mu_geo = hwsd[['MUGLB_NEW', 'geometry']].drop_duplicates(subset='MUGLB_NEW')

    distances = mu_geo.geometry.apply(lambda geom: pt2polyDist(geom, point))
    intersects = mu_geo.geometry.apply(lambda geom: point.intersects(geom))

    mu_id_dist = pd.DataFrame({
        'MUGLB_NEW': mu_geo['MUGLB_NEW'],
        'distance': distances.where(~intersects, 0),
    })
    mu_id_dist['distance'] = mu_id_dist.groupby('MUGLB_NEW')['distance'].transform(min)
    mu_id_dist = mu_id_dist.nsmallest(2, 'distance')

    hwsd = hwsd.drop(columns=['geometry'])
    hwsd = pd.merge(mu_id_dist, hwsd, on='MUGLB_NEW', how='left').drop_duplicates()

    MUGLB_NEW_Select = hwsd['MUGLB_NEW'].tolist()
    wise_data = getWISE30sec_data(MUGLB_NEW_Select)
    wise_data = pd.merge(wise_data, mu_id_dist, on='MUGLB_NEW', how='left')

    return wise_data
    

def get_WRB_descriptions(WRB_Comp_List):        
    """
    Retrieve WRB descriptions based on provided WRB component list.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()
        placeholders = ", ".join(["%s"] * len(WRB_Comp_List))
        sql = f"SELECT WRB_tax, Description_en, Management_en, Description_es, Management_es, Description_ks, Management_ks, Description_fr, Management_fr FROM wrb_fao90_desc WHERE WRB_tax IN ({placeholders})"
        cur.execute(sql, WRB_Comp_List)
        results = cur.fetchall()
        data = pd.DataFrame(results, columns=['WRB_tax', 'Description_en', 'Management_en', 'Description_es', 'Management_es', 'Description_ks', 'Management_ks', 'Description_fr', 'Management_fr'])
        return data
    except Exception as err:
        print(err)
        return None
    finally:
        conn.close()

def sda_return(propQry):
    """
    Queries data from the USDA's Soil Data Mart (SDM) Tabular Service and returns it as a pandas DataFrame.
    """
    base_url = "https://sdmdataaccess.nrcs.usda.gov/Tabular/SDMTabularService/post.rest"
    request_data = {
        "format": "JSON+COLUMNNAME",
        "query": propQry
    }

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
      
      
#####################################################################################################
#                                       Utility Functions                                           #
#####################################################################################################

def getSand(field):
    sand_percentages = {
        "sand": 92.0,
        "loamy sand": 80.0,
        "sandy loam": 61.5,
        "sandy clay loam": 62.5,
        "loam": 37.5,
        "silt": 10.0,
        "silt loam": 25.0,
        "silty clay loam": 10.0,
        "clay loam": 32.5,
        "sandy clay": 55.0,
        "silty clay": 10.0,
        "clay": 22.5
    }

    return sand_percentages.get(field.lower() if field else None, np.nan)


def getClay(field):
    clay_percentages = {
        "sand": 5.0,
        "loamy sand": 7.5,
        "sandy loam": 10.0,
        "sandy clay loam": 27.5,
        "loam": 17.0,
        "silt": 6.0,
        "silt loam": 13.5,
        "silty clay loam": 33.5,
        "clay loam": 33.5,
        "sandy clay": 45.0,
        "silty clay": 50.0,
        "clay": 70.0
    }

    return clay_percentages.get(field.lower() if field else None, np.nan)

def silt_calc(row):
    sand = row['sand']
    clay = row['clay']
    silt = 100 - (sand+clay)
    return silt
      
def getTexture(row, sand=None, silt=None, clay=None):
    sand = sand if sand is not None else row.get('sandtotal_r') or row.get('sand')
    silt = silt if silt is not None else row.get('silttotal_r') or row.get('silt')
    clay = clay if clay is not None else row.get('claytotal_r') or row.get('clay')

    silt_clay = silt + 1.5 * clay
    silt_2x_clay = silt + 2.0 * clay

    if silt_clay < 15:
        return "Sand"
    elif silt_clay < 30:
        return "Loamy sand"
    elif (7 <= clay <= 20 and sand > 52) or (clay < 7 and silt < 50):
        if silt_2x_clay >= 30:
            return "Sandy loam"
    elif 7 <= clay <= 27 and 28 <= silt < 50 and sand <= 52:
        return "Loam"
    elif silt >= 50 and ((12 <= clay < 27) or (silt < 80 and clay < 12)):
        return "Silt loam"
    elif silt >= 80 and clay < 12:
        return "Silt"
    elif 20 <= clay < 35 and silt < 28 and sand > 45:
        return "Sandy clay loam"
    elif 27 <= clay < 40 and sand <= 45:
        if sand > 20:
            return "Clay loam"
        else:
            return "Silty clay loam"
    elif clay >= 35 and sand >= 45:
        return "Sandy clay"
    elif clay >= 40:
        if silt >= 40:
            return "Silty clay"
        elif sand <= 45:
            return "Clay"

    return None  # Default return value

def getCF(cf):
    if 0 <= cf < 2:
        return 0
    elif 2 <= cf < 16:
        return 8
    elif 16 <= cf < 36:
        return 25
    elif 36 <= cf < 61:
        return 48
    elif 61 <= cf <= 100:
        return 80
    else:
        return np.nan

def getCF_fromClass(cf):
    cf_to_value = {
        "0-1%": 0,
        "0-15%": 0,
        "1-15%": 8,
        "15-35%": 25,
        "35-60%": 48,
        ">60%": 80
    }
    
    return cf_to_value.get(cf, np.nan)

def getOSDCF(cf):
    frag_vol_35 = ['gravelly', 'fine gravelly', 'medium gravelly', 'coarse gravelly', 
                   'cobbly', 'stony', 'bouldery', 'channery', 'flaggy']
    frag_vol_35_list = [re.compile(pattern) for pattern in frag_vol_35]

    frag_vol_60 = ['very gravelly', 'very cobbly', 'very stony', 'extremely bouldery', 
                   'very channery', 'very flaggy']
    frag_vol_60_list = [re.compile(pattern) for pattern in frag_vol_60]

    frag_vol_90 = ['extremely gravelly', 'extremely cobbly', 'extremely stony', 
                   'extremely channery', 'extremely flaggy']
    frag_vol_90_list = [re.compile(pattern) for pattern in frag_vol_90]

    if cf is None:
        return 8
    elif any(regex.match(cf) for regex in frag_vol_35_list):
        return 25
    elif any(regex.match(cf) for regex in frag_vol_60_list):
        return 48
    elif any(regex.match(cf) for regex in frag_vol_90_list):
        return 80
    else:
        return np.nan

def agg_data_layer(data, bottom, sd=2, depth=False):
    # Handle edge cases
    if np.isnan(bottom):
        return (pd.Series([np.nan]), pd.Series([np.nan])) if depth else pd.Series([np.nan])
    if bottom == 0:
        return (pd.Series([np.nan]), pd.Series([np.nan])) if depth else pd.Series([np.nan])

    # Define depth ranges
    ranges = [
        (1, ['sl1'], [1]),
        (11, ['sl1', 'sl2'], [1, bottom]),
        (21, ['sl1', 'sl2', 'sl3'], [1, 10, bottom]),
        (51, ['sl1', 'sl2', 'sl3', 'sl4'], [1, 10, 20, bottom]),
        (71, ['sl1', 'sl2', 'sl3', 'sl4', 'sl5'], [1, 10, 20, 50, bottom]),
        (101, ['sl1', 'sl2', 'sl3', 'sl4', 'sl5', 'sl6'], [1, 10, 20, 50, 70, bottom]),
        (120, ['sl1', 'sl2', 'sl3', 'sl4', 'sl5', 'sl6', 'sl7'], [1, 10, 20, 50, 70, 100, bottom]),
        (float('inf'), ['sl1', 'sl2', 'sl3', 'sl4', 'sl5', 'sl6', 'sl7'], [1, 10, 20, 50, 70, 100, 120])
    ]

    # Process data for each range
    for max_val, indices, depths in ranges:
        if bottom < max_val:
            data_d = [round(data[:d].mean(), sd) for d in depths]
            data_d = pd.Series(data_d, index=indices)
            d_lyrs = pd.Series(depths, index=indices)
            return (data_d, d_lyrs) if depth else data_d

def aggregate_data(data, bottom_depths, sd=2):
    if not bottom_depths or np.isnan(bottom_depths[0]):
        return pd.Series([np.nan])

    top_depths = [0] + bottom_depths[:-1]

    results = []

    for top, bottom in zip(top_depths, bottom_depths):
        mask = (data.index >= top) & (data.index <= bottom)
        data_subset = data[mask]
        if not data_subset.empty:
            result = round(data_subset.mean(), sd)
            results.append(result)
        else:
            results.append(np.nan)

    return pd.Series(results)
  
def getProfile(data, variable, c_bot=False):
    var = []
    var_grp = []
    var_pct_intpl = []
    var_pct_intpl_grp = []
    if variable == "sandtotal_r":
        for i in range(len(data)):
            if data["texture"].iloc[i] is None:
                var.append(data["sandtotal_r"].iloc[i])
                var_grp.append(np.nan)
            else:
                var.append(data["sandtotal_r"].iloc[i])
                var_grp.append(getSand(data["texture"].iloc[i]))
    if variable == "claytotal_r":
        for i in range(len(data)):
            if data["texture"].iloc[i] is None:
                var.append(data["claytotal_r"].iloc[i])
                var_grp.append(np.nan)
            else:
                var.append(data["claytotal_r"].iloc[i])
                var_grp.append(getClay(data["texture"].iloc[i]))
    if variable == "total_frag_volume":
        for i in range(len(data)):
            if data["total_frag_volume"].iloc[i] is None:
                var.append(np.nan)
                var_grp.append(np.nan)
            else:
                var.append(data["total_frag_volume"].iloc[i])
                var_grp.append(getCF(data["total_frag_volume"].iloc[i]))
    if variable == "CEC":
        for i in range(len(data)):
            if data["CEC"].iloc[i] is None:
                var.append(np.nan)
            else:
                var.append(data["CEC"].iloc[i])
    if variable == "pH":
        for i in range(len(data)):
            if data["pH"].iloc[i] is None:
                var.append(np.nan)
            else:
                var.append(data["pH"].iloc[i])
    if variable == "EC":
        for i in range(len(data)):
            if data["EC"].iloc[i] is None:
                var.append(np.nan)
            else:
                var.append(data["EC"].iloc[i])
                
    #Return empty fields when there is no depth data or the top depth is not 0
    if variable == "sandtotal_r" or variable == "claytotal_r" or variable == "total_frag_volume":
        if (pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any()):
            var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(152), columns = np.arange(2))
            var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
            return var_pct_intpl_final

        if data["hzdept_r"].iloc[0] != 0:
            var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(152), columns = np.arange(2))
            var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
            return var_pct_intpl_final

        MisHrz = 0
        for i in range(len(data["hzdept_r"])):
            if i == len(data["hzdept_r"]) - 1:
                break

            if (data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]):
                MisHrz = 1
            elif (data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]):
                data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]

        if MisHrz == 1:
            var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(152), columns = np.arange(2))
            var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
            return var_pct_intpl_final

        if len(data["hzdept_r"]) == 1:
            for i in range(int(data["hzdepb_r"].iloc[0]) - int(data["hzdept_r"].iloc[0])):
                var_pct_intpl.append(var[0])
                var_pct_intpl_grp.append(var_grp[0])
        else:
            for i in range(len(data["hzdepb_r"])):
                for j in range(int(data["hzdepb_r"].iloc[i]) - int(data["hzdept_r"].iloc[i])):
                    var_pct_intpl.append(var[i])
                    var_pct_intpl_grp.append(var_grp[i])

        var_pct_intpl_final = pd.DataFrame([var_pct_intpl, var_pct_intpl_grp])
        var_pct_intpl_final = var_pct_intpl_final.T
        var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']

        if len(var_pct_intpl_final.index) > 152:
            var_pct_intpl_final = var_pct_intpl_final.iloc[0:152]
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            Na_add = 152 - len(var_pct_intpl_final.index)
            pd_add = pd.DataFrame(np.nan, index = np.arange(Na_add), columns = np.arange(2))
            pd_add.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
            var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
    else:
        if (pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any()):
            var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(152), columns = np.arange(1))
            var_pct_intpl_final.columns = ['var_pct_intpl']
            return var_pct_intpl_final

        if data["hzdept_r"].iloc[0] != 0:
            var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(152), columns = np.arange(1))
            var_pct_intpl_final.columns = ['var_pct_intpl']
            return var_pct_intpl_final

        MisHrz = 0
        for i in range(len(data["hzdept_r"])):
            if i == len(data["hzdept_r"]) - 1:
                break

            if (data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]):
                MisHrz = 1
            elif (data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]):
                data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]

        if MisHrz == 1:
            var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(152), columns = np.arange(1))
            var_pct_intpl_final.columns = ['var_pct_intpl']
            return var_pct_intpl_final

        if len(data["hzdept_r"]) == 1:
            for i in range(int(data["hzdepb_r"].iloc[0]) - int(data["hzdept_r"].iloc[0])):
                var_pct_intpl.append(var[0])
        else:
            for i in range(len(data["hzdepb_r"])):
                for j in range(int(data["hzdepb_r"].iloc[i]) - int(data["hzdept_r"].iloc[i])):
                    var_pct_intpl.append(var[i])

        var_pct_intpl_final = pd.DataFrame([var_pct_intpl])
        var_pct_intpl_final = var_pct_intpl_final.T
        var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        var_pct_intpl_final.columns = ['var_pct_intpl']

        if len(var_pct_intpl_final.index) > 152:
            var_pct_intpl_final = var_pct_intpl_final.iloc[0:152]
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            Na_add = 152 - len(var_pct_intpl_final.index)
            pd_add = pd.DataFrame(np.nan, index = np.arange(Na_add), columns = np.arange(1))
            pd_add.columns = ['var_pct_intpl']
            var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
    if c_bot == True:
        if len(data["hzdept_r"]) == 1:
            c_very_bottom = data["hzdepb_r"].iloc[0]
        else:
            c_very_bottom = data["hzdepb_r"].values[-1]
        return c_very_bottom, var_pct_intpl_final
    else:
        return var_pct_intpl_final


def getProfile_SG(data, variable, c_bot=False):
        var = []
        var_grp = []
        var_pct_intpl = []
        var_pct_intpl_grp = []
        if variable == "sand":
            for i in range(len(data)):
                if data["texture"].iloc[i] is None:
                    var.append(data["sand"].iloc[i])
                    var_grp.append(np.nan)
                else:
                    var.append(data["sand"].iloc[i])
                    var_grp.append(getSand(data["texture"].iloc[i]))
        if variable == "clay":
            for i in range(len(data)):
                if data["texture"].iloc[i] is None:
                    var.append(data["clay"].iloc[i])
                    var_grp.append(np.nan)
                else:
                    var.append(data["clay"].iloc[i])
                    var_grp.append(getClay(data["texture"].iloc[i]))
        if variable == "cfvo":
            for i in range(len(data)):
                if data["cfvo"].iloc[i] is None:
                    var.append(np.nan)
                    var_grp.append(np.nan)
                else:
                    var.append(data["cfvo"].iloc[i])
                    var_grp.append(getCF(data["cfvo"].iloc[i]))
        if variable == "cec":
            for i in range(len(data)):
                if data["cec"].iloc[i] is None:
                    var.append(np.nan)
                    var_grp.append(np.nan)
                else:
                    var.append(data["cec"].iloc[i])
                    var_grp.append(getCF(data["cec"].iloc[i]))
        if variable == "phh2o":
            for i in range(len(data)):
                if data["phh2o"].iloc[i] is None:
                    var.append(np.nan)
                    var_grp.append(np.nan)
                else:
                    var.append(data["phh2o"].iloc[i])
                    var_grp.append(getCF(data["phh2o"].iloc[i]))

        #Return empty fields when there is no depth data or the top depth is not 0
        if variable == "sand" or variable == "clay" or variable == "cfvo" or variable == "cec" or variable == "phh2o":
            if (pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any()):
                var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(200), columns = np.arange(2))
                var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
                return var_pct_intpl_final
    
            if data["hzdept_r"].iloc[0] != 0:
                var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(200), columns = np.arange(2))
                var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
                return var_pct_intpl_final
    
            MisHrz = 0
            for i in range(len(data["hzdept_r"])):
                if i == len(data["hzdept_r"]) - 1:
                    break
    
                if (data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]):
                    MisHrz = 1
                elif (data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]):
                    data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]
    
            if MisHrz == 1:
                var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(200), columns = np.arange(2))
                var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
                return var_pct_intpl_final
    
            if len(data["hzdept_r"]) == 1:
                for i in range(int(data["hzdepb_r"].iloc[0]) - int(data["hzdept_r"].iloc[0])):
                    var_pct_intpl.append(var[0])
                    var_pct_intpl_grp.append(var_grp[0])
            else:
                for i in range(len(data["hzdepb_r"])):
                    for j in range(int(data["hzdepb_r"].iloc[i]) - int(data["hzdept_r"].iloc[i])):
                        var_pct_intpl.append(var[i])
                        var_pct_intpl_grp.append(var_grp[i])
    
            var_pct_intpl_final = pd.DataFrame([var_pct_intpl, var_pct_intpl_grp])
            var_pct_intpl_final = var_pct_intpl_final.T
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
            var_pct_intpl_final.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
    
            if len(var_pct_intpl_final.index) > 200:
                var_pct_intpl_final = var_pct_intpl_final.iloc[0:200]
                var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
            else:
                Na_add = 100 - len(var_pct_intpl_final.index)
                pd_add = pd.DataFrame(np.nan, index = np.arange(Na_add), columns = np.arange(2))
                pd_add.columns = ['var_pct_intpl', 'var_pct_intpl_grp']
                var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
                var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            if (pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any()):
                var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(200), columns = np.arange(1))
                var_pct_intpl_final.columns = ['var_pct_intpl']
                return var_pct_intpl_final
    
            if data["hzdept_r"].iloc[0] != 0:
                var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(200), columns = np.arange(1))
                var_pct_intpl_final.columns = ['var_pct_intpl']
                return var_pct_intpl_final
    
            MisHrz = 0
            for i in range(len(data["hzdept_r"])):
                if i == len(data["hzdept_r"]) - 1:
                    break
    
                if (data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]):
                    MisHrz = 1
                elif (data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]):
                    data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]
    
            if MisHrz == 1:
                var_pct_intpl_final = pd.DataFrame(np.nan, index = np.arange(200), columns = np.arange(1))
                var_pct_intpl_final.columns = ['var_pct_intpl']
                return var_pct_intpl_final
    
            if len(data["hzdept_r"]) == 1:
                for i in range(int(data["hzdepb_r"].iloc[0]) - int(data["hzdept_r"].iloc[0])):
                    var_pct_intpl.append(var[0])
            else:
                for i in range(len(data["hzdepb_r"])):
                    for j in range(int(data["hzdepb_r"].iloc[i]) - int(data["hzdept_r"].iloc[i])):
                        var_pct_intpl.append(var[i])
    
            var_pct_intpl_final = pd.DataFrame([var_pct_intpl])
            var_pct_intpl_final = var_pct_intpl_final.T
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
            var_pct_intpl_final.columns = ['var_pct_intpl']
    
            if len(var_pct_intpl_final.index) > 200:
                var_pct_intpl_final = var_pct_intpl_final.iloc[0:200]
                var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
            else:
                Na_add = 100 - len(var_pct_intpl_final.index)
                pd_add = pd.DataFrame(np.nan, index = np.arange(Na_add), columns = np.arange(1))
                pd_add.columns = ['var_pct_intpl']
                var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
                var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        if c_bot == True:
            if len(data["hzdept_r"]) == 1:
                c_very_bottom = data["hzdepb_r"].iloc[0]
            else:
                c_very_bottom = data["hzdepb_r"].values[-1]
            return c_very_bottom, var_pct_intpl_final
        else:
            return var_pct_intpl_final


def drop_cokey_horz(df):
    """
    Function to drop duplicate rows of component horizon data when more than one instance of a component are duplicates.
    Function assumes that the dataframe contains: 
      (1) unique cokey identifier ('cokey')
      (2) generic compname identifier ('compname')
    Can handle dataframes that include a 'slope_r' column as well as those that do not.
    """ 
    drop_instances = []

    # Base columns to compare
    columns_to_compare = [
        'hzdept_r', 'hzdepb_r', 'sandtotal_r',
        'silttotal_r', 'claytotal_r', 'total_frag_volume', 'texture'
    ]
    
    # Check if 'slope_r' column exists, and if so, add it to the columns to compare
    if 'slope_r' in df.columns:
        columns_to_compare.append('slope_r')

    # Group by 'compname'
    for _, comp_group in df.groupby('compname', sort=False):
        
        # Group the component group by 'cokey'
        grouped_by_cokey = [group for _, group in comp_group.groupby('cokey', sort=False)]

        # Iterate over combinations of the component instances
        for j, group_j in enumerate(grouped_by_cokey):
            for k, group_k in enumerate(grouped_by_cokey):
                if j >= k:
                    continue

                # Check if the two groups are the same based on specified columns
                if group_j[columns_to_compare].reset_index(drop=True).equals(group_k[columns_to_compare].reset_index(drop=True)):
                    drop_instances.append(group_k['cokey'])

    # Drop duplicates and reset index
    if drop_instances:
        drop_instances = pd.concat(drop_instances).drop_duplicates(keep='first').reset_index(drop=True)
    else:
        drop_instances = None

    return drop_instances

def haversine(lon1, lat1, lon2, lat2):
    """
    Calculate the great circle distance between two points on the earth specified in decimal degrees.
    
    Args:
    - lon1, lat1: Longitude and latitude of the first point.
    - lon2, lat2: Longitude and latitude of the second point.

    Returns:
    - Distance in kilometers between the two points.
    """
    # Convert decimal degrees to radians
    lon1, lat1, lon2, lat2 = map(math.radians, [lon1, lat1, lon2, lat2])

    # Haversine formula
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    a = math.sin(dlat / 2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2)**2
    c = 2 * math.asin(math.sqrt(a))
    r = 6371  # Radius of the earth in kilometers. Use 3956 for miles.

    return c * r

def pt2polyDist(poly, point):
    """
    Calculate the distance between a point and the closest point on the exterior of a polygon.
    
    Args:
    - poly: A shapely Polygon object.
    - point: A shapely Point object.

    Returns:
    - Distance in meters between the point and the polygon.
    """
    pol_ext = LinearRing(poly.exterior.coords)
    d = pol_ext.project(point)
    p = pol_ext.interpolate(d)
    closest_point_coords = list(p.coords)[0]
    #dist_m = haversine(point.x, point.y, closest_point_coords[0], closest_point_coords[1]) * 1000  # Convert to meters
    dist_m = haversine(point.x, point.y, *closest_point_coords) * 1000
    return round(dist_m, 0)

def calculate_location_score(group, ExpCoeff):
    """
    Computes a location score based on the distance and the share of a given group of data.
    
    Parameters:
    - group (DataFrame): A group of data containing 'distance' and 'share' columns.
    - ExpCoeff (float): Exponential coefficient to adjust sensitivity of the score to distance values.
    
    Returns:
    - float: Calculated location score.
    
    The score is adjusted based on the provided exponential coefficient (ExpCoeff). The function provides 
    a way to compute a normalized score for locations, giving preference to locations with a closer distance 
    (smaller distance values) and higher share values.
    """
    
    # Parameter validation
    if not isinstance(group, pd.DataFrame) or 'distance' not in group or 'share' not in group:
        raise ValueError("Group should be a DataFrame containing 'distance' and 'share' columns.")
    if not isinstance(ExpCoeff, (int, float)):
        raise ValueError("ExpCoeff should be a numeric value.")
    
    # Calculate total share
    total_share = sum(group.share.values)
    
    # Zero distance check
    if group.distance.min() == 0:
        return 1.0 if total_share > 100.0 else total_share / 100
    
    # Calculate distance multiplier
    distance_multiplier = max(math.exp(ExpCoeff * group.distance.min()), 0.25)
    
    # Calculate and return the score
    return distance_multiplier if total_share > 100.0 else (total_share / 100) * distance_multiplier


def assign_max_distance_scores(group):
    """
    Assigns the maximum distance scores and minimum distance to the entire group.
    
    Parameters:
    - group: DataFrame group based on certain criteria.
    
    Returns:
    - Modified group with updated distance scores and minimum distance.
    """
    
    # Compute the required values once
    max_distance_score = group['distance_score'].max()
    max_distance_score_norm = group['distance_score_norm'].max()
    min_distance = group['distance'].min()
    
    # Use .loc for efficient modification
    group.loc[:, 'distance_score'] = max_distance_score
    group.loc[:, 'distance_score_norm'] = max_distance_score_norm
    group.loc[:, 'min_dist'] = min_distance
    
    return group


# Gower Distance Functions
def check_pairwise_arrays(X, Y, precomputed=False, dtype=None):
    """
    Check if the input arrays X and Y are valid for pairwise computations.
    
    Args:
    - X (array-like): First input array.
    - Y (array-like): Second input array.
    - precomputed (bool): Whether the distances are precomputed.
    - dtype (type, optional): Desired data type for input arrays.

    Returns:
    - tuple: Validated and possibly transformed versions of X and Y.
    """
    
    # Determine the appropriate dtype
    X, Y, dtype_float = pairwise._return_float_dtype(X, Y)
    
    # Use a consistent name for the estimator in error messages
    estimator = 'check_pairwise_arrays'
    
    # If dtype is not provided, use the determined float dtype
    if dtype is None:
        dtype = dtype_float

    # Validate the input arrays
    X = validation.check_array(X, accept_sparse='csr', dtype=dtype, estimator=estimator)
    if Y is X or Y is None:
        Y = X
    else:
        Y = validation.check_array(Y, accept_sparse='csr', dtype=dtype, estimator=estimator)

    # Check for valid shapes based on whether distances are precomputed
    if precomputed and X.shape[1] != Y.shape[0]:
        raise ValueError(f"Precomputed metric requires shape (n_queries, n_indexed). Got ({X.shape[0]}, {X.shape[1]}) for {Y.shape[0]} indexed.")
    elif X.shape[1] != Y.shape[1]:
        raise ValueError(f"Incompatible dimension for X and Y matrices: X.shape[1] == {X.shape[1]} while Y.shape[1] == {Y.shape[1]}")

    return X, Y

      
def gower_distances(X, Y=None, feature_weight=None, categorical_features=None):
    """
    Computes the gower distances between X and Y.
    Gower is a similarity measure for categorical, boolean, and numerical mixed data.

    Parameters:
    ----------
    X : array-like, or pd.DataFrame
        Shape (n_samples, n_features)
    Y : array-like, or pd.DataFrame, optional
        Shape (n_samples, n_features)
    feature_weight : array-like, optional
        Shape (n_features). According to the Gower formula, it's an attribute weight.
    categorical_features : array-like, optional
        Shape (n_features). Indicates whether a column is a categorical attribute.

    Returns:
    -------
    ndarray : Gower distances. Shape (n_samples, n_samples)
    """
    
    if issparse(X) or (Y is not None and issparse(Y)):
        raise TypeError("Sparse matrices are not supported for gower distance")
    
    # Ensure arrays are numpy arrays
    X = np.asarray(X)
    
    dtype = np.object if not np.issubdtype(X.dtype, np.number) or np.isnan(X.sum()) else type(np.zeros(1, X.dtype).flat[0])
    X, Y = check_pairwise_arrays(X, Y, dtype=dtype)
    
    n_rows, n_cols = X.shape
    
    if categorical_features is None:
        categorical_features = np.array([not np.issubdtype(type(val), np.number) for val in X[0, :]])
    else:
        categorical_features = np.array(categorical_features)
    
    if np.issubdtype(categorical_features.dtype, np.int):
        new_categorical_features = np.zeros(n_cols, dtype=bool)
        new_categorical_features[categorical_features] = True
        categorical_features = new_categorical_features
    
    # Split data into categorical and numeric
    X_cat = X[:, categorical_features]
    X_num = X[:, ~categorical_features]
    
    # Calculate ranges and max values for normalization
    max_of_numeric = np.nanmax(X_num, axis=0)
    ranges_of_numeric = np.where(max_of_numeric != 0, max_of_numeric - np.nanmin(X_num, axis=0), 1.0)
    
    # Normalize numeric data
    X_num /= max_of_numeric
    
    # Handle feature weights
    if feature_weight is None:
        feature_weight = np.ones(n_cols)
    
    feature_weight_cat = feature_weight[categorical_features]
    feature_weight_num = feature_weight[~categorical_features]
    
    Y_cat = X_cat if Y is None else Y[:, categorical_features]
    Y_num = X_num if Y is None else Y[:, ~categorical_features]
    Y_num /= max_of_numeric
    
    dm = np.zeros((n_rows, Y.shape[0]), dtype=np.float32)
    
    # Calculate pairwise gower distances
    for i in range(n_rows):
        start = i if Y is None else 0
        result = _gower_distance_row(X_cat[i, :], X_num[i, :], Y_cat[start:, :], Y_num[start:, :], 
                                     feature_weight_cat, feature_weight_num, feature_weight.sum(), 
                                     categorical_features, ranges_of_numeric, max_of_numeric)
        dm[i, start:] = result
        if Y is None:  # If Y is not provided, the matrix is symmetric
            dm[start:, i] = result
    
    return dm


def _gower_distance_row(xi_cat, xi_num, xj_cat, xj_num, feature_weight_cat, feature_weight_num, feature_weight_sum, ranges_of_numeric):
    """
    Compute the Gower distance between a single row and a set of rows.
    
    This function calculates the Gower distance between a single data point (xi)
    and a set of data points (xj). Both categorical and numerical features are 
    considered in the calculation.
    
    Parameters:
    - xi_cat: Categorical data for xi.
    - xi_num: Numerical data for xi.
    - xj_cat: Categorical data for xj.
    - xj_num: Numerical data for xj.
    - feature_weight_cat: Weights for categorical features.
    - feature_weight_num: Weights for numerical features.
    - feature_weight_sum: Sum of all feature weights.
    - ranges_of_numeric: Normalized ranges for numeric features.

    Returns:
    - Gower distance between xi and each row in xj.
    """
    
    # Calculate distance for categorical data
    sij_cat = np.where(xi_cat == xj_cat, 0, 1)
    sum_cat = np.sum(feature_weight_cat * sij_cat, axis=1)
    
    # Calculate distance for numerical data
    abs_delta = np.abs(xi_num - xj_num)
    sij_num = np.divide(abs_delta, ranges_of_numeric, out=np.zeros_like(abs_delta), where=ranges_of_numeric != 0)
    sum_num = np.sum(feature_weight_num * sij_num, axis=1)
    
    # Combine distances for categorical and numerical data
    sum_sij = (sum_cat + sum_num) / feature_weight_sum
    
    return sum_sij


def compute_text_comp(bedrock, p_sandpct_intpl, soilHorizon):
    """
    Computes a value based on the depth of bedrock and length of sand percentages.
    
    Args:
    - bedrock (int or None): Depth of bedrock.
    - p_sandpct_intpl (Series): Series of sand percentages.
    - soilHorizon (list): List of soil horizons.
    
    Returns:
    - int: A computed value based on lookup table and input parameters.
    """
    
    # Return 0 if all values in soilHorizon are None
    if all(x is None for x in soilHorizon):
        return 0
    
    len_sand = len(p_sandpct_intpl.dropna())
    
    # Lookup table for determining return values
    lookup = {
        None: {1: 3, 10: 8, 20: 15, 50: 23, 70: 30, 100: 37, float('inf'): 45},
        10: {1: 3, 10: 45, float('inf'): 0},
        20: {1: 3, 10: 8, 20: 45, float('inf'): 0},
        50: {1: 3, 10: 8, 20: 15, 50: 45, float('inf'): 0},
        70: {1: 3, 10: 8, 20: 15, 50: 25, 70: 45, float('inf'): 0},
        100: {1: 3, 10: 8, 20: 15, 50: 25, 70: 35, 100: 45, float('inf'): 0}
    }
    
    # Categorize bedrock depth
    if bedrock is None or bedrock > 100:
        bedrock = None
    elif bedrock <= 10:
        bedrock = 10
    elif bedrock <= 20:
        bedrock = 20
    elif bedrock <= 50:
        bedrock = 50
    elif bedrock <= 70:
        bedrock = 70
    else:
        bedrock = 100

    # Return appropriate value from lookup table based on len_sand
    for key in lookup[bedrock]:
        if len_sand <= key:
            return lookup[bedrock][key]

    return 0

# Generalized function to remove elements if "hzname" contains 'R' or 'r' (indicates bedrock)
def remove_bedrock(data):
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame(data)
    
    # Filter the DataFrame to exclude rows where "hzname" contains 'R' or 'r'
    df_filtered = df[~df['hzname'].str.contains('R|r', na=False)]
    
    # Convert the filtered DataFrame back to a list of dictionaries
    result = df_filtered.to_dict('records')
    return result
  
def compute_rf_comp(bedrock, p_cfg_intpl, rfvDepth):
    if all(x is None for x in rfvDepth):
        return 0
    
    len_cfg = len(p_cfg_intpl.dropna())
    
    lookup = {
        None: {1: 3, 10: 6, 20: 10, 50: 16, 70: 22, 100: 26, float('inf'): 30},
        10: {1: 3, 10: 30, float('inf'): 0},
        20: {1: 3, 10: 6, 20: 30, float('inf'): 0},
        50: {1: 3, 10: 6, 20: 10, 50: 30, float('inf'): 0},
        70: {1: 3, 10: 6, 20: 10, 50: 15, 70: 30, float('inf'): 0},
        100: {1: 3, 10: 6, 20: 10, 50: 15, 70: 20, 100: 30, float('inf'): 0}
    }
    
    if bedrock is None or bedrock > 100:
        bedrock = None
    elif bedrock <= 10:
        bedrock = 10
    elif bedrock <= 20:
        bedrock = 20
    elif bedrock <= 50:
        bedrock = 50
    elif bedrock <= 70:
        bedrock = 70
    else:
        bedrock = 100

    for key in lookup[bedrock]:
        if len_cfg <= key:
            return lookup[bedrock][key]

    return 0

def compute_crack_comp(cracks):
    return 5 if cracks is not None else 0

def compute_lab_comp(cr_df):
    return 20 if not cr_df.dropna().empty else 0

def compute_data_completeness(bedrock, p_sandpct_intpl, soilHorizon, p_cfg_intpl, rfvDepth, cracks, cr_df):
    text_comp = compute_text_comp(bedrock, p_sandpct_intpl, soilHorizon)
    rf_comp = compute_rf_comp(bedrock, p_cfg_intpl, rfvDepth)
    crack_comp = compute_crack_comp(cracks)
    lab_comp = compute_lab_comp(cr_df)
    
    data_completeness = text_comp + rf_comp + crack_comp + lab_comp
    
    #Generate data completeness comment
    if text_comp < 45:
        text_comment = " soil texture,"
    else:
        text_comment = ""
    if rf_comp < 30:
        rf_comment = " soil rock fragments,"
    else:
        rf_comment = ""
    if lab_comp < 20:
        lab_comment = " soil color (20-50cm),"
    else:
        lab_comment = ""
    if crack_comp < 5:
        crack_comment = " soil cracking,"
    else:
        crack_comment = ""
    if data_completeness < 100:
        text_completeness = "To improve predictions, complete data entry for:" + crack_comment + text_comment + rf_comment + lab_comment + " and re-sync."
    else:
        text_completeness = "SoilID data entry for this site is complete."
        
    return data_completeness, text_completeness

def simulate_correlated_triangular(n, params, correlation_matrix):
    # Generate uncorrelated standard normal variables
    uncorrelated_normal = np.random.multivariate_normal(mean=np.zeros(len(params)), cov=np.eye(len(params)), size=n)
    
    # Cholesky decomposition of the correlation matrix
    L = np.linalg.cholesky(correlation_matrix)
    
    # Compute correlated variables using Cholesky decomposition
    correlated_normal = uncorrelated_normal @ L
    
    # Transform standard normal variables to match triangular marginal distributions
    samples = np.empty((n, len(params)))
    
    for i in range(len(params)):
        a = params[i][0]  # Lower limit of the triangle distribution
        b = params[i][1]  # Mode (peak) of the triangle distribution
        c = params[i][2]  # Upper limit of the triangle distribution
        
        normal_var = correlated_normal[:, i]
        u = norm.cdf(normal_var)  # Transform to uniform [0, 1] range
        
        for j in range(len(u)):
            u_i = u[j]
            if u_i <= (b - a) / (c - a):
                samples[j, i] = a + np.sqrt(u_i * (c - a) * (b - a))
            else:
                samples[j, i] = c - np.sqrt((1 - u_i) * (c - a) * (c - b))
    
    return samples
  
def trim_fraction(text):
    """
    Removes trailing ".0" from a given text string.
    
    Parameters:
    - text (str): Input string.
    
    Returns:
    - str: Text without trailing ".0".
    """
    return text.rstrip('.0') if text.endswith('.0') else text


def calculate_distance_score(row, ExpCoeff):
    """
    Calculate distance score based on the conditions provided (US).
    """
    if row['distance'] == 0:
        if row['comppct_r'] > 100:
            return 1
        else:
            return round(row['comppct_r'] / 100, 3)
    else:
        if row['comppct_r'] > 100:
            return round(max(0.25, math.exp(ExpCoeff * row['distance'])), 3)
        else:
            factor = max(0.25, math.exp(ExpCoeff * row['distance']))
            return round(row['comppct_r'] / 100 * factor, 3)

def extract_muhorzdata_STATSGO(mucompdata_pd):
    """
    Extracts muhorzdata based on the provided mucompdata_pd dataframe and the sda_return function.
    
    Args:
        mucompdata_pd (pd.DataFrame): Input dataframe with mucompdata data.
        
    External Functions:
        sda_return (function): Function to execute the database query.
    
    Returns:
        pd.DataFrame: Extracted and processed muhorzdata.
    """
    # Convert cokey values to ASCII and form a list
    cokey_list = [str(val).encode('ascii', 'ignore').decode('utf-8') for val in mucompdata_pd['cokey'].tolist()]
    
    # Form the muhorzdata query
    muhorzdataQry = 'SELECT cokey, chorizon.chkey, hzdept_r, hzdepb_r, hzname, sandtotal_r, silttotal_r, claytotal_r, cec7_r, ecec_r, ph1to1h2o_r, ec_r,lep_r, chfrags.fragvol_r FROM chorizon LEFT OUTER JOIN chfrags ON chfrags.chkey = chorizon.chkey WHERE cokey IN ({})'.format(','.join(cokey_list))
    
    # Execute the query
    muhorzdata_out = sda_return(propQry=muhorzdataQry)
    if mucompdata_out is None:
        return "Soil ID not available in this area"
    else:
        muhorzdata = muhorzdata_out['Table']
        
        # Convert the list of lists to a DataFrame
        muhorzdata_pd = pd.DataFrame(muhorzdata[1:], columns=muhorzdata[0])
        
        # Additional processing steps
        ...  # (rest of the code remains unchanged)
        
        return muhorzdata_pd

def extract_statsgo_mucompdata(lon, lat):
    """
    Extracts and processes STATSGO data for the given longitude and latitude.
    
    Args:
        lon (float): Longitude of the point of interest.
        lat (float): Latitude of the point of interest.
        
    External Functions:
        sda_return (function): Function to execute the sda query.
        trim_fraction (function): Function to trim fractions.
    
    Returns:
        pd.DataFrame: Processed mucompdata.
    """
    # Create LPKS point
    point = Point(lon, lat)
    point.crs = {'init': 'epsg:4326'}
    
    # Create a bounding box to clip STATSGO data around the point
    s_buff = gpd.GeoSeries(point).buffer(0.1)  # 0.1 deg buffer around point = ~11km
    box = shapely.geometry.box(*s_buff.total_bounds)
    
    # Load STATSGO mukey data
    statsgo_mukey = gpd.read_file('%s/gsmsoilmu_a_us.shp' % current_app.config['DATA_BACKEND'], bbox=box.bounds, mode='r', driver="ESRI Shapefile")
    
    # Filter out mapunits with duplicate geometries
    mu_geo = statsgo_mukey[['MUKEY', 'geometry']].drop_duplicates(subset=['geometry'])
    
    # Calculate distances and intersection flags for each mapunit
    distances = [pt2polyDist(geom, point) for geom in mu_geo['geometry']]
    intersects = [point.intersects(geom) for geom in mu_geo['geometry']]
    
    # Create a DataFrame for distances and intersections
    mu_id_dist = pd.DataFrame({
        'MUKEY': mu_geo['MUKEY'].values,
        'distance': distances,
        'pt_intersect': intersects
    })
    
    # Update distance to 0 for intersecting mapunits
    mu_id_dist.loc[mu_id_dist.pt_intersect, 'distance'] = 0
    mu_id_dist['distance'] = mu_id_dist.groupby(['MUKEY'])['distance'].transform(min)
    mukey_dist_final = mu_id_dist.drop_duplicates(subset=['MUKEY']).sort_values(by='distance').head(2)
    
    # Build the mucompdata query
    mukey_list = mukey_dist_final['MUKEY'].tolist()
    mucompdataQry = f"SELECT component.mukey, component.cokey, component.compname, component.comppct_r, component.compkind, component.majcompflag, component.slope_r, component.elev_r, component.nirrcapcl, component.nirrcapscl, component.nirrcapunit, component.irrcapcl, component.irrcapscl, component.irrcapunit, component.taxorder, component.taxsubgrp FROM component WHERE mukey IN ({','.join(map(str, mukey_list))})"
    mucompdata_out = sda_return(propQry=mucompdataQry)
    
    # Process the mucompdata results
    if mucompdata_out is None:
        return("Soil ID not available in this area")
    else:
        mucompdata = mucompdata_out['Table']
        mucompdata_pd = pd.DataFrame(mucompdata[1:], columns=mucompdata[0])
        mucompdata_pd = pd.merge(mucompdata_pd, mukey_dist_final, on='mukey').sort_values(['distance', 'cokey'])
        mucompdata_pd.replace("NULL", np.nan, inplace=True)
        mucompdata_pd[['slope_r', 'elev_r', 'distance']] = mucompdata_pd[['slope_r', 'elev_r', 'distance']].astype(float)
        mucompdata_pd.nirrcapcl = mucompdata_pd.nirrcapcl.apply(trim_fraction)
        mucompdata_pd.irrcapcl = mucompdata_pd.irrcapcl.apply(trim_fraction)
        
        # Subset dataframe to extract only components within 5000m -- STATSGO
        mucompdata_pd = mucompdata_pd[mucompdata_pd['distance'] <= 5000]
    
    return mucompdata_pd
  
  
def fill_missing_comppct_r(mucompdata_pd):
    """
    Fills missing or zero values in the 'comppct_r' column based on the difference 
    between the sum of all components in the map unit subtracted from 100.

    Parameters:
    - mucompdata_pd (pd.DataFrame): DataFrame containing soil data.

    Returns:
    - pd.DataFrame: Updated DataFrame with processed 'comppct_r' values.
    """

    mukeys_with_missing_data = mucompdata_pd.query('comppct_r==0 | comppct_r.isnull()')['mukey'].drop_duplicates()

    if not mukeys_with_missing_data.empty:
        subset_data = mucompdata_pd[mucompdata_pd['mukey'].isin(mukeys_with_missing_data)]

        aggregated_data = subset_data.groupby('mukey').agg(
            total_pct=('comppct_r', 'sum'),
            total_rows=('comppct_r', 'size'),
            non_null_count=('comppct_r', 'count'),
            zero_count=('comppct_r', lambda x: (x == 0).sum())
        )

        aggregated_data['missing_data_count'] = aggregated_data['total_rows'] - aggregated_data['non_null_count']
        aggregated_data['percent_diff'] = 100 - aggregated_data['total_pct']
        aggregated_data['value_to_fill'] = aggregated_data['percent_diff'] / (aggregated_data['missing_data_count'] + aggregated_data['zero_count'])

        for idx, row in aggregated_data.iterrows():
            condition = (mucompdata_pd['mukey'] == idx) & (mucompdata_pd['comppct_r'].isin([0, np.nan]))
            mucompdata_pd.loc[condition, 'comppct_r'] = row['value_to_fill']

    # Clean up the dataframe
    mucompdata_pd = mucompdata_pd.drop_duplicates().sort_values(by='distance').reset_index(drop=True)

    # Handle minor components that are either 0 or NaN
    mucompdata_pd['comppct_r'].replace({np.nan: 1, 0: 1}, inplace=True)
    mucompdata_pd['comppct_r'] = mucompdata_pd['comppct_r'].astype(int)

    return mucompdata_pd


def process_distance_scores(mucompdata_pd, ExpCoeff):
    """
    Process distance scores and perform group-wise aggregations.

    Parameters:
    - mucompdata_pd (pd.DataFrame): DataFrame containing soil data.
    
    External Functions:
    - calculate_distance_score (function): A function to calculate distance scores.

    Returns:
    - pd.DataFrame: Updated DataFrame with processed distance scores and aggregations.
    """

    # Calculate distance score for each group
    mucompdata_pd['distance_score'] = mucompdata_pd.apply(lambda row: calculate_distance_score(row, ExpCoeff), axis=1)

    # Group by cokey and mukey and aggregate required values
    grouped_data = mucompdata_pd.groupby(['cokey', 'mukey']).agg(
        distance_score=('distance_score', 'sum'),
        comppct=('comppct_r', 'sum'),
        minDistance=('distance', 'min')
    ).reset_index()
    
    # Calculate conditional probabilities
    total_distance_score = grouped_data['distance_score'].sum()
    grouped_data['cond_prob'] = grouped_data['distance_score'] / total_distance_score
    
    # Merge dataframes on 'cokey'
    mucompdata_pd = mucompdata_pd.merge(grouped_data[['cokey', 'cond_prob']], on='cokey', how='left')
    
    # Additional processing
    mucompdata_pd = mucompdata_pd.sort_values('distance_score', ascending=False)
    # Normalize distance score
    mucompdata_pd['distance_score_norm'] = (mucompdata_pd['distance_score'] / mucompdata_pd['distance_score'].max())
    mucompdata_pd = mucompdata_pd[~mucompdata_pd['compkind'].str.contains("Miscellaneous area")]
    
    mucompdata_pd = mucompdata_pd.reset_index(drop=True)
    
    # Create a list of component groups
    mucompdata_comp_grps = [g for _, g in mucompdata_pd.groupby(['compname'], sort=False)]
    mucompdata_comp_grps = mucompdata_comp_grps[:min(12, len(mucompdata_comp_grps))]
    
    # Assign max within-group location-based score to all members of the group
    for group in mucompdata_comp_grps:
        group['distance_score'] = group['distance_score'].max()
        group['distance_score_norm'] = group['distance_score_norm'].max()
        group = group.sort_values('distance').reset_index(drop=True)
        group['min_dist'] = group['distance'].iloc[0]
    
    # Concatenate the list of dataframes
    mucompdata_pd = pd.concat(mucompdata_comp_grps).reset_index(drop=True)

    return mucompdata_pd



#####################################################################################################
#                                       Soil Color Functions                                        #
#####################################################################################################

def pedon_color(lab_Color, horizonDepth):
    lbIdx = len(horizonDepth) - 1
    pedon_top = [0] + [horizonDepth[i] for i in range(lbIdx)]
    
    pedon_bottom = horizonDepth
    pedon_l, pedon_a, pedon_b = lab_Color.iloc[:, 0], lab_Color.iloc[:, 1], lab_Color.iloc[:, 2]

    # Check for None values
    if None in (pedon_top, pedon_bottom, pedon_l, pedon_a, pedon_b):
        return np.nan

    if pedon_top[0] != 0:
        return np.nan

    # Check for missing horizons
    pedon_MisHrz = any(pedon_top[i + 1] != pedon_bottom[i] for i in range(len(pedon_top) - 1))
    if pedon_MisHrz:
        return np.nan

    pedon_l_intpl, pedon_a_intpl, pedon_b_intpl = [], [], []

    if len(pedon_top) == 1:
        pedon_l_intpl = [pedon_l[0]] * (pedon_bottom[0] - pedon_top[0])
        pedon_a_intpl = [pedon_a[0]] * (pedon_bottom[0] - pedon_top[0])
        pedon_b_intpl = [pedon_b[0]] * (pedon_bottom[0] - pedon_top[0])
    else:
        for i in range(len(pedon_bottom)):
            pedon_l_intpl.extend([pedon_l[i]] * (pedon_bottom[i] - pedon_top[i]))
            pedon_a_intpl.extend([pedon_a[i]] * (pedon_bottom[i] - pedon_top[i]))
            pedon_b_intpl.extend([pedon_b[i]] * (pedon_bottom[i] - pedon_top[i]))

    pedon_len = len(pedon_l_intpl)
    if pedon_len >= 37:
        pedon_l_mean = np.mean(pedon_l_intpl[31:37])
        pedon_a_mean = np.mean(pedon_a_intpl[31:37])
        pedon_b_mean = np.mean(pedon_b_intpl[31:37])
    elif 31 <= pedon_len < 37:
        pedon_l_mean = np.mean(pedon_l_intpl[31:])
        pedon_a_mean = np.mean(pedon_a_intpl[31:])
        pedon_b_mean = np.mean(pedon_b_intpl[31:])
    else:
        pedon_l_mean, pedon_a_mean, pedon_b_mean = np.nan, np.nan, np.nan

    if np.isnan(pedon_l_mean) or np.isnan(pedon_a_mean) or np.isnan(pedon_b_mean):
        return np.nan
    else:
        return [pedon_l_mean, pedon_a_mean, pedon_b_mean]


def lab2munsell(color_ref, LAB_ref, LAB):
    """
    Converts LAB color values to Munsell notation using the closest match from a reference dataframe.
    
    Parameters:
    - color_ref (pd.DataFrame): Reference dataframe with LAB and Munsell values.
    - LAB_ref (list): Reference LAB values.
    - LAB (list): LAB values to be converted.
    
    Returns:
    - str: Munsell color notation.
    """
    idx = pd.DataFrame(euclidean_distances([LAB], LAB_ref)).idxmin(axis=1).iloc[0]
    munsell_color = f"{color_ref.at[idx, 'hue']} {int(color_ref.at[idx, 'value'])}/{int(color_ref.at[idx, 'chroma'])}"
    return munsell_color

def munsell2rgb(color_ref, munsell_ref, munsell):
    """
    Converts Munsell notation to RGB values using a reference dataframe.
    
    Parameters:
    - color_ref (pd.DataFrame): Reference dataframe with Munsell and RGB values.
    - munsell_ref (pd.DataFrame): Reference dataframe with Munsell values.
    - munsell (list): Munsell values [hue, value, chroma] to be converted.
    
    Returns:
    - list: RGB values.
    """
    idx = munsell_ref.query(f'hue == "{munsell[0]}" & value == {int(munsell[1])} & chroma == {int(munsell[2])}').index[0]
    return [color_ref.at[idx, col] for col in ['r', 'g', 'b']]

def rgb2lab(color_ref, rgb_ref, rgb):
    """
    Convert RGB values to LAB color values using a reference dataframe.
    
    Parameters:
    - color_ref (pd.DataFrame): Reference dataframe containing RGB and LAB values.
    - rgb_ref (list): Reference RGB values.
    - rgb (list): RGB values to be converted.
    
    Returns:
    - list: LAB values.
    """
    idx = pd.DataFrame(euclidean_distances([rgb], rgb_ref)).idxmin(axis=1).iloc[0]
    return [color_ref.at[idx, col] for col in ['L', 'A', 'B']]

def getProfileLAB(data_osd, color_ref):
    """
    The function processes the given data_osd DataFrame and computes LAB values for soil profiles.
    """
    LAB_ref  =  color_ref[['L', 'A', 'B']]
    rgb_ref  =  color_ref[['r', 'g', 'b']]
    munsell_ref = color_ref[['hue', 'value', 'chroma']]
    
    # Convert the specific columns to numeric
    data_osd[['top', 'bottom', 'r', 'g', 'b']] = data_osd[['top', 'bottom', 'r', 'g', 'b']].apply(pd.to_numeric)
    
    def validate_data(data):
        """
        Validates the data based on given conditions.
        """
        if (data.top.isnull().any() or data.bottom.isnull().any()):
            return False
        if data.r.isnull().all() or data.g.isnull().all() or data.b.isnull().all():
            return False
        if data.top.iloc[0] != 0:
            return False
        return True

    def correct_depth_discrepancies(data):
        """
        Corrects depth discrepancies by adding layers when needed.
        """
        layers_to_add = []
        for i in range(len(data.top) - 1):
            if (data.top.iloc[i + 1] > data.bottom.iloc[i]):
                layer_add = pd.DataFrame({
                    'top': data.bottom.iloc[i], 
                    'bottom': data.top.iloc[i + 1], 
                    'r': np.nan, 
                    'g': np.nan, 
                    'b': np.nan
                }, index=[i+0.5])
                layers_to_add.append(layer_add)
                
        if layers_to_add:
            data = pd.concat([data] + layers_to_add).sort_index().reset_index(drop=True)
            
        return data
    
    def convert_rgb_to_lab(row):
        """
        Converts RGB values to LAB.
        """
        if pd.isnull(row['r']) or pd.isnull(row['g']) or pd.isnull(row['b']):
            return np.nan, np.nan, np.nan
      
        LAB = rgb2lab(color_ref, rgb_ref, [row['r'], row['g'], row['b']])
        return(LAB)
        # Check the structure of LAB and extract values accordingly
        try:
            if isinstance(LAB, list) and all(isinstance(x, np.ndarray) for x in LAB):
                # Assuming LAB is a list of arrays
                return LAB[0], LAB[1], LAB[2]
            elif isinstance(LAB, np.ndarray) and LAB.ndim == 1:
                # Assuming LAB is a 1D array
                return LAB[0], LAB[1], LAB[2]
            else:
                # Handle other cases or unknown structures
                print(f"Unexpected structure of LAB: {LAB}")
                return np.nan, np.nan, np.nan
        except IndexError as e:
            # Handle indexing errors
            print(f"Indexing error in LAB: {e}")
            return np.nan, np.nan, np.nan

    
    if not validate_data(data_osd):
        return pd.DataFrame(np.nan, index=np.arange(120), columns=['L', 'A', 'B'])
    
    data_osd = correct_depth_discrepancies(data_osd)
    
    data_osd['L'], data_osd['A'], data_osd['B'] = zip(*data_osd.apply(convert_rgb_to_lab, axis=1))

    l_intpl, a_intpl, b_intpl = [], [], []
    
    for index, row in data_osd.iterrows():
        l_intpl.extend([row['L']] * (int(row['bottom']) - int(row['top'])))
        a_intpl.extend([row['A']] * (int(row['bottom']) - int(row['top'])))
        b_intpl.extend([row['B']] * (int(row['bottom']) - int(row['top'])))
       
    lab_intpl = pd.DataFrame({'L': l_intpl, 'A': a_intpl, 'B': b_intpl}).head(120)
    return(lab_intpl) 
    if len(lab_intpl) < 120:
        lab_intpl = lab_intpl.append(pd.DataFrame(np.nan, index=np.arange(120 - len(lab_intpl)), columns=['L', 'A', 'B']))
        
    return lab_intpl


def calculate_deltaE2000(LAB1, LAB2):
    """
    Computes the Delta E 2000 value between two LAB color values.

    Args:
        LAB1 (list): First LAB color value.
        LAB2 (list): Second LAB color value.

    Returns:
        float: Delta E 2000 value.
    """

    L1star, a1star, b1star = LAB1
    L2star, a2star, b2star = LAB2

    C1abstar = math.sqrt(a1star ** 2 + b1star ** 2)
    C2abstar = math.sqrt(a2star ** 2 + b2star ** 2)
    Cabstarbar = (C1abstar + C2abstar) / 2.0

    G = 0.5 * (1.0 - math.sqrt(Cabstarbar ** 7 / (Cabstarbar ** 7 + 25 ** 7)))

    a1prim = (1.0 + G) * a1star
    a2prim = (1.0 + G) * a2star

    C1prim = math.sqrt(a1prim ** 2 + b1star ** 2)
    C2prim = math.sqrt(a2prim ** 2 + b2star ** 2)

    h1prim = math.atan2(b1star, a1prim) if (b1star != 0 or a1prim != 0) else 0
    h2prim = math.atan2(b2star, a2prim) if (b2star != 0 or a2prim != 0) else 0

    deltaLprim = L2star - L1star
    deltaCprim = C2prim - C1prim

    if (C1prim * C2prim) == 0:
        deltahprim = 0
    elif abs(h2prim - h1prim) <= 180:
        deltahprim = h2prim - h1prim
    elif abs(h2prim - h1prim) > 180 and (h2prim - h1prim) < 360:
        deltahprim = h2prim - h1prim - 360.0
    else:
        deltahprim = h2prim - h1prim + 360.0

    deltaHprim = 2 * math.sqrt(C1prim * C2prim) * math.sin(deltahprim / 2.0)

    Lprimbar = (L1star + L2star) / 2.0
    Cprimbar = (C1prim + C2prim) / 2.0

    if abs(h1prim - h2prim) <= 180:
        hprimbar = (h1prim + h2prim) / 2.0
    elif abs(h1prim - h2prim) > 180 and (h1prim + h2prim) < 360:
        hprimbar = (h1prim + h2prim + 360) / 2.0
    else:
        hprimbar = (h1prim + h2prim - 360) / 2.0

    T = 1.0 - 0.17 * math.cos(hprimbar - 30.0) + 0.24 * math.cos(2.0 * hprimbar) + 0.32 * math.cos(3.0 * hprimbar + 6.0) - 0.20 * math.cos(4.0 * hprimbar - 63.0)

    deltatheta = 30.0 * math.exp(-(math.pow((hprimbar - 275.0) / 25.0, 2.0)))
    RC = 2.0 * math.sqrt(Cprimbar ** 7 / (Cprimbar ** 7 + 25 ** 7))
    SL = 1.0 + (0.015 * (Lprimbar - 50.0)**2) / math.sqrt(20.0 + (Lprimbar - 50.0)**2)
    SC = 1.0 + 0.045 * Cprimbar
    SH = 1.0 + 0.015 * Cprimbar * T
    RT = -math.sin(2.0 * deltatheta) * RC

    kL, kC, kH = 1.0, 1.0, 1.0
    term1 = (deltaLprim / (kL * SL))**2
    term2 = (deltaCprim / (kC * SC))**2
    term3 = (deltaHprim / (kH * SH))**2
    term4 = RT * (deltaCprim / (kC * SC)) * (deltaHprim / (kH * SH))

    return math.sqrt(term1 + term2 + term3 + term4)
  
# Not currently implemented for US SoilID  
def interpolate_color_values(top, bottom, color_values):
    """
    Interpolates the color values based on depth.

    Args:
        top (pd.Series): Top depths.
        bottom (pd.Series): Bottom depths.
        color_values (pd.Series): Corresponding color values.

    Returns:
        np.array: Interpolated color values for each depth.
    """

    if top[0] != 0:
        raise ValueError("The top depth must start from 0.")
    
    MisHrz = any([top[i + 1] != bottom[i] for i in range(len(top) - 1)])
    if MisHrz:
        raise ValueError("There is a mismatch in horizon depths.")

    color_intpl = []
    for i, color_val in enumerate(color_values):
        color_intpl.extend([color_val] * (bottom[i] - top[i]))

    return np.array(color_intpl)
  

# Not currently implemented for US SoilID
def getColor_deltaE2000_OSD_pedon(data_osd, data_pedon):
    """
    Calculate the Delta E 2000 value between averaged LAB values of OSD and pedon samples.
    
    The function interpolates the color values based on depth for both OSD and pedon samples.
    It then computes the average LAB color value for the 31-37 cm depth range.
    Finally, it calculates the Delta E 2000 value between the two averaged LAB values.

    Args:
        data_osd (object): Contains depth and RGB data for the OSD sample.
            - top: List of top depths.
            - bottom: List of bottom depths.
            - r, g, b: Lists of RGB color values corresponding to each depth.
        
        data_pedon (object): Contains depth and LAB data for the pedon sample.
            - [0]: List of bottom depths.
            - [1]: DataFrame with LAB color values corresponding to each depth.

    Returns:
        float: Delta E 2000 value between the averaged LAB values of OSD and pedon.
        Returns NaN if the data is not adequate for calculations.
    """
    # Extract relevant data for OSD and pedon
    top, bottom, r, g, b = data_osd.top, data_osd.bottom, data_osd.r, data_osd.g, data_osd.b
    ref_top, ref_bottom, ref_lab = [0] + data_pedon[0][:-1], data_pedon[0], data_pedon[1]

    # Convert RGB values to LAB for OSD
    osd_colors_rgb = interpolate_color_values(top, bottom, list(zip(r, g, b)))
    osd_colors_lab = [color.rgb2lab([[color_val]])[0][0] for color_val in osd_colors_rgb]
    
    # Calculate average LAB for OSD at 31-37 cm depth
    osd_avg_lab = np.mean(osd_colors_lab[31:37], axis=0) if len(osd_colors_lab) > 31 else np.nan
    if np.isnan(osd_avg_lab).any():
        return np.nan

    # Convert depth values to LAB for pedon
    pedon_colors_lab = interpolate_color_values(ref_top, ref_bottom, list(zip(ref_lab.iloc[:, 0], ref_lab.iloc[:, 1], ref_lab.iloc[:, 2])))
    
    # Calculate average LAB for pedon at 31-37 cm depth
    pedon_avg_lab = np.mean(pedon_colors_lab[31:37], axis=0) if len(pedon_colors_lab) > 31 else np.nan
    if np.isnan(pedon_avg_lab).any():
        return np.nan

    # Return the Delta E 2000 value between the averaged LAB values
    return calculate_deltaE2000(osd_avg_lab, pedon_avg_lab)


def simulate_correlated_triangular(n, params, correlation_matrix):
    """
    Simulate correlated triangular distributed variables.
    
    Parameters:
    - n: Number of samples.
    - params: List of tuples, where each tuple contains three parameters (a, b, c) for the triangular distribution.
    - correlation_matrix: 2D numpy array representing the desired correlations between the variables.

    Returns:
    - samples: 2D numpy array with n rows and as many columns as there are sets of parameters in params.
    """
    
    # Generate uncorrelated standard normal variables
    uncorrelated_normal = np.random.normal(size=(n, len(params)))
    
    # Cholesky decomposition of the correlation matrix
    L = cholesky(correlation_matrix)
    
    # Compute correlated variables using Cholesky decomposition
    correlated_normal = uncorrelated_normal @ L
    
    # Transform standard normal variables to match triangular marginal distributions
    samples = np.zeros((n, len(params)))
    
    for i, (a, b, c) in enumerate(params):
        normal_var = correlated_normal[:, i]
        u = norm.cdf(normal_var)  # Transform to uniform [0, 1] range
        
        # Transform the uniform values into triangularly distributed values
        condition = u <= (b - a) / (c - a)
        samples[condition, i] = a + np.sqrt(u[condition] * (c - a) * (b - a))
        samples[~condition, i] = c - np.sqrt((1 - u[~condition]) * (c - a) * (c - b))
    
    return samples

def extract_values(obj, key):
    """
    Pull all values of the specified key from a nested dictionary or list.
    
    Parameters:
    - obj (dict or list): The nested dictionary or list to search.
    - key: The key to look for.
    
    Returns:
    - list: A list of values associated with the specified key.
    """
    
    arr = []
    
    def extract(obj, key):
        if isinstance(obj, dict):
            if key in obj:
                arr.append(obj[key])
            for k, v in obj.items():
                if isinstance(v, (dict, list)):
                    extract(v, key)
        elif isinstance(obj, list):
            for item in obj:
                extract(item, key)
    
    extract(obj, key)
    return arr

  
def getSG_descriptions(WRB_Comp_List):
    try:
        conn = getDataStore_Connection()
        
        # Execute a SQL query and return the results
        def execute_query(query, params):
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

        # First SQL query
        sql1 = 'SELECT lu.WRB_1984_Full FROM wrb2006_to_fao90 AS lu WHERE lu.WRB_2006_Full IN %s'
        names = execute_query(sql1, (tuple(WRB_Comp_List),))
        WRB_Comp_List = [item for t in list(names) for item in t]

        # Second SQL query
        sql2 = '''SELECT WRB_tax, Description_en, Management_en, Description_es, Management_es, 
                  Description_ks, Management_ks, Description_fr, Management_fr 
                  FROM wrb_fao90_desc WHERE WRB_tax IN %s'''
        results = execute_query(sql2, (tuple(WRB_Comp_List),))

        # Convert results to DataFrame
        data = pd.DataFrame(results, columns=['WRB_tax', 'Description_en', 'Management_en', 'Description_es', 
                                              'Management_es', 'Description_ks', 'Management_ks', 
                                              'Description_fr', 'Management_fr'])
        return data

    except Exception as err:
        print(err)
        return None

    finally:
        conn.close()

# Functions for AWC simulation

def simulate_correlated_triangular(n, params, correlation_matrix):
    """
    Simulate correlated triangular distributed variables.
    
    Parameters:
    - n: Number of samples.
    - params: List of tuples, where each tuple contains three parameters (a, b, c) for the triangular distribution.
    - correlation_matrix: 2D numpy array representing the desired correlations between the variables.

    Returns:
    - samples: 2D numpy array with n rows and as many columns as there are sets of parameters in params.
    """
    
    # Generate uncorrelated standard normal variables
    uncorrelated_normal = np.random.normal(size=(n, len(params)))
    
    # Cholesky decomposition of the correlation matrix
    L = cholesky(correlation_matrix)
    
    # Compute correlated variables using Cholesky decomposition
    correlated_normal = uncorrelated_normal @ L
    
    # Transform standard normal variables to match triangular marginal distributions
    samples = np.zeros((n, len(params)))
    
    for i, (a, b, c) in enumerate(params):
        normal_var = correlated_normal[:, i]
        u = norm.cdf(normal_var)  # Transform to uniform [0, 1] range
        
        # Transform the uniform values into triangularly distributed values
        condition = u <= (b - a) / (c - a)
        samples[condition, i] = a + np.sqrt(u[condition] * (c - a) * (b - a))
        samples[~condition, i] = c - np.sqrt((1 - u[~condition]) * (c - a) * (c - b))
    
    return samples

def acomp(X, parts=None, total=1):
    if parts is None:
        parts = list(range(X.shape[1]))

    parts = list(set(parts))
    
    if isinstance(X, pd.DataFrame):
        Xn = X.iloc[:, parts].to_numpy()
    else:
        Xn = X[:, parts]

    Xn /= Xn.sum(axis=1)[:, np.newaxis] / total

    return gsi_simshape(Xn, X)

def gsi_simshape(x, oldx):
    if oldx.ndim >= 2:
        return x
    return x.flatten() if oldx.ndim == 0 else x.reshape(-1)


# Temporary function to infill missing data. TODO: create loopup table with average values for l-r-h by series
def infill_soil_data(df):
    # Group by 'cokey'
    grouped = df.groupby('cokey')
    
    # Filtering groups
    def filter_group(group):
        # Step 2: Check for missing 'r' values where 'hzdepb_r' <= 50
        if (group['hzdepb_r'] <= 50).any() and group[['sandtotal_r', 'claytotal_r', 'silttotal_r']].isnull().any().any():
            return False  # Exclude group
        return True  # Include group
    
    # Apply the filter to the groups
    filtered_groups = grouped.filter(filter_group)
    
    # Step 3: Replace missing '_l' and '_h' values with corresponding '_r' values +/- 8
    for col in ['sandtotal', 'claytotal', 'silttotal']:
        filtered_groups[col + '_l'].fillna(filtered_groups[col + '_r'] - 8, inplace=True)
        filtered_groups[col + '_h'].fillna(filtered_groups[col + '_r'] + 8, inplace=True)
    
    # Step 4 and 5: Replace missing 'dbovendry_l' and 'dbovendry_h' with 'dbovendry_r' +/- 0.01
    filtered_groups['dbovendry_l'].fillna(filtered_groups['dbovendry_r'] - 0.01, inplace=True)
    filtered_groups['dbovendry_h'].fillna(filtered_groups['dbovendry_r'] + 0.01, inplace=True)
    
    # Step 6 and 7: Replace missing 'wthirdbar_l' and 'wthirdbar_h' with 'wthirdbar_r' +/- 1
    filtered_groups['wthirdbar_l'].fillna(filtered_groups['wthirdbar_r'] - 1, inplace=True)
    filtered_groups['wthirdbar_h'].fillna(filtered_groups['wthirdbar_r'] + 1, inplace=True)
    
    # Step 8 and 9: Replace missing 'wfifteenbar_l' and 'wfifteenbar_h' with 'wfifteenbar_r' +/- 0.6
    filtered_groups['wfifteenbar_l'].fillna(filtered_groups['wfifteenbar_r'] - 0.6, inplace=True)
    filtered_groups['wfifteenbar_h'].fillna(filtered_groups['wfifteenbar_r'] + 0.6, inplace=True)
    
    return filtered_groups
  
def aggregate_data_vi(data, max_depth, sd=2):
    """
    Aggregate data by specific depth ranges and compute the mean of each range for each column.

    Args:
        data (pd.DataFrame): The DataFrame containing the data with index as depth.
        max_depth (float): The maximum depth to consider for aggregation.
        sd (int): The number of decimal places to round the aggregated data.

    Returns:
        pd.DataFrame: A DataFrame with aggregated data for each column within specified depth ranges.
    """
    if not max_depth or np.isnan(max_depth):
        return pd.DataFrame(columns=['hzdept_r', 'hzdepb_r', 'Data'])

    # Define the depth ranges
    depth_ranges = [(0, 30), (30, 100)]
    # Initialize the result list
    results = []

    # Iterate over each column in the dataframe
    for column in data.columns:
        column_results = []
        for top, bottom in depth_ranges:
            if max_depth <= top:
                column_results.append([top, bottom, np.nan])
            else:
                mask = (data.index >= top) & (data.index <= min(bottom, max_depth))
                data_subset = data.loc[mask, column]
                if not data_subset.empty:
                    result = round(data_subset.mean(), sd)
                    column_results.append([top, min(bottom, max_depth), result])
                else:
                    column_results.append([top, min(bottom, max_depth), np.nan])
        # Append the results for the current column to the overall results list
        results.append(pd.DataFrame(column_results, columns=['hzdept_r', 'hzdepb_r', f'Aggregated Data ({column})']))

    # Concatenate the results for each column into a single dataframe
    result_df = pd.concat(results, axis=1)
    
    # If there are multiple columns, remove the repeated 'Top Depth' and 'Bottom Depth' columns
    if len(data.columns) > 1:
        result_df = result_df.loc[:,~result_df.columns.duplicated()]
        
    return result_df
  
  
# ROSETTA Simulation
# Define a function to perform Rosetta simulation
def rosetta_simulate(data):
    # Create a SoilData instance
    soildata = SoilData.from_array(data)

    # Create a RosettaSoil instance
    rs = RosettaSoil()

    # Perform Rosetta simulation
    rosetta_sim = rs.predict(soildata)
    
    return rosetta_sim

# Define a function to simulate data for each aoi
def mukey_sim_rosseta(aoi_data, cor_matrix):
    mukey_sim_list = []
    
    for i in range(len(aoi_data)):
        data = mukey_data[i]
        sim_data = multi_sim_hydro(data, cor_matrix)  # Assuming you have a function multi_sim_hydro
        combined_data = data + sim_data.tolist()
        mukey_sim_list.append(combined_data)
    
    return mukey_sim_list
 
