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

###################################################################################################
#                                       Helper Functions.                                         #
###################################################################################################
# Standard libraries
import logging
import math
import re

# Third-party libraries
import geopandas as gpd
import numpy as np
import pandas as pd
from numpy.linalg import cholesky
from osgeo import ogr
from rosetta import SoilData, rosetta
from scipy.interpolate import UnivariateSpline
from scipy.sparse import issparse
from scipy.stats import entropy, norm
from shapely.geometry import Point
from sklearn.impute import SimpleImputer
from sklearn.metrics import pairwise
from sklearn.utils import validation

# local libraries
import soil_id.config

from .services import sda_return

###################################################################################################
#                                       Utility Functions                                         #
###################################################################################################


def getSand(field: str) -> float:
    """
    Given a soil texture name (e.g. "loam", "sand", "clay"), return the approximate
    sand percentage based on a predefined lookup table.

    Args:
        field (str): The soil texture name (e.g. "sandy loam", "clay", etc.).
                     Case-insensitive. May be None or an empty string.

    Returns:
        float: The sand percentage for that texture, or numpy.nan if the texture
               is unrecognized or field is None.
    """
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
        "clay": 22.5,
    }

    if field is None:
        return np.nan

    # Convert field to lowercase, then do the lookup in the dictionary
    return sand_percentages.get(field.lower(), np.nan)


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
        "clay": 70.0,
    }

    return clay_percentages.get(field.lower() if field else None, np.nan)


def silt_calc(row):
    sand = row["sand"]
    clay = row["clay"]
    silt = 100 - (sand + clay)
    return silt


def getTexture(row=None, sand=None, silt=None, clay=None):
    """
    Classify soil texture based on sand, silt, and clay proportions.

    Parameters:
    - row (dict): Dictionary-like object with 'sandtotal_r', 'silttotal_r', 'claytotal_r' keys.
    - sand (float): Percentage of sand.
    - silt (float): Percentage of silt.
    - clay (float): Percentage of clay.

    Returns:
    - str: Soil texture classification.
    """

    # Handle missing inputs: if not provided individually, try to get from row.
    if sand is None or silt is None or clay is None:
        if row is not None:
            sand = row.get("sandtotal_r", np.nan)
            silt = row.get("silttotal_r", np.nan)
            clay = row.get("claytotal_r", np.nan)

    # Replace any NaN with 0 for the calculation.
    sand = np.nan_to_num(sand, nan=0)
    silt = np.nan_to_num(silt, nan=0)
    clay = np.nan_to_num(clay, nan=0)

    # Calculate derived values.
    silt_clay = silt + 1.5 * clay
    silt_2x_clay = silt + 2.0 * clay

    # Define conditions and corresponding texture classifications.
    conditions = [
        silt_clay < 15,
        (silt_clay >= 15) & (silt_clay < 30),
        (((7 <= clay) & (clay <= 20)) & (sand > 52))
        | ((clay < 7) & (silt < 50) & (silt_2x_clay >= 30)),
        (7 <= clay) & (clay <= 27) & (28 <= silt) & (silt < 50) & (sand <= 52),
        (silt >= 50) & (((12 <= clay) & (clay < 27)) | ((silt < 80) & (clay < 12))),
        (silt >= 80) & (clay < 12),
        (20 <= clay) & (clay < 35) & (silt < 28) & (sand > 45),
        (27 <= clay) & (clay < 40) & (sand <= 45) & (sand > 20),
        (clay >= 35) & (sand >= 45),
        (clay >= 40) & (silt >= 40) & (sand <= 45),
    ]

    choices = [
        "Sand",
        "Loamy sand",
        "Sandy loam",
        "Loam",
        "Silt loam",
        "Silt",
        "Sandy clay loam",
        "Clay loam",
        "Sandy clay",
        "Clay",
    ]

    # Compute the texture classification.
    result = np.select(conditions, choices, default="Unknown")

    # Ensure that a plain Python string is returned.
    if isinstance(result, np.ndarray):
        try:
            result = result.item()  # Extract single element from an array of size 1.
        except Exception:
            result = str(result)

    return result


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
        ">60%": 80,
    }

    return cf_to_value.get(cf, np.nan)


def getCF_class(row, cf=None):
    cf = cf if cf is not None else row.get("rfv")
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
    else:
        return np.nan


def getOSDCF(cf):
    frag_vol_35 = [
        "gravelly",
        "fine gravelly",
        "medium gravelly",
        "coarse gravelly",
        "cobbly",
        "stony",
        "bouldery",
        "channery",
        "flaggy",
    ]
    frag_vol_35_list = [re.compile(pattern) for pattern in frag_vol_35]

    frag_vol_60 = [
        "very gravelly",
        "very cobbly",
        "very stony",
        "extremely bouldery",
        "very channery",
        "very flaggy",
    ]
    frag_vol_60_list = [re.compile(pattern) for pattern in frag_vol_60]

    frag_vol_90 = [
        "extremely gravelly",
        "extremely cobbly",
        "extremely stony",
        "extremely channery",
        "extremely flaggy",
    ]
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
        (1, ["sl1"], [1]),
        (11, ["sl1", "sl2"], [1, bottom]),
        (21, ["sl1", "sl2", "sl3"], [1, 10, bottom]),
        (51, ["sl1", "sl2", "sl3", "sl4"], [1, 10, 20, bottom]),
        (71, ["sl1", "sl2", "sl3", "sl4", "sl5"], [1, 10, 20, 50, bottom]),
        (101, ["sl1", "sl2", "sl3", "sl4", "sl5", "sl6"], [1, 10, 20, 50, 70, bottom]),
        (
            120,
            ["sl1", "sl2", "sl3", "sl4", "sl5", "sl6", "sl7"],
            [1, 10, 20, 50, 70, 100, bottom],
        ),
        (
            float("inf"),
            ["sl1", "sl2", "sl3", "sl4", "sl5", "sl6", "sl7"],
            [1, 10, 20, 50, 70, 100, 120],
        ),
    ]

    # Process data for each range
    for max_val, indices, depths in ranges:
        if bottom < max_val:
            data_d = [round(data[:d].mean(), sd) for d in depths]
            data_d = pd.Series(data_d, index=indices)
            d_lyrs = pd.Series(depths, index=indices)
            return (data_d, d_lyrs) if depth else data_d


def aggregate_data(data, bottom_depths, sd=2):
    """
    Aggregate values of a given data series into segments defined by depth intervals
    and return their mean values, rounded to a specified number of decimal places.

    This function partitions the data index into depth intervals determined by
    the provided `bottom_depths`. Each interval starts at one of the `top_depths`
    (which is derived from `bottom_depths` by shifting them up by one) and ends
    at the corresponding bottom depth. For each interval, the function extracts
    the data values within that depth range and computes their mean. The results
    are rounded to `sd` decimal places and returned as a Series, where each element
    corresponds to the aggregated mean for the respective depth interval.

    Parameters
    ----------
    data : pandas.Series
        A series indexed by depth (or a comparable numeric index), which
        will be aggregated over the defined depth intervals.
    bottom_depths : list or array-like
        A list or array of numerical values representing the lower boundaries
        of depth intervals. The top boundary for the first interval is implicitly 0.
    sd : int, optional, default 2
        Number of decimal places to which the computed mean values will be rounded.

    Returns
    -------
    pandas.Series
        A series of aggregated mean values for each depth interval, indexed by
        the order of the intervals. If any interval does not contain data values,
        NaN is returned for that interval.
    """
    if not bottom_depths or np.isnan(bottom_depths[0]):
        return pd.Series([np.nan])

    top_depths = [0] + bottom_depths[:-1]

    results = []

    for top, bottom in zip(top_depths, bottom_depths):
        mask = (data.index >= top) & (data.index < bottom)
        data_subset = data[mask]
        if not data_subset.empty:
            result = round(data_subset.mean(), sd)
            results.append(result)
        else:
            results.append(np.nan)

    return pd.Series(results)


def getProfile(data, variable):
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

    # Return empty fields when there is no depth data or the top depth is not 0
    if variable == "sandtotal_r" or variable == "claytotal_r" or variable == "total_frag_volume":
        if pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any():
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(2))
            var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
            return var_pct_intpl_final

        if data["hzdept_r"].iloc[0] != 0:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(2))
            var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
            return var_pct_intpl_final

        MisHrz = 0
        for i in range(len(data["hzdept_r"])):
            if i == len(data["hzdept_r"]) - 1:
                break

            if data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]:
                MisHrz = 1
            elif data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]:
                data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]

        if MisHrz == 1:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(2))
            var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
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
        var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]

        if len(var_pct_intpl_final.index) > 200:
            var_pct_intpl_final = var_pct_intpl_final.iloc[0:200]
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            Na_add = 200 - len(var_pct_intpl_final.index)
            pd_add = pd.DataFrame(np.nan, index=np.arange(Na_add), columns=np.arange(2))
            pd_add.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
            var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
    else:
        if pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any():
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(1))
            var_pct_intpl_final.columns = ["var_pct_intpl"]
            return var_pct_intpl_final

        if data["hzdept_r"].iloc[0] != 0:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(1))
            var_pct_intpl_final.columns = ["var_pct_intpl"]
            return var_pct_intpl_final

        MisHrz = 0
        for i in range(len(data["hzdept_r"])):
            if i == len(data["hzdept_r"]) - 1:
                break

            if data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]:
                MisHrz = 1
            elif data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]:
                data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]

        if MisHrz == 1:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(1))
            var_pct_intpl_final.columns = ["var_pct_intpl"]
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
        var_pct_intpl_final.columns = ["var_pct_intpl"]

        if len(var_pct_intpl_final.index) > 200:
            var_pct_intpl_final = var_pct_intpl_final.iloc[0:200]
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            Na_add = 200 - len(var_pct_intpl_final.index)
            pd_add = pd.DataFrame(np.nan, index=np.arange(Na_add), columns=np.arange(1))
            pd_add.columns = ["var_pct_intpl"]
            var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
    return var_pct_intpl_final


def max_comp_depth(data):
    if len(data["hzdept_r"]) == 1:
        c_very_bottom = data["hzdepb_r"].iloc[0]
    else:
        c_very_bottom = data["hzdepb_r"].values[-1]

    # Check if c_very_bottom is greater than 200 and assign 200 if true
    if c_very_bottom > 200:
        c_very_bottom = 200
    return c_very_bottom


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

    # Return empty fields when there is no depth data or the top depth is not 0
    if (
        variable == "sand"
        or variable == "clay"
        or variable == "cfvo"
        or variable == "cec"
        or variable == "phh2o"
    ):
        if pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any():
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(2))
            var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
            return var_pct_intpl_final

        if data["hzdept_r"].iloc[0] != 0:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(2))
            var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
            return var_pct_intpl_final

        MisHrz = 0
        for i in range(len(data["hzdept_r"])):
            if i == len(data["hzdept_r"]) - 1:
                break

            if data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]:
                MisHrz = 1
            elif data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]:
                data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]

        if MisHrz == 1:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(2))
            var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
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
        var_pct_intpl_final.columns = ["var_pct_intpl", "var_pct_intpl_grp"]

        if len(var_pct_intpl_final.index) > 200:
            var_pct_intpl_final = var_pct_intpl_final.iloc[0:200]
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            Na_add = 100 - len(var_pct_intpl_final.index)
            pd_add = pd.DataFrame(np.nan, index=np.arange(Na_add), columns=np.arange(2))
            pd_add.columns = ["var_pct_intpl", "var_pct_intpl_grp"]
            var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
    else:
        if pd.isnull(data["hzdept_r"]).any() or pd.isnull(data["hzdepb_r"]).any():
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(1))
            var_pct_intpl_final.columns = ["var_pct_intpl"]
            return var_pct_intpl_final

        if data["hzdept_r"].iloc[0] != 0:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(1))
            var_pct_intpl_final.columns = ["var_pct_intpl"]
            return var_pct_intpl_final

        MisHrz = 0
        for i in range(len(data["hzdept_r"])):
            if i == len(data["hzdept_r"]) - 1:
                break

            if data["hzdept_r"].iloc[i + 1] > data["hzdepb_r"].iloc[i]:
                MisHrz = 1
            elif data["hzdept_r"].iloc[i + 1] < data["hzdepb_r"].iloc[i]:
                data["hzdept_r"].iloc[i + 1] == data["hzdepb_r"].iloc[i]

        if MisHrz == 1:
            var_pct_intpl_final = pd.DataFrame(np.nan, index=np.arange(200), columns=np.arange(1))
            var_pct_intpl_final.columns = ["var_pct_intpl"]
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
        var_pct_intpl_final.columns = ["var_pct_intpl"]

        if len(var_pct_intpl_final.index) > 200:
            var_pct_intpl_final = var_pct_intpl_final.iloc[0:200]
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
        else:
            Na_add = 100 - len(var_pct_intpl_final.index)
            pd_add = pd.DataFrame(np.nan, index=np.arange(Na_add), columns=np.arange(1))
            pd_add.columns = ["var_pct_intpl"]
            var_pct_intpl_final = pd.concat([var_pct_intpl_final, pd_add], axis=0)
            var_pct_intpl_final = var_pct_intpl_final.reset_index(drop=True)
    if c_bot:
        if len(data["hzdept_r"]) == 1:
            c_very_bottom = data["hzdepb_r"].iloc[0]
        else:
            c_very_bottom = data["hzdepb_r"].values[-1]
        return c_very_bottom, var_pct_intpl_final
    else:
        return var_pct_intpl_final


def drop_cokey_horz(df):
    """
    Function to drop duplicate rows of component horizon data when more than one instance of a
    component are duplicates.

    Function assumes that the dataframe contains:
      (1) unique cokey identifier ('cokey')
      (2) generic compname identifier ('compname')
    Can handle dataframes that include a 'slope_r' column as well as those that do not.
    """
    columns_to_compare = [
        "hzdept_r",
        "hzdepb_r",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "total_frag_volume",
        "texture",
    ]
    if "slope_r" in df.columns:
        columns_to_compare.append("slope_r")

    # Generate a hashable representation of each cokey's horizon data
    df["_group_key"] = df[columns_to_compare].astype(str).agg("|".join, axis=1)
    df["_cokey_grouped"] = df.groupby("cokey")["_group_key"].transform(lambda x: "|".join(x))

    drop_instances = []

    # Group by compname
    for _, comp_group in df.groupby("compname", sort=False):
        # Get unique cokeys and their data signature within this compname group
        cokey_map = comp_group.groupby("cokey")["_cokey_grouped"].first()

        # Find duplicates
        duplicated = cokey_map.duplicated(keep="first")
        drop_instances.extend(cokey_map[duplicated].index.tolist())

    return pd.Series(drop_instances, name="cokey_to_drop")


def calculate_location_score(group, ExpCoeff):
    """
    Computes a location score based on the distance and the share of a given group of data.

    Parameters:
    - group (DataFrame): A group of data containing 'distance' and 'share' columns.
    - ExpCoeff (float): Exponential coefficient to adjust sensitivity of the score
                        to distance values.

    Returns:
    - float: Calculated location score.

    The score is adjusted based on the provided exponential coefficient (ExpCoeff). The
    function provides a way to compute a score for locations, giving preference
    to locations with a closer distance (smaller distance values) and higher aerial
    coverage within a mapunit.
    """

    # Parameter validation
    if not isinstance(group, pd.DataFrame) or "distance" not in group or "share" not in group:
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
    max_distance_score = group["distance_score"].max()
    min_distance = group["distance"].min()

    # Use .loc for efficient modification
    group.loc[:, "distance_score"] = max_distance_score
    group.loc[:, "min_dist"] = min_distance

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
    estimator = "check_pairwise_arrays"

    # If dtype is not provided, use the determined float dtype
    if dtype is None:
        dtype = dtype_float

    # impute missing values
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")  # You can change the strategy
    X = imputer.fit_transform(X)

    # Validate the input arrays
    X = validation.check_array(X, accept_sparse="csr", dtype=dtype, estimator=estimator)
    if Y is X or Y is None:
        Y = X
    else:
        Y = validation.check_array(Y, accept_sparse="csr", dtype=dtype, estimator=estimator)

    # Check for valid shapes based on whether distances are precomputed
    if precomputed and X.shape[1] != Y.shape[0]:
        raise ValueError(
            "Precomputed metric requires shape (n_queries, n_indexed)."
            f"Got ({X.shape[0]}, {X.shape[1]}) for {Y.shape[0]} indexed."
        )
    elif X.shape[1] != Y.shape[1]:
        raise ValueError(
            "Incompatible dimension for X and Y matrices:"
            f"X.shape[1] == {X.shape[1]} while Y.shape[1] == {Y.shape[1]}"
        )

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

    dtype = (
        object
        if not np.issubdtype(X.dtype, np.number) or np.isnan(X.sum())
        else type(np.zeros(1, X.dtype).flat[0])
    )
    X, Y = check_pairwise_arrays(X, Y, dtype=dtype)

    n_rows, n_cols = X.shape

    if categorical_features is None:
        categorical_features = np.array(
            [not np.issubdtype(type(val), np.number) for val in X[0, :]]
        )
    else:
        categorical_features = np.array(categorical_features)

    if np.issubdtype(categorical_features.dtype, int):
        new_categorical_features = np.zeros(n_cols, dtype=bool)
        new_categorical_features[categorical_features] = True
        categorical_features = new_categorical_features

    # Split data into categorical and numeric
    X_cat = X[:, categorical_features]
    X_num = X[:, ~categorical_features]

    # Calculate ranges and max values for normalization
    max_of_numeric = np.nanmax(X_num, axis=0)
    min_of_numeric = np.nanmin(X_num, axis=0)
    ranges_of_numeric = max_of_numeric - min_of_numeric
    ranges_of_numeric[ranges_of_numeric == 0] = 1

    # Normalize numeric data
    X_num = (X_num - min_of_numeric) / ranges_of_numeric

    # Handle feature weights
    if feature_weight is None:
        feature_weight = np.ones(X_num.shape[1])

    feature_weight_num = feature_weight[~categorical_features]

    # Conditional processing for Y
    if Y is not None:
        Y_cat = Y[:, categorical_features]
        Y_num = Y[:, ~categorical_features]
        # Normalize numeric data safely for Y_num
        Y_num = np.where(
            ranges_of_numeric != 0, (Y_num - min_of_numeric) / ranges_of_numeric, Y_num
        )
    else:
        Y_cat = X_cat.copy()
        Y_num = X_num.copy()

    # Ensure feature_weight_cat is defined
    feature_weight_cat = feature_weight[categorical_features]

    # # Handle feature weights
    # if feature_weight is None:
    #     feature_weight = np.ones(n_cols)

    # feature_weight_cat = feature_weight[categorical_features]
    # feature_weight_num = feature_weight[~categorical_features]

    # Y_cat = X_cat if Y is None else Y[:, categorical_features]
    # Y_num = X_num if Y is None else Y[:, ~categorical_features]
    # Y_num /= max_of_numeric

    dm = np.zeros((n_rows, Y.shape[0]), dtype=np.float32)

    # Calculate pairwise gower distances
    for i in range(X_num.shape[0]):
        start = i if Y is None else 0
        result = _gower_distance_row(
            X_cat[i, :],
            X_num[i, :],
            Y_cat[start:, :],
            Y_num[start:, :],
            feature_weight_cat,
            feature_weight_num,
            feature_weight.sum(),
            categorical_features,
            ranges_of_numeric,
            max_of_numeric,
        )
        dm[i, start:] = result
        if Y is None:  # If Y is not provided, the matrix is symmetric
            dm[start:, i] = result

    return dm
    # # Calculate pairwise gower distances
    # for i in range(n_rows):
    #     start = i if Y is None else 0
    #     result = _gower_distance_row(
    #         X_cat[i, :],
    #         X_num[i, :],
    #         Y_cat[start:, :],
    #         Y_num[start:, :],
    #         feature_weight_cat,
    #         feature_weight_num,
    #         feature_weight.sum(),
    #         categorical_features,
    #         ranges_of_numeric,
    #         max_of_numeric,
    #     )
    #     dm[i, start:] = result
    #     if Y is None:  # If Y is not provided, the matrix is symmetric
    #         dm[start:, i] = result

    # return dm


def _gower_distance_row(
    xi_cat,
    xi_num,
    xj_cat,
    xj_num,
    feature_weight_cat,
    feature_weight_num,
    feature_weight_sum,
    categorical_features,
    ranges_of_numeric,
    max_of_numeric,
):
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
    sij_num = np.divide(
        abs_delta,
        ranges_of_numeric,
        out=np.zeros_like(abs_delta),
        where=ranges_of_numeric != 0,
    )
    sum_num = np.sum(feature_weight_num * sij_num, axis=1)

    # Combine distances for categorical and numerical data
    sum_sij = (sum_cat + sum_num) / feature_weight_sum

    return sum_sij


def compute_site_similarity(
    p_slope, mucompdata, slices, additional_columns=None, feature_weight=None
):
    """
    Compute gower distances for site similarity based on the provided feature weights.

    Parameters:
    - p_slope: DataFrame containing sample_pedon information.
    - mucompdata: DataFrame containing component data.
    - slices: DataFrame containing slices of soil.
    - additional_columns: List of additional columns to consider.
    - feature_weight: Array of weights for features.

    Returns:
    - D_site: Gower distances array for site similarity.
    """
    # Combine pedon slope data with component data
    site_vars = pd.concat([p_slope, mucompdata[["compname", "slope_r", "elev_r"]]], axis=0)

    # If additional columns are specified, merge them
    if additional_columns:
        site_vars = pd.merge(slices, site_vars, on="compname", how="left")
        site_mat = site_vars[additional_columns]
    else:
        site_mat = site_vars[["slope_r", "elev_r"]]

    site_mat = site_mat.set_index(slices.compname.values)

    # Compute the gower distances
    D_site = gower_distances(site_mat, feature_weight=feature_weight)

    # Replace NaN values with the maximum value in the array
    D_site = np.where(np.isnan(D_site), np.nanmax(D_site), D_site)

    return D_site


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
        None: {1: 3, 10: 8, 20: 15, 50: 23, 70: 30, 100: 37, float("inf"): 45},
        10: {1: 3, 10: 45, float("inf"): 0},
        20: {1: 3, 10: 8, 20: 45, float("inf"): 0},
        50: {1: 3, 10: 8, 20: 15, 50: 45, float("inf"): 0},
        70: {1: 3, 10: 8, 20: 15, 50: 25, 70: 45, float("inf"): 0},
        100: {1: 3, 10: 8, 20: 15, 50: 25, 70: 35, 100: 45, float("inf"): 0},
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


def compute_rf_comp(bedrock, p_cfg_intpl, rfvDepth):
    if all(x is None for x in rfvDepth):
        return 0

    len_cfg = len(p_cfg_intpl.dropna())

    lookup = {
        None: {1: 3, 10: 6, 20: 10, 50: 16, 70: 22, 100: 26, float("inf"): 30},
        10: {1: 3, 10: 30, float("inf"): 0},
        20: {1: 3, 10: 6, 20: 30, float("inf"): 0},
        50: {1: 3, 10: 6, 20: 10, 50: 30, float("inf"): 0},
        70: {1: 3, 10: 6, 20: 10, 50: 15, 70: 30, float("inf"): 0},
        100: {1: 3, 10: 6, 20: 10, 50: 15, 70: 20, 100: 30, float("inf"): 0},
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


def compute_data_completeness(
    bedrock, p_sandpct_intpl, soilHorizon, p_cfg_intpl, rfvDepth, cracks, cr_df
):
    text_comp = compute_text_comp(bedrock, p_sandpct_intpl, soilHorizon)
    rf_comp = compute_rf_comp(bedrock, p_cfg_intpl, rfvDepth)
    crack_comp = compute_crack_comp(cracks)
    lab_comp = compute_lab_comp(cr_df)

    data_completeness = text_comp + rf_comp + crack_comp + lab_comp

    # Generate data completeness comment
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
        text_completeness = (
            "To improve predictions, complete data entry for:"
            + crack_comment
            + text_comment
            + rf_comment
            + lab_comment
            + " and re-sync."
        )
    else:
        text_completeness = "SoilID data entry for this site is complete."

    return data_completeness, text_completeness


def trim_fraction(text):
    """
    Removes trailing ".0" from a given text string.

    Parameters:
    - text (str): Input string.

    Returns:
    - str: Text without trailing ".0".
    """
    if text is None:
        return None  # or return None, depending on what you need
    return text.rstrip(".0") if text.endswith(".0") else text


def calculate_distance_score(row, ExpCoeff):
    """
    Calculate distance score based on the conditions provided (US).
    """
    if row["distance"] == 0:
        if row["comppct_r"] > 100:
            return 1
        else:
            return round(row["comppct_r"] / 100, 3)
    else:
        if row["comppct_r"] > 100:
            return round(max(0.25, math.exp(ExpCoeff * row["distance"])), 3)
        else:
            factor = max(0.25, math.exp(ExpCoeff * row["distance"]))
            return round(row["comppct_r"] / 100 * factor, 3)


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
    cokey_list = [
        str(val).encode("ascii", "ignore").decode("utf-8")
        for val in mucompdata_pd["cokey"].tolist()
    ]

    # Form the muhorzdata query
    muhorzdataQry = f"""SELECT cokey, chorizon.chkey, hzdept_r, hzdepb_r, hzname,
                        sandtotal_l, sandtotal_r, sandtotal_h, silttotal_l,
                        silttotal_r, silttotal_h, claytotal_l, claytotal_r,
                        claytotal_h, dbovendry_l, dbovendry_r, dbovendry_h,
                        wthirdbar_l, wthirdbar_r, wthirdbar_h, wfifteenbar_l,
                        wfifteenbar_r, wfifteenbar_h, cec7_r, ecec_r, ph1to1h2o_r,
                        ec_r, lep_r, chfrags.fragvol_r
                        FROM chorizon
                        LEFT OUTER JOIN chfrags ON chfrags.chkey = chorizon.chkey
                        WHERE cokey IN ({",".join(cokey_list)})"""

    # Execute the query
    muhorzdata_out = sda_return(propQry=muhorzdataQry)
    if muhorzdata_out is None:
        return "Soil ID not available in this area"
    else:
        muhorzdata = muhorzdata_out["Table"].iloc[0]

        # Convert the list of lists to a DataFrame
        muhorzdata_pd = pd.DataFrame(muhorzdata[1:], columns=muhorzdata[0])

        return muhorzdata_pd


def calculate_distances_and_intersections(mu_geo, point):
    """
    Calculate distances and intersections of geometries to a point.

    Args:
        mu_geo (GeoDataFrame): GeoDataFrame containing mapunit geometries.
        point (Point): Shapely Point object for the location of interest.

    Returns:
        DataFrame: Contains mapunit keys, distances, and intersection flags.
    """

    # Ensure the point is wrapped in a GeoDataFrame and projected correctly
    point_utm, epsg_code = convert_geometry_to_utm(point)

    # Ensure the point is a GeoDataFrame (for compatibility)
    if not isinstance(point_utm, gpd.GeoDataFrame):
        point_utm = gpd.GeoDataFrame(geometry=[point_utm], crs=epsg_code)

    # Transform the GeoDataFrame to the same UTM CRS
    mu_geo_utm = mu_geo.to_crs(epsg_code)

    # Reset index for clean operations
    mu_geo_utm = mu_geo_utm.reset_index(drop=True)
    point_utm = point_utm.reset_index(drop=True)

    # Extract the single geometry for the point
    point_geometry = point_utm.geometry.iloc[0]

    # Calculate distances and intersections
    distances = mu_geo_utm["geom"].distance(point_geometry)
    intersects = mu_geo_utm["geom"].intersects(point_geometry)
    return pd.DataFrame(
        {"hwsd2": mu_geo_utm["hwsd2"], "dist_meters": distances, "pt_intersect": intersects}
    )


def load_statsgo_data(box):
    """
    Load STATSGO data within a given bounding box.

    Args:
        box (BaseGeometry): Bounding box to filter the data.

    Returns:
        GeoDataFrame: Filtered STATSGO data.
    """
    try:
        return gpd.read_file(
            soil_id.config.STATSGO_PATH, bbox=box.bounds, mode="r", driver="ESRI Shapefile"
        )
    except Exception as e:
        logging.error(f"STATSGO error: {e}")
        return None


def convert_geometry_to_utm(geometry, src_crs="EPSG:4326"):
    """
    Converts a geometry from a geographic CRS to a UTM CRS based on its centroid.

    Parameters:
    - geometry (shapely.geometry or geopandas.GeoDataFrame): The input geometry.
    - src_crs (str, optional): The source CRS, default is "EPSG:4326".

    Returns:
    - tuple: (GeoDataFrame with transformed geometry, target CRS string)
    """
    # Ensure the geometry is a GeoDataFrame
    if isinstance(geometry, Point):
        geometry = gpd.GeoDataFrame(geometry=[geometry], columns=["geometry"], crs=src_crs)
    elif isinstance(geometry, gpd.GeoSeries):
        geometry = geometry.to_frame(name="geometry")

    if "geometry" not in geometry:
        raise ValueError("Input does not contain a valid geometry column.")

    # Ensure the CRS is correctly assigned before transformation
    if geometry.crs is None:
        geometry.set_crs(src_crs, inplace=True)

    # Convert to source CRS (to ensure consistency)
    geometry = geometry.to_crs(src_crs)

    # Compute the centroid
    centroid = geometry.geometry.centroid.iloc[0]  # Extract the first centroid
    if centroid.is_empty:
        raise ValueError("Geometry centroid is empty. Check input geometries.")

    lon, lat = centroid.x, centroid.y

    # Determine UTM zone
    utm_zone = int((lon + 180) / 6) + 1
    hemisphere = "north" if lat >= 0 else "south"
    epsg_code = f"326{utm_zone:02d}" if hemisphere == "north" else f"327{utm_zone:02d}"
    target_crs = f"EPSG:{epsg_code}"

    # Transform to UTM CRS
    geometry_utm = geometry.to_crs(target_crs)

    return geometry_utm, target_crs


# def get_utm_crs(lat, lon):
#     # Define an area of interest that is just the point (or a small buffer around it)
#     aoi = pyproj.aoi.AreaOfInterest(
#         west_lon_degree=lon, south_lat_degree=lat,
#         east_lon_degree=lon, north_lat_degree=lat
#     )
#     # Query for UTM CRS candidates covering the area of interest
#     utm_crs_list = query_utm_crs_info(datum_name="WGS84", area_of_interest=aoi)
#     if not utm_crs_list:
#         raise ValueError("No UTM CRS found for the specified location.")

#     # Select the first matching CRS
#     crs = CRS.from_epsg(utm_crs_list[0].code)
#     return crs


# def get_target_utm_srid(lat, lon):
#     """
#     Determine the target UTM SRID (as an integer) based on latitude and longitude.

#     Parameters:
#         lat (float): The latitude coordinate.
#         lon (float): The longitude coordinate.

#     Returns:
#         int: The UTM EPSG code as an integer. For example, for a point in the northern
#              hemisphere in UTM zone 33, the function returns 32633.

#     Raises:
#         ValueError: If the latitude is not in the valid range [-90, 90] or the longitude
#                     is not in the valid range [-180, 180].
#     """
#     # Basic input validation
#     if not (-90 <= lat <= 90):
#         raise ValueError("Latitude must be between -90 and 90.")
#     if not (-180 <= lon <= 180):
#         raise ValueError("Longitude must be between -180 and 180.")

#     # Determine UTM zone: zones are 6° wide starting at -180.
#     utm_zone = int((lon + 180) / 6) + 1

#     # For the northern hemisphere, UTM EPSG codes start at 32600; for the southern, 32700.
#     if lat >= 0:
#         return 32600 + utm_zone
#     else:
#         return 32700 + utm_zone


def create_circular_buffer(lon, lat, buffer_dist):
    """
    Creates a circular buffer in meters around a point and converts it to EPSG:4326.
    """
    point_geo = Point(lon, lat)
    point_utm, epsg_code = convert_geometry_to_utm(point_geo)

    # Ensure geometry is singular
    if isinstance(point_utm, gpd.GeoDataFrame):
        point_utm = point_utm.geometry.iloc[0]

    # Create buffer
    buffered_circle = point_utm.buffer(buffer_dist)

    # Convert buffer back to geographic CRS
    circle_gdf = gpd.GeoDataFrame(geometry=[buffered_circle], crs=epsg_code)
    geographic_circle = circle_gdf.to_crs("EPSG:4326").geometry.iloc[0]

    return geographic_circle


def extract_mucompdata_STATSGO(lon, lat):
    """
    Extracts and processes STATSGO data for the given longitude and latitude.

    Args:
        lon (float): Longitude of the point of interest.
        lat (float): Latitude of the point of interest.

    External Functions:
        sda_return (function): Function to execute the SDA query.
        trim_fraction (function): Function to trim fractions.

    Returns:
        DataFrame: Processed mucompdata, or error message if data is unavailable.
    """

    point = Point(lon, lat)
    buffer = create_circular_buffer(lon, lat, buffer_dist=5000)

    statsgo_mukey = load_statsgo_data(buffer)
    if statsgo_mukey is None:
        logging.warning(f"Soil ID not available in this area: {lon}.{lat}")
        return None

    mu_geo = statsgo_mukey[["MUKEY", "geometry"]].drop_duplicates(subset=["geometry"])
    mu_id_dist = calculate_distances_and_intersections(mu_geo, point)
    mu_id_dist.loc[mu_id_dist["pt_intersect"], "dist_meters"] = 0
    mu_id_dist["dist_meters"] = mu_id_dist.groupby("mukey")["dist_meters"].transform("min")
    mukey_dist_final = mu_id_dist.drop_duplicates("mukey").sort_values("dist_meters").head(2)

    mukey_list = mukey_dist_final["mukey"].tolist()
    if not mukey_list:
        logging.warning(f"Soil ID not available in this area: {lon}.{lat}")
        return None

    mucompdataQry = f"""SELECT component.mukey, component.cokey, component.compname,
                        component.comppct_r, component.compkind, component.majcompflag,
                        component.slope_r, component.elev_r, component.nirrcapcl,
                        component.nirrcapscl, component.nirrcapunit, component.irrcapcl,
                        component.irrcapscl, component.irrcapunit, component.taxorder,
                        component.taxsubgrp
                        FROM component
                        WHERE mukey IN ({",".join(map(str, mukey_list))})"""
    mucompdata_out = sda_return(propQry=mucompdataQry)

    if not mucompdata_out.empty:
        mucompdata = mucompdata_out["Table"].iloc[0]
        mucompdata_pd = pd.DataFrame(mucompdata[1:], columns=mucompdata[0])
        mucompdata_pd = pd.merge(mucompdata_pd, mukey_dist_final, on="mukey").sort_values(
            ["dist_meters", "cokey"]
        )

        mucompdata_pd = mucompdata_pd[mucompdata_pd["dist_meters"] <= 5000]

        if mucompdata_pd.empty:
            logging.warning(f"Soil ID not available in this area: {lon}.{lat}")
            return None
        else:
            return mucompdata_pd
    else:
        logging.warning(f"Soil ID not available in this area: {lon}.{lat}")
        return None


def process_site_data(mucompdata_pd):
    """
    Processes mucompdata by selecting relevant columns, renaming, sorting,
    and converting data types.

    Args:
        mucompdata_pd (pd.DataFrame): The DataFrame containing mucompdata to be processed.
        trim_fraction (function): A function to trim fractions from numeric data stored as strings.

    Returns:
        pd.DataFrame: The processed mucompdata DataFrame.
    """
    # rename columns
    if not isinstance(mucompdata_pd, pd.DataFrame):
        return None

    mucompdata_pd = mucompdata_pd.rename(
        columns={
            "dist_meters": "distance",
        }
    )

    # Define the columns to keep and rename 'distance_m' to 'distance'
    relevant_columns = [
        "mukey",
        "cokey",
        "compname",
        "compkind",
        "majcompflag",
        "comppct_r",
        "distance",
        "slope_r",
        "elev_r",
        "nirrcapcl",
        "nirrcapscl",
        "nirrcapunit",
        "irrcapcl",
        "irrcapscl",
        "irrcapunit",
        "taxorder",
        "taxsubgrp",
    ]
    mucompdata_pd = mucompdata_pd[relevant_columns].sort_values(["distance", "cokey"])

    # Replace 'NULL' with NaN and convert numeric columns to float
    mucompdata_pd.replace("NULL", np.nan, inplace=True)
    mucompdata_pd[["slope_r", "elev_r", "distance"]] = mucompdata_pd[
        ["slope_r", "elev_r", "distance"]
    ].astype(float)

    # Convert specified columns to string
    cols_to_str = [
        "mukey",
        "cokey",
        "compname",
        "compkind",
        "majcompflag",
        "nirrcapcl",
        "nirrcapscl",
        "nirrcapunit",
        "irrcapcl",
        "irrcapscl",
        "irrcapunit",
        "taxorder",
        "taxsubgrp",
    ]
    mucompdata_pd[cols_to_str] = mucompdata_pd[cols_to_str].astype(str)

    # Apply the trim_fraction function to specified columns
    mucompdata_pd["nirrcapcl"] = mucompdata_pd["nirrcapcl"].apply(trim_fraction)
    mucompdata_pd["irrcapcl"] = mucompdata_pd["irrcapcl"].apply(trim_fraction)

    return mucompdata_pd


def process_horizon_data(muhorzdata_pd):
    """
    Process the muhorzdata DataFrame by subsetting columns, converting data types,
    infilling missing values, and renaming columns for clarity.

    Args:
        muhorzdata_pd (pd.DataFrame): The DataFrame to process.

    Returns:
        pd.DataFrame: The processed DataFrame.
    """

    # rename columns
    muhorzdata_pd = muhorzdata_pd.rename(
        columns={
            "fragvol_r": "total_frag_volume",
        }
    )

    # Subset the DataFrame to include only the relevant columns
    relevant_columns = [
        "cokey",
        "hzdept_r",
        "hzdepb_r",
        "chkey",
        "hzname",
        "sandtotal_l",
        "sandtotal_r",
        "sandtotal_h",
        "silttotal_l",
        "silttotal_r",
        "silttotal_h",
        "claytotal_l",
        "claytotal_r",
        "claytotal_h",
        "dbovendry_l",
        "dbovendry_r",
        "dbovendry_h",
        "wthirdbar_l",
        "wthirdbar_r",
        "wthirdbar_h",
        "wfifteenbar_l",
        "wfifteenbar_r",
        "wfifteenbar_h",
        "total_frag_volume",
        "cec7_r",
        "ecec_r",
        "ph1to1h2o_r",
        "ec_r",
        "lep_r",
    ]

    muhorzdata_pd = muhorzdata_pd[relevant_columns]

    # Convert specific columns to appropriate data types
    numeric_columns = [
        "sandtotal_l",
        "sandtotal_r",
        "sandtotal_h",
        "silttotal_l",
        "silttotal_r",
        "silttotal_h",
        "claytotal_l",
        "claytotal_r",
        "claytotal_h",
        "dbovendry_l",
        "dbovendry_r",
        "dbovendry_h",
        "wthirdbar_l",
        "wthirdbar_r",
        "wthirdbar_h",
        "wfifteenbar_l",
        "wfifteenbar_r",
        "wfifteenbar_h",
        "total_frag_volume",
        "cec7_r",
        "ecec_r",
        "ph1to1h2o_r",
        "ec_r",
        "lep_r",
    ]
    muhorzdata_pd[numeric_columns] = muhorzdata_pd[numeric_columns].apply(pd.to_numeric)
    muhorzdata_pd[["cokey", "chkey", "hzname"]] = muhorzdata_pd[
        ["cokey", "chkey", "hzname"]
    ].astype(str)

    # Infill missing CEC values with ECEC
    muhorzdata_pd["CEC"] = muhorzdata_pd["cec7_r"].fillna(muhorzdata_pd["ecec_r"])

    # Rename columns for better clarity
    muhorzdata_pd = muhorzdata_pd.rename(
        columns={"cec7_r": "CEC", "ph1to1h2o_r": "pH", "ec_r": "EC"}
    )

    # Assign textures
    muhorzdata_pd["texture"] = muhorzdata_pd.apply(getTexture, axis=1)

    # Replace "NULL" strings with numpy NaN
    muhorzdata_pd.replace("NULL", np.nan, inplace=True)

    # Replace NaN with 0 for depth columns and convert them to int
    muhorzdata_pd[["hzdept_r", "hzdepb_r"]] = (
        muhorzdata_pd[["hzdept_r", "hzdepb_r"]].fillna(0).astype(int)
    )

    return muhorzdata_pd


def fill_missing_comppct_r(mucompdata_pd):
    """
    Fills missing or zero values in the 'comppct_r' column based on the difference
    between the sum of all components in the map unit subtracted from 100.

    Parameters:
    - mucompdata_pd (pd.DataFrame): DataFrame containing soil data.

    Returns:
    - pd.DataFrame: Updated DataFrame with processed 'comppct_r' values.
    """

    mukeys_with_missing_data = mucompdata_pd.query("comppct_r==0 | comppct_r.isnull()")[
        "mukey"
    ].drop_duplicates()

    if not mukeys_with_missing_data.empty:
        subset_data = mucompdata_pd[mucompdata_pd["mukey"].isin(mukeys_with_missing_data)]

        aggregated_data = subset_data.groupby("mukey").agg(
            total_pct=("comppct_r", "sum"),
            total_rows=("comppct_r", "size"),
            non_null_count=("comppct_r", "count"),
            zero_count=("comppct_r", lambda x: (x == 0).sum()),
        )

        aggregated_data["missing_data_count"] = (
            aggregated_data["total_rows"] - aggregated_data["non_null_count"]
        )
        aggregated_data["percent_diff"] = 100 - aggregated_data["total_pct"]
        aggregated_data["value_to_fill"] = aggregated_data["percent_diff"] / (
            aggregated_data["missing_data_count"] + aggregated_data["zero_count"]
        )

        for idx, row in aggregated_data.iterrows():
            condition = (mucompdata_pd["mukey"] == idx) & (
                mucompdata_pd["comppct_r"].isin([0, np.nan])
            )
            mucompdata_pd.loc[condition, "comppct_r"] = row["value_to_fill"]

    # Clean up the dataframe
    mucompdata_pd = (
        mucompdata_pd.drop_duplicates().sort_values(by="distance").reset_index(drop=True)
    )

    # Handle minor components that are either 0 or NaN
    mucompdata_pd.replace({"comppct_r": {np.nan: 1, 0: 1}}, inplace=True)
    mucompdata_pd["comppct_r"] = mucompdata_pd["comppct_r"].astype(int)

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

    Notes on location function:
        Individual probability
        Based on Fan et al 2018 EQ 1, the conditional probability for each component
        is calculated by taking the sum of all occurances of a component in the
        home and adjacent mapunits and dividing this by the sum of all map units
        and components. We have modified this approach so that each instance of a
        component occurance is evaluated separately and assinged a weight and the
        max distance score for each component group is assigned to all component instances.
    # --------------------------------------------------------------------

    """

    # Calculate distance score for each group
    mucompdata_pd["distance_score"] = mucompdata_pd.apply(
        lambda row: calculate_distance_score(row, ExpCoeff), axis=1
    )

    # Group by cokey and mukey and aggregate required values
    grouped_data = (
        mucompdata_pd.groupby(["cokey", "mukey"])
        .agg(
            distance_score=("distance_score", "sum"),
            comppct=("comppct_r", "sum"),
            minDistance=("distance", "min"),
        )
        .reset_index()
    )

    # Calculate conditional probabilities
    total_distance_score = grouped_data["distance_score"].sum()
    grouped_data["cond_prob"] = grouped_data["distance_score"] / total_distance_score

    # Merge dataframes on 'cokey'
    mucompdata_pd = mucompdata_pd.merge(
        grouped_data[["cokey", "cond_prob"]], on="cokey", how="left"
    )

    # Additional processing
    mucompdata_pd = mucompdata_pd.sort_values("distance_score", ascending=False)

    mucompdata_pd = mucompdata_pd[~mucompdata_pd["compkind"].str.contains("Miscellaneous area")]

    mucompdata_pd = mucompdata_pd.reset_index(drop=True)

    # Create a list of component groups
    mucompdata_comp_grps = [g for _, g in mucompdata_pd.groupby(["compname"], sort=False)]
    mucompdata_comp_grps = mucompdata_comp_grps[: min(12, len(mucompdata_comp_grps))]

    # Assign max within-group location-based score to all members of the group
    for group in mucompdata_comp_grps:
        group["distance_score"] = group["distance_score"].max()
        group = group.sort_values("distance").reset_index(drop=True)
        group["min_dist"] = group["distance"].iloc[0]

    # Concatenate the list of dataframes
    mucompdata_pd = pd.concat(mucompdata_comp_grps).reset_index(drop=True)

    return mucompdata_pd


###################################################################################################
#                                       Soil Color Functions                                      #
###################################################################################################


def pedon_color(lab_Color, top, bottom):
    pedon_l, pedon_a, pedon_b = (
        lab_Color.iloc[:, 0],
        lab_Color.iloc[:, 1],
        lab_Color.iloc[:, 2],
    )

    # Check for None values
    if any(x is None for x in [top, bottom]) or any(
        s.isnull().any() for s in [pedon_l, pedon_a, pedon_b]
    ):
        return np.nan

    if top[0] != 0:
        return np.nan

    # Check for missing horizons
    pedon_MisHrz = any(top[i + 1] != bottom[i] for i in range(len(top) - 1))
    if pedon_MisHrz:
        return np.nan

    pedon_l_intpl, pedon_a_intpl, pedon_b_intpl = [], [], []

    if len(top) == 1:
        pedon_l_intpl = [pedon_l[0]] * (bottom[0] - top[0])
        pedon_a_intpl = [pedon_a[0]] * (bottom[0] - top[0])
        pedon_b_intpl = [pedon_b[0]] * (bottom[0] - top[0])
    else:
        for i in range(len(bottom)):
            pedon_l_intpl.extend([pedon_l[i]] * (bottom[i] - top[i]))
            pedon_a_intpl.extend([pedon_a[i]] * (bottom[i] - top[i]))
            pedon_b_intpl.extend([pedon_b[i]] * (bottom[i] - top[i]))

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
        return [np.nan, np.nan, np.nan]
    else:
        return [pedon_l_mean, pedon_a_mean, pedon_b_mean]


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


# Functions for AWC simulation


def simulate_correlated_triangular(n, params, correlation_matrix):
    """
    Simulate correlated triangular distributed variables.

    Parameters:
    - n: Number of samples.
    - params: List of tuples, where each tuple contains three parameters (a, b, c) for the
              triangular distribution.
    - correlation_matrix: 2D numpy array representing the desired correlations between
                          the variables.

    Returns:
    - samples: 2D numpy array with n rows and as many columns as there are sets of
                  parameters in params.
    """
    # Sets the seed for numpy's random functions
    np.random.seed(soil_id.config.RANDOM_SEED)

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

        # Check for degenerate case where c - a == 0
        if c - a == 0:
            samples[:, i] = a
            continue

        # Compute the condition for the triangular distribution
        condition = u <= ((b - a) / (c - a))
        # Compute the two branches of the inverse CDF
        samples[condition, i] = a + np.sqrt(u[condition] * (c - a) * (b - a))
        samples[~condition, i] = c - np.sqrt((1 - u[~condition]) * (c - a) * (c - b))

    return samples


def regularize_matrix(matrix, epsilon=1e-8):
    """Add a small positive value to the diagonal of the matrix to ensure positive definiteness."""
    return matrix + np.eye(matrix.shape[0]) * epsilon


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


# Temporary function to infill missing data.
# TODO: create loopup table with average values for l-r-h by series
def impute_rfv_values(row):
    if row["rfv_r"] == 0:
        row["rfv_r"] = 0.02
        row["rfv_l"] = 0.01
        row["rfv_h"] = 0.03
    elif 0 < row["rfv_r"] <= 2:
        row["rfv_l"] = 0.01  # if pd.isna(row['rfv_l']) else row['rfv_l']
        row["rfv_h"] = row["rfv_r"] + 2  # if pd.isna(row['rfv_h']) else row['rfv_h']
    elif row["rfv_r"] > 2:
        row["rfv_l"] = row["rfv_r"] - 2  # if pd.isna(row['rfv_l']) else row['rfv_l']
        row["rfv_h"] = row["rfv_r"] + 2  # if pd.isna(row['rfv_h']) else row['rfv_h']
    return row


def remove_organic_layer(df):
    # Function to remove organic horizons and adjust depths within each group
    def process_group(group):
        # Filter out rows where 'hzname' contains a capital 'O'
        filtered_group = group[~group["hzname"].str.contains("O")]

        # Reset index to assist in calculating depth differences
        filtered_group = filtered_group.reset_index(drop=True)

        # If the group is empty after filtering, return it directly
        if filtered_group.empty:
            return filtered_group

        # Calculate and adjust the depth differences
        for i in range(len(filtered_group)):
            if i == 0:
                depth_diff = filtered_group.loc[i, "hzdept_r"]
            else:
                depth_diff += (
                    filtered_group.loc[i, "hzdept_r"] - filtered_group.loc[i - 1, "hzdepb_r"]
                )

            filtered_group.loc[i, "hzdept_r"] -= depth_diff
            filtered_group.loc[i, "hzdepb_r"] -= depth_diff

        return filtered_group

    # Group by 'compname_grp' and apply the processing function to each group
    result = df.groupby("compname_grp").apply(process_group).reset_index(drop=True)

    return result


def infill_soil_data(df):
    # Step 1: Group by 'compname'
    grouped = df.groupby("compname_grp")

    # Step 2: Check for missing 'r' values where 'hzdepb_r' <= 50
    # Filtering groups
    def filter_group(group):
        if (group["hzdepb_r"] <= 50).any() and group[
            ["sandtotal_r", "claytotal_r", "silttotal_r"]
        ].isnull().any().any():
            return False  # Exclude group
        return True  # Include group

    # Apply the filter to the groups
    filtered_groups = grouped.filter(filter_group)

    # Step 3: Replace missing '_l' and '_h' for particle size values
    # with corresponding '_r' values +/- 8
    for col in ["sandtotal", "claytotal", "silttotal"]:
        filtered_groups[col + "_l"] = (
            filtered_groups[col + "_l"]
            .fillna(filtered_groups[col + "_r"] - 8)
            .apply(lambda x: max(x, 0))
        )

        filtered_groups[col + "_h"] = (
            filtered_groups[col + "_h"]
            .fillna(filtered_groups[col + "_r"] + 8)
            .apply(lambda x: max(x, 0))
        )

    # Step 4 and 5: Replace missing 'dbovendry_l' and 'dbovendry_h' with 'dbovendry_r' +/- 0.01
    filtered_groups["dbovendry_l"] = (
        filtered_groups["dbovendry_l"]
        .fillna(filtered_groups["dbovendry_r"] - 0.01)
        .apply(lambda x: max(x, 0))
    )

    filtered_groups["dbovendry_h"] = (
        filtered_groups["dbovendry_h"]
        .fillna(filtered_groups["dbovendry_r"] + 0.01)
        .apply(lambda x: max(x, 0))
    )

    # Step 6 and 7: Replace missing 'wthirdbar_l' and 'wthirdbar_h' with 'wthirdbar_r' +/- 1
    filtered_groups["wthirdbar_l"] = (
        filtered_groups["wthirdbar_l"]
        .fillna(filtered_groups["wthirdbar_r"] - 1)
        .apply(lambda x: max(x, 0))
    )

    filtered_groups["wthirdbar_h"] = (
        filtered_groups["wthirdbar_h"]
        .fillna(filtered_groups["wthirdbar_r"] + 1)
        .apply(lambda x: max(x, 0))
    )

    # Step 8 and 9: Replace missing 'wfifteenbar_l' and 'wfifteenbar_h' with 'wfifteenbar_r' +/- 0.6
    filtered_groups["wfifteenbar_l"] = (
        filtered_groups["wfifteenbar_l"]
        .fillna(filtered_groups["wfifteenbar_r"] - 0.6)
        .apply(lambda x: max(x, 0))
    )

    filtered_groups["wfifteenbar_h"] = (
        filtered_groups["wfifteenbar_h"]
        .fillna(filtered_groups["wfifteenbar_r"] + 0.6)
        .apply(lambda x: max(x, 0))
    )

    # Step 10 and 11: Impute 'rfv_l' and 'rfv_h' values with 'rfv_r' +/- value
    filtered_groups = filtered_groups.apply(impute_rfv_values, axis=1)

    return filtered_groups


def slice_and_aggregate_soil_data(df):
    """
    Function to slice a DataFrame with soil data into 1 cm increments based
    on depth ranges provided in 'hzdept_r' and 'hzdepb_r' columns, and calculate mean
    values for each depth increment across all other data columns.

    Parameters:
    df (pd.DataFrame): DataFrame where each row represents a soil sample with 'hzdept_r'
    and 'hzdepb_r' columns.

    Returns:
    pd.DataFrame: A DataFrame with depth ranges and mean values of soil properties for each range.
    """

    # Select numeric columns for aggregation, excluding the depth range columns
    data_columns = df.select_dtypes(include=[np.number]).columns.difference(
        ["hzdept_r", "hzdepb_r"]
    )

    # Generate a DataFrame for each 1 cm increment within each row's depth range
    rows_list = []
    for _, row in df.iterrows():
        for depth in np.arange(row["hzdept_r"], row["hzdepb_r"]):
            rows_list.append({**{col: row[col] for col in data_columns}, "Depth": depth})

    # Create a single DataFrame from the list of rows
    aggregated_data = pd.DataFrame(rows_list)

    # Calculate mean values for each depth increment
    depth_increment_means = aggregated_data.groupby("Depth").mean().reset_index()

    # Define depth ranges
    depth_ranges = [(0, 30), (30, 100)]
    results = []

    # Process each depth range
    for top, bottom in depth_ranges:
        mask = (depth_increment_means["Depth"] >= top) & (depth_increment_means["Depth"] < bottom)
        subset = depth_increment_means[mask]

        # Calculate the mean for each column in the subset
        mean_values = subset.mean()
        mean_values["hzdept_r"] = top
        mean_values["hzdepb_r"] = bottom

        results.append(mean_values)

    result_df = pd.DataFrame(results).fillna(np.nan)

    # Check and add a row for the 30-100 cm depth range if not covered
    if 30 not in result_df["hzdept_r"].values:
        missing_row = {col: np.nan for col in result_df.columns}
        missing_row["hzdept_r"] = 30
        missing_row["hzdepb_r"] = 100
        result_df = result_df.append(missing_row, ignore_index=True)

    return result_df


def process_data_with_rosetta(df, vars, v=3, conf=None, include_sd=False):
    """
    Parameters:
    - df (DataFrame): The the DataFrame to be processed.
    - vars (list): List of variable names to be processed.
    - v (str): The version of the ROSETTA model to use.
    - conf (dict, optional): Additional request configuration options.
    - include_sd (bool): Whether to include standard deviation in the output.

    Returns:
    - DataFrame: The processed results from the ROSETTA python package.
    """
    # Select only the specified vars columns and other columns
    df_vars = df[vars]
    df_other = df.drop(columns=vars)

    # Convert the vars df to a matrix (2D list)
    df_vars_matrix = df_vars.values.tolist()

    mean, stdev, codes = rosetta(v, SoilData.from_array(df_vars_matrix))

    # Convert van Genuchten params to DataFrame
    vg_params = pd.DataFrame(mean)
    vg_params.columns = ["theta_r", "theta_s", "alpha", "npar", "ksat"]

    # Add model codes and version to the DataFrame
    vg_params[".rosetta.model"] = pd.Categorical.from_codes(
        codes, categories=["-1", "1", "2", "3", "4", "5"]
    )
    vg_params[".rosetta.version"] = v

    # If include_sd is True, add standard deviations to the DataFrame
    if include_sd:
        vg_sd = pd.DataFrame(stdev)
        vg_sd.columns = [f"sd_{name}" for name in vg_params.columns]
        result = pd.concat(
            [
                df_other.reset_index(drop=True),
                df_vars.reset_index(drop=True),
                vg_params.reset_index(drop=True),
                vg_sd,
            ],
            axis=1,
        )
    else:
        result = pd.concat(
            [
                df_other.reset_index(drop=True),
                df_vars.reset_index(drop=True),
                vg_params.reset_index(drop=True),
            ],
            axis=1,
        )

    return result


def vg_function(phi, theta_r, theta_s, alpha, n):
    """
    Calculates the van Genuchten equation.

    Parameters:
    phi (numpy array): An array of phi values.
    theta_r (float): Residual water content.
    theta_s (float): Saturated water content.
    alpha (float): Scale parameter in the van Genuchten equation.
    n (float): Shape parameter in the van Genuchten equation.

    Returns:
    numpy array: Calculated water content values based on the van Genuchten equation.
    """
    return theta_r + ((theta_s - theta_r) / ((1 + (alpha * phi) ** n) ** (1 - 1 / n)))


def calculate_vwc_awc(sim_data, phi_min=1e-6, phi_max=1e8, pts=100):
    """
    Calculates the volumetric water content (VWC) at specific matric potentials and determines
    the available water capacity (AWC).

    Parameters:
    sim_data (pandas DataFrame): DataFrame containing soil layers and their properties.

    Returns:
    pandas DataFrame: A DataFrame containing the VWC at saturation, field capacity, and permanent
    wilting point, along with the AWC for the specified layer.
    """
    required_columns = ["theta_r", "theta_s", "alpha", "npar", "layerID"]
    if not all(col in sim_data.columns for col in required_columns):
        raise ValueError("One or more required columns are missing.")

    # Handling missing values
    if sim_data[required_columns].isnull().any().any():
        raise ValueError("One or more required values are NA.")

    phi = np.logspace(np.log10(phi_min), np.log10(phi_max), pts)
    h = phi * 10.19716

    results = []
    for _, row in sim_data.iterrows():
        m = pd.DataFrame({"phi": phi})
        m["theta"] = vg_function(
            h,
            theta_r=row["theta_r"],
            theta_s=row["theta_s"],
            alpha=10 ** row["alpha"],
            n=10 ** row["npar"],
        )

        vg_fwd = UnivariateSpline(m["phi"], m["theta"], s=0)

        # Extract VWC at specific matric potentials (kPa)
        data = {
            "layerID": row["layerID"],
            "sat": vg_fwd(0),  # Saturation
            "fc": vg_fwd(33),  # Field Capacity
            "pwp": vg_fwd(1500),  # Permanent Wilting Point
        }
        data["awc"] = data["fc"] - data["pwp"]
        results.append(data)

    return pd.DataFrame(results)


def information_gain(data, target_col, feature_cols):
    """
    Calculate information gain for each feature with respect to the target variable.

    Args:
        data (pd.DataFrame): The DataFrame containing the dataset, including target
                             and feature columns.
        target_col (str): The name of the target variable column.
        feature_cols (list): List of feature column names.

    Returns:
        dict: A dictionary where keys are feature names and values are information gain scores.
    """

    def entropy_score(series):
        # Calculate entropy for a given series (e.g., target variable)
        value_counts = series.value_counts()
        probabilities = value_counts / len(series)
        return entropy(probabilities, base=2)

    # Calculate entropy of the entire dataset based on the target variable
    total_entropy = entropy_score(data[target_col])

    # Calculate information gain for each feature
    information_gains = {}
    for feature_col in feature_cols:
        # Calculate weighted average of entropies for each unique value of the feature
        feature_entropy = data.groupby(feature_col)[target_col].apply(entropy_score)
        feature_counts = data[feature_col].value_counts()
        weighted_feature_entropy = sum((feature_counts / len(data)) * feature_entropy.fillna(0))
        information_gain = total_entropy - weighted_feature_entropy
        information_gains[feature_col] = information_gain

    # Sort the information gains in descending order
    sorted_information_gains = sorted(information_gains.items(), key=lambda x: x[1], reverse=True)

    return sorted_information_gains


# function to get data and aggregate SG data
def sg_get_and_agg(variable, sg_data_w, bottom, return_depth=False):
    pd_int = getProfile_SG(sg_data_w, variable, c_bot=False)
    if return_depth:
        pd_lpks, lpks_depths = agg_data_layer(data=pd_int.var_pct_intpl, bottom=bottom, depth=True)
        return (pd_lpks.replace(np.nan, ""), lpks_depths)
    else:
        pd_lpks = agg_data_layer(data=pd_int.var_pct_intpl, bottom=bottom, depth=False)
        return pd_lpks.replace(np.nan, "")


def adjust_depth_interval(data, target_length=200):
    """Adjusts the depth interval of user data."""

    # Convert input to a DataFrame
    if isinstance(data, list):
        data = pd.DataFrame(data)
    elif isinstance(data, pd.Series):
        data = data.to_frame()

    # Ensure data is a DataFrame at this point
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Data must be a list, Series, or DataFrame")

    length = len(data)

    if length > target_length:
        # Truncate data if it exceeds the target length
        data = data.iloc[:target_length]
    elif length < target_length:
        # Extend data if it's shorter than the target length
        add_length = target_length - length
        add_data = pd.DataFrame(np.nan, index=np.arange(add_length), columns=data.columns)
        data = pd.concat([data, add_data])

    data.reset_index(drop=True, inplace=True)
    return data


# Helper function to update dataframes based on depth conditions
def update_intpl_data(
    df, col_names, values, very_bottom, OSD_depth_add, OSD_depth_remove, OSD_max_bottom_int
):
    if OSD_depth_add:
        layer_add = very_bottom - OSD_max_bottom_int
        pd_add = pd.DataFrame([values] * layer_add, columns=col_names)
        df = pd.concat([df.loc[: OSD_max_bottom_int - 1], pd_add], axis=0).reset_index(drop=True)
    elif OSD_depth_remove:
        df = df.loc[:very_bottom].reset_index(drop=True)
    return df


# Creates a new soil horizon layer row in the soil horizon table
def create_new_layer(row, hzdept, hzdepb):
    return pd.DataFrame(
        {
            "cokey": row["cokey"],
            "hzdept_r": hzdepb,
            "hzdepb_r": hzdept,
            "chkey": row["chkey"],
            "hzname": None,
            "sandtotal_r": np.nan,
            "silttotal_r": np.nan,
            "claytotal_r": np.nan,
            "total_frag_volume": np.nan,
            "CEC": np.nan,
            "pH": np.nan,
            "EC": np.nan,
            "lep_r": np.nan,
            "comppct_r": row["comppct_r"],
            "compname": row["compname"],
            "slope_r": np.nan,
            "texture": None,
        },
        index=[0],
    )


# Creates a new row entry in the OSD (Official Series Description) soil horizon table
def create_new_layer_osd(row, top, bottom):
    """Create a new layer with specified top and bottom depths."""
    new_row = row.copy()
    new_row["top"] = top
    new_row["bottom"] = bottom
    for col in [
        "hzname",
        "texture_class",
        "cf_class",
        "matrix_dry_color_hue",
        "matrix_dry_color_value",
        "matrix_dry_color_chroma",
    ]:
        new_row[col] = None
    for col in [
        "srgb_r",
        "srgb_g",
        "srgb_b",
        "total_frag_volume",
        "claytotal_r",
        "sandtotal_r",
    ]:
        new_row[col] = np.nan
    return new_row


##################################################################################################
#                                       Database and API Functions                               #
##################################################################################################
def find_region_for_location(lon, lat):
    """
    Determines the location type (US, Global, or None) of the given longitude and latitude
    based on soil datasets.

    Args:
    - lon (float): Longitude of the point.
    - lat (float): Latitude of the point.

    Returns:
    - str or None: 'US' if point is in US soil dataset,
                   'Global' if in global dataset,
                   None otherwise.
    """

    drv_us = ogr.GetDriverByName("ESRI Shapefile")
    ds_in_us = drv_us.Open(soil_id.config.US_AREA_PATH, 0)
    layer_us = ds_in_us.GetLayer(0)

    # Setup coordinate transformation
    geo_ref = layer_us.GetSpatialRef()
    pt_ref = ogr.osr.SpatialReference()
    pt_ref.ImportFromEPSG(4326)
    coord_transform = ogr.osr.CoordinateTransformation(pt_ref, geo_ref)

    # Transform the coordinate system of the input point
    lon, lat, _ = coord_transform.TransformPoint(lat, lon)

    # Create a point geometry
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.SetPoint_2D(0, lon, lat)

    # Filter layers using the point
    layer_us.SetSpatialFilter(pt)

    # Determine location type
    if len(layer_us) > 0:
        return "US"
    else:
        return "Global"


# calculate AWC for ROI
def calculate_aws(df, quantile):
    total = (df[quantile] * df["depth"] * df["n"]).sum()
    return pd.DataFrame({f"aws{quantile}_100": [total]})


def rename_simulated_soil_profile_columns(df, soil_property_columns, depth):
    new_column_names = {}
    for col in soil_property_columns:
        new_column_names[col] = f"{col}_{depth}"
    df.rename(columns=new_column_names, inplace=True)


# Function to handle the update of ecological site data
def update_esd_data(df):
    """
    Processes the given DataFrame by updating missing ESD data based on component
    groups with the same names, filling missing URLs and ecoclass IDs and names.
    """
    if "edit_url" not in df.columns:
        df["edit_url"] = ""  # Initialize 'edit_url' as "" if it does not exist

    # Replace group-specific data for missing ESD components
    df["compname_grp"] = df["compname"].str.replace(r"[0-9]+", "")
    grouped = df.groupby("compname_grp", sort=False)

    # Generate a list of updated groups
    updated_groups = []
    for _, group in grouped:
        unique_ids = group["ecoclassid"].dropna().unique()
        unique_names = group["ecoclassname"].dropna().unique()
        if group["ecoclassid"].isnull().all() or group["ecoclassname"].isnull().all():
            # Fill all missing values if all are missing within the group
            group.fillna(
                {
                    "ecoclassid": unique_ids[0] if unique_ids.size > 0 else "",
                    "ecoclassname": unique_names[0] if unique_names.size > 0 else "",
                },
                inplace=True,
            )
        elif len(unique_ids) == 1 and len(unique_names) == 1:
            # Fill missing values with existing unique values if present
            group.loc[:, "ecoclassid"] = group["ecoclassid"].fillna(unique_ids[0])
            group.loc[:, "ecoclassname"] = group["ecoclassname"].fillna(unique_ids[0])

        # Handle URLs separately as they might not exist
        urls = group["edit_url"].dropna().unique()
        group["edit_url"] = group["edit_url"].fillna(urls[0] if urls.size > 0 else "")
        updated_groups.append(group)

    return pd.concat(updated_groups, ignore_index=True)
