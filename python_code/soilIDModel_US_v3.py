# Standard libraries
import collections
import io
import json
import re

# Third-party libraries
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import shapely

# Flask
from flask import current_app, jsonify

# Import local fucntions
from model.local_functions_SoilID_v3 import (
    acomp,
    agg_data_layer,
    aggregate_data,
    calculate_vwc_awc,
    drop_cokey_horz,
    extract_muhorzdata_STATSGO,
    extract_statsgo_mucompdata,
    fill_missing_comppct_r,
    getCF_fromClass,
    getClay,
    getOSDCF,
    getProfile,
    getProfileLAB,
    getSand,
    getTexture,
    gower_distances,
    infill_soil_data,
    lab2munsell,
    load_model_output,
    munsell2rgb,
    process_data_with_rosetta,
    process_distance_scores,
    pt2polyDist,
    save_model_output,
    save_rank_output,
    sda_return,
    simulate_correlated_triangular,
    slice_and_aggregate_soil_data,
    trim_fraction,
)
from osgeo import ogr
from pandas.io.json import json_normalize
from scipy.stats import spearmanr
from shapely.geometry import Point
from skbio.stats.composition import ilr, ilr_inv

# entry points
# getSoilLocationBasedGlobal
# getSoilLocationBasedUS
# rankPredictionUS
# rankPredictionGlobal
# getSoilGridsGlobal
# getSoilGridsUS

# when a site is created, call getSoilLocationBasedUS/getSoilLocationBasedGlobal.
# when a site is created, call getSoilGridsGlobal
# after user has collected data, call rankPredictionUS/rankPredictionGlobal.


##############################################################################################
#                                   Database and API Functions                               #
##############################################################################################
def findSoilLocation(lon, lat):
    """
    Determines the location type (US, Global, or None) of the given longitude
        and latitude based on soil datasets.

    Args:
    - lon (float): Longitude of the point.
    - lat (float): Latitude of the point.

    Returns:
    - str or None: 'US' if point is in US soil dataset, 'Global' if in global dataset,
        None otherwise.
    """

    drv_h = ogr.GetDriverByName("ESRI Shapefile")
    ds_in_h = drv_h.Open(
        "%s/HWSD_global_noWater_no_country.shp" % current_app.config["DATA_BACKEND"], 0
    )
    layer_global = ds_in_h.GetLayer(0)

    drv_us = ogr.GetDriverByName("ESRI Shapefile")
    ds_in_us = drv_us.Open("%s/SoilID_US_Areas.shp" % current_app.config["DATA_BACKEND"], 0)
    layer_us = ds_in_us.GetLayer(0)

    # Setup coordinate transformation
    geo_ref = layer_global.GetSpatialRef()
    pt_ref = ogr.osr.SpatialReference()
    pt_ref.ImportFromEPSG(4326)
    coord_transform = ogr.osr.CoordinateTransformation(pt_ref, geo_ref)

    # Transform the coordinate system of the input point
    lon, lat, _ = coord_transform.TransformPoint(lon, lat)

    # Create a point geometry
    pt = ogr.Geometry(ogr.wkbPoint)
    pt.SetPoint_2D(0, lon, lat)

    # Filter layers using the point
    layer_global.SetSpatialFilter(pt)
    layer_us.SetSpatialFilter(pt)

    # Determine location type
    if not (len(layer_global) or len(layer_us)):
        return None
    elif len(layer_us):
        return "US"
    else:
        return "Global"


############################################################################################
#                                   getSoilLocationBasedUS                                 #
############################################################################################
def getSoilLocationBasedUS(lon, lat, plot_id):
    # Load in LAB to Munsell conversion look-up table
    color_ref = pd.read_csv("%s/LandPKS_munsell_rgb_lab.csv" % current_app.config["DATA_BACKEND"])
    LAB_ref = color_ref[["L", "A", "B"]]
    # rgb_ref = color_ref[["r", "g", "b"]]
    munsell_ref = color_ref[["hue", "value", "chroma"]]

    # Load in SSURGO data from SoilWeb
    # current production API
    # soilweb_url = f"""https://casoilresource.lawr.ucdavis.edu/api/landPKS.php?q=spn&lon={lon}&lat={lat}&r=1000"""

    # testing API
    soilweb_url = f"""https://soilmap2-1.lawr.ucdavis.edu/dylan/soilweb/api/landPKS.php?q=spn&lon={lon}&lat={lat}&r=1000"""
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

    OSD_compkind = ["Series", "Variant", "Family", "Taxadjunct"]
    # Check if point is in a NOTCOM area, and if so then infill with STATSGO from NRCS SDA
    if not out["spn"]:
        # Create LPKS point
        point = Point(lon, lat)
        point.crs = {"init": "epsg:4326"}

        # Create a bounding box to clip STATSGO data around the point
        s_buff = gpd.GeoSeries(point).buffer(0.1)  # 0.1 deg buffer around point = ~11km
        box = shapely.geometry.box(*s_buff.total_bounds)

        # Load STATSGO mukey data
        statsgo_mukey = gpd.read_file(
            "%s/gsmsoilmu_a_us.shp" % current_app.config["DATA_BACKEND"],
            bbox=box.bounds,
            mode="r",
            driver="ESRI Shapefile",
        )
        # Filter out mapunits with duplicate geometries
        mu_geo = statsgo_mukey[["MUKEY", "geometry"]].drop_duplicates(subset=["geometry"])

        # Calculate distances and intersection flags for each mapunit
        distances = [pt2polyDist(geom, point) for geom in mu_geo["geometry"]]
        intersects = [point.intersects(geom) for geom in mu_geo["geometry"]]

        # Create a DataFrame for distances and intersections
        mu_id_dist = pd.DataFrame(
            {
                "MUKEY": mu_geo["MUKEY"].values,
                "distance": distances,
                "pt_intersect": intersects,
            }
        )

        # Update distance to 0 for intersecting mapunits
        mu_id_dist.loc[mu_id_dist.pt_intersect, "distance"] = 0
        mu_id_dist["distance"] = mu_id_dist.groupby(["MUKEY"])["distance"].transform(min)
        mukey_dist_final = (
            mu_id_dist.drop_duplicates(subset=["MUKEY"]).sort_values(by="distance").head(2)
        )

        # Build the mucompdata query
        mukey_list = mukey_dist_final["MUKEY"].tolist()
        mucompdataQry = f"""SELECT component.mukey, component.cokey, component.compname,
        component.comppct_r, component.compkind, component.majcompflag,
        component.slope_r, component.elev_r, component.nirrcapcl, component.nirrcapscl,
        component.nirrcapunit, component.irrcapcl, component.irrcapscl,
        component.irrcapunit, component.taxorder, component.taxsubgrp
        FROM component WHERE mukey IN ({','.join(map(str, mukey_list))})"""

        mucompdata_out = sda_return(propQry=mucompdataQry)

        # Process the mucompdata results
        if mucompdata_out:
            mucompdata = mucompdata_out["Table"]
            mucompdata_pd = pd.DataFrame(mucompdata[1:], columns=mucompdata[0])
            mucompdata_pd = pd.merge(mucompdata_pd, mukey_dist_final, on="mukey").sort_values(
                ["distance", "cokey"]
            )
            mucompdata_pd.replace("NULL", np.nan, inplace=True)
            mucompdata_pd[["slope_r", "elev_r", "distance"]] = mucompdata_pd[
                ["slope_r", "elev_r", "distance"]
            ].astype(float)
            mucompdata_pd.nirrcapcl = mucompdata_pd.nirrcapcl.apply(trim_fraction)
            mucompdata_pd.irrcapcl = mucompdata_pd.irrcapcl.apply(trim_fraction)

            # Subset dataframe to extract only components within 5000m -- STATSGO
            mucompdata_pd = mucompdata_pd[mucompdata_pd["distance"] <= 5000]

            if mucompdata_pd.empty:
                return "Soil ID not available in this area"
            data_source = "STATSGO"
        else:
            return "Soil ID not available in this area"
    else:
        mucompdata_pd = pd.json_normalize(out["spn"])
        relevant_columns = [
            "mukey",
            "cokey",
            "compname",
            "compkind",
            "majcompflag",
            "comppct_r",
            "distance_m",
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
        mucompdata_pd = (
            mucompdata_pd[relevant_columns]
            .rename(columns={"distance_m": "distance"})
            .sort_values(["distance", "cokey"])
        )
        mucompdata_pd.replace("NULL", np.nan, inplace=True)
        mucompdata_pd[["slope_r", "elev_r", "distance"]] = mucompdata_pd[
            ["slope_r", "elev_r", "distance"]
        ].astype(float)
        # Modify mucompdata_pd DataFrame
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
        mucompdata_pd["nirrcapcl"] = mucompdata_pd["nirrcapcl"].apply(trim_fraction)
        mucompdata_pd["irrcapcl"] = mucompdata_pd["irrcapcl"].apply(trim_fraction)

        # Filter out data for distances over 1000m
        mucompdata_pd = mucompdata_pd[mucompdata_pd["distance"] <= 1000]

        if mucompdata_pd.empty:
            return "Soil ID not available in this area"
        data_source = "SSURGO"

    # Set the Exponential Decay Coefficient based on data source
    if data_source == "STATSGO":
        ExpCoeff = -0.0002772
    elif data_source == "SSURGO":
        ExpCoeff = -0.008

    # Process and fill missing or zero values in the 'comppct_r' column.
    mucompdata_pd = fill_missing_comppct_r(mucompdata_pd)

    # --------------------------------------------------------------------
    # Location based calculation
    # -----------------------------
    """
    # --------------------------------------------------------------------
    ############################################################################
    Individual probability
    Based on Fan et al 2018 EQ 1, the conditional probability for each component
    is calculated by taking the sum of all occurances of a component in the
    home and adjacent mapunits and dividing this by the sum of all map units
    and components. We have modified this approach so that each instance of a
    component occurance is evaluated separately and assinged a weight and the
    max distance score for each component group is assigned to all component instances.
    ############################################################################
    """
    # Process distance scores and perform group-wise aggregations.
    mucompdata_pd = process_distance_scores(mucompdata_pd, ExpCoeff)

    if mucompdata_pd.empty:
        return "Soil ID not available in this area"
    # Add the data source column
    mucompdata_pd["data_source"] = data_source

    # Get cokeys of components where compkind is in OSD_compkind
    cokey_series = mucompdata_pd[mucompdata_pd["compkind"].isin(OSD_compkind)]["cokey"].tolist()

    # Get component key list
    comp_key = mucompdata_pd["cokey"].tolist()

    # ------------------------------------------------------------------------
    # Extracts horizon data
    # ------------------------------------------------------------------------
    if data_source == "SSURGO":
        # Convert JSON data to DataFrame
        muhorzdata_pd = pd.json_normalize(out["hz"])[
            [
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
        ]
        # Convert specific columns to appropriate data types
        numeric_columns = [
            "total_frag_volume",
            "sandtotal_r",
            "silttotal_r",
            "claytotal_r",
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
        muhorzdata_pd.cec7_r = np.where(
            pd.isnull(muhorzdata_pd.cec7_r), muhorzdata_pd.ecec_r, muhorzdata_pd.cec7_r
        )

        # Rename columns for better clarity
        muhorzdata_pd.rename(
            columns={"cec7_r": "CEC", "ph1to1h2o_r": "pH", "ec_r": "EC"}, inplace=True
        )

        # Filter rows based on component keys
        muhorzdata_pd = muhorzdata_pd.loc[muhorzdata_pd["cokey"].isin(comp_key)]

        # If dataset is empty and none of the components are Series, switch to STATSGO
        if (
            muhorzdata_pd[["hzdept_r", "hzdepb_r"]].isnull().all().all()
            and not mucompdata_pd["compkind"].isin(OSD_compkind).any()
        ):
            # STATSGO Component Data Processing
            mucompdata_pd = extract_statsgo_mucompdata(lon, lat)

            # Process the mucompdata results
            if mucompdata_out is None:
                return "Soil ID not available in this area"
            else:
                data_source = "STATSGO"
                ExpCoeff = -0.0002772  # Expotential decay coefficient: 0.25 @ ~5km

                # Process and fill missing or zero values in the 'comppct_r' column.
                mucompdata_pd = fill_missing_comppct_r(mucompdata_pd)

                # --------------------------------------------------------------------
                # Location based calculation
                # -----------------------------
                """
                # --------------------------------------------------------------------
                ############################################################################
                Individual probability
                Based on Fan et al 2018 EQ 1, the conditional probability for each component
                is calculated by taking the sum of all occurances of a component in the
                home and adjacent mapunits and dividing this by the sum of all map units
                and components. We have modified this approach so that each instance of a
                component occurance is evaluated separately and assinged a weight and the
                max distance score for each component group is assigned to all component instances.
                ############################################################################
                """
                # Process distance scores and perform group-wise aggregations.
                mucompdata_pd = process_distance_scores(mucompdata_pd, ExpCoeff)

                if mucompdata_pd.empty:
                    return "Soil ID not available in this area"
                # Add the data source column
                mucompdata_pd["data_source"] = data_source

                # Get cokeys of components where compkind is in OSD_compkind
                cokey_series = mucompdata_pd[mucompdata_pd["compkind"].isin(OSD_compkind)][
                    "cokey"
                ].tolist()

                # -----------------------------------------------------------------------------------------------------------
                # STATSGO Horizon Data Query
                muhorzdata_pd = extract_muhorzdata_STATSGO(mucompdata_pd)

    elif data_source == "STATSGO":
        # STATSGO Horizon Data Query
        muhorzdata_pd = extract_muhorzdata_STATSGO(mucompdata_pd)

    # Merge muhorzdata_pd with selected columns from mucompdata_pd
    muhorzdata_pd = pd.merge(
        muhorzdata_pd,
        mucompdata_pd[["cokey", "comppct_r", "compname", "distance_score", "slope_r"]],
        on="cokey",
        how="left",
    )

    # Replace "NULL" strings with numpy NaN
    muhorzdata_pd.replace("NULL", np.nan, inplace=True)

    # Filter out components with missing horizon depth data that aren't either
    # a Series, Variant, or Family
    filter_condition = muhorzdata_pd["cokey"].isin(cokey_series) | (
        pd.notnull(muhorzdata_pd["hzdept_r"]) & pd.notnull(muhorzdata_pd["hzdepb_r"])
    )
    muhorzdata_pd = muhorzdata_pd[filter_condition]

    # Replace NaN with 0 for depth columns and convert them to int
    muhorzdata_pd[["hzdept_r", "hzdepb_r"]] = (
        muhorzdata_pd[["hzdept_r", "hzdepb_r"]].fillna(0).astype(int)
    )

    # Assign textures
    muhorzdata_pd["texture"] = muhorzdata_pd.apply(getTexture, axis=1)

    # Drop duplicates and reset index
    muhorzdata_pd.drop_duplicates(inplace=True)
    muhorzdata_pd.reset_index(drop=True, inplace=True)

    # Check for duplicate component instances
    hz_drop = drop_cokey_horz(muhorzdata_pd)
    if hz_drop is not None:
        muhorzdata_pd = muhorzdata_pd[~muhorzdata_pd.cokey.isin(hz_drop)]

    muhorzdata_pd.reset_index(drop=True, inplace=True)

    # Extract unique cokeys
    comp_key = muhorzdata_pd["cokey"].unique().tolist()

    # Subset mucompdata_pd using the updated comp_key
    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(comp_key)]

    # Sort mucompdata_pd based on 'distance_score' and 'distance'
    mucompdata_pd.sort_values(["distance_score", "distance"], ascending=[False, True], inplace=True)
    mucompdata_pd.reset_index(drop=True, inplace=True)

    # Create a new column "compname_grp" that duplicates the "compname" column
    mucompdata_pd["compname_grp"] = mucompdata_pd["compname"]

    # Update comp_key with the unique cokeys from the sorted mucompdata_pd
    comp_key = mucompdata_pd["cokey"].unique().tolist()

    # Create a dictionary to map each cokey to its index (rank) for sorting
    cokey_Index = dict(zip(comp_key, range(len(comp_key))))

    # Map the ranking of each cokey from mucompdata_pd to muhorzdata_pd
    muhorzdata_pd["Comp_Rank"] = muhorzdata_pd["cokey"].map(cokey_Index)

    # Sort muhorzdata_pd based on the 'Comp_Rank' and 'hzdept_r' columns
    muhorzdata_pd.sort_values(["Comp_Rank", "hzdept_r"], ascending=[True, True], inplace=True)

    # Drop the 'Comp_Rank' column after sorting
    muhorzdata_pd.drop("Comp_Rank", axis=1, inplace=True)
    muhorzdata_pd.reset_index(drop=True, inplace=True)

    # Update component names in mucompdata_pd to handle duplicates
    component_names = mucompdata_pd["compname"].tolist()
    name_counts = collections.Counter(component_names)

    for name, count in name_counts.items():
        if count > 1:  # If a component name is duplicated
            suffixes = range(1, count + 1)  # Generate suffixes for the duplicate names
            for suffix in suffixes:
                index = component_names.index(
                    name
                )  # Find the index of the first occurrence of the duplicate name
                component_names[index] = name + str(suffix)  # Append the suffix

    mucompdata_pd["compname"] = component_names
    muhorzdata_pd.rename(columns={"compname": "compname_grp"}, inplace=True)
    # Merge the modified component names from mucompdata_pd to muhorzdata_pd
    muhorzdata_pd = muhorzdata_pd.merge(
        mucompdata_pd[["cokey", "compname"]], on="cokey", how="left"
    )

    # Remove bedrock by filtering out 'R|r' in hzname
    muhorzdata_pd = muhorzdata_pd[~muhorzdata_pd["hzname"].str.contains("R", case=False, na=False)]

    # Group data by cokey
    muhorzdata_group_cokey = [group for _, group in muhorzdata_pd.groupby("cokey", sort=False)]

    # Helper function to create a new layer
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

    getProfile_cokey = []
    c_bottom_depths = []
    clay_texture = []
    snd_lyrs = []
    cly_lyrs = []
    txt_lyrs = []
    hz_lyrs = []
    rf_lyrs = []
    cec_lyrs = []
    ph_lyrs = []
    ec_lyrs = []
    OSD_text_int = []
    OSD_rfv_int = []

    for group in muhorzdata_group_cokey:
        # Sort by top horizon depth and remove duplicates
        group_sorted = group.sort_values(by="hzdept_r").drop_duplicates().reset_index(drop=True)

        # Remove organic horizons with incorrect depth values
        group_sorted = group_sorted[
            group_sorted["hzdepb_r"] >= group_sorted["hzdept_r"]
        ].reset_index(drop=True)

        # Infill missing horizons
        if group_sorted.hzdept_r.iloc[0] != 0:
            layer = create_new_layer(group_sorted.iloc[0], group_sorted.hzdept_r.iloc[0], 0)
            group_sorted = (
                pd.concat([group_sorted, layer]).sort_values("hzdept_r").reset_index(drop=True)
            )

        # Check for missing subsurface horizons and infill
        for j in range(len(group_sorted) - 1):
            if group_sorted.hzdept_r.iloc[j + 1] > group_sorted.hzdepb_r.iloc[j]:
                layer = create_new_layer(
                    group_sorted.iloc[j],
                    group_sorted.hzdept_r.iloc[j + 1],
                    group_sorted.hzdepb_r.iloc[j],
                )
                group_sorted = (
                    pd.concat([group_sorted, layer]).sort_values("hzdept_r").reset_index(drop=True)
                )

        c_very_bottom, sand_pct_intpl = getProfile(group_sorted, "sandtotal_r", c_bot=True)
        sand_pct_intpl.columns = ["c_sandpct_intpl", "c_sandpct_intpl_grp"]
        clay_pct_intpl = getProfile(group_sorted, "claytotal_r")
        clay_pct_intpl.columns = ["c_claypct_intpl", "c_claypct_intpl_grp"]
        cf_pct_intpl = getProfile(group_sorted, "total_frag_volume")
        cf_pct_intpl.columns = ["c_cfpct_intpl", "c_cfpct_intpl_grp"]
        cec_intpl = getProfile(group_sorted, "CEC")
        cec_intpl.columns = ["c_cec_intpl"]
        ph_intpl = getProfile(group_sorted, "pH")
        ph_intpl.columns = ["c_ph_intpl"]
        ec_intpl = getProfile(group_sorted, "EC")
        ec_intpl.columns = ["c_ec_intpl"]
        compname = pd.DataFrame([group_sorted.compname.unique()] * len(sand_pct_intpl))
        comppct = pd.DataFrame([group_sorted.comppct_r.unique()] * len(sand_pct_intpl))
        cokey = pd.DataFrame([group_sorted.cokey.unique()] * len(sand_pct_intpl))
        getProfile_cokey_temp2 = pd.concat(
            [
                sand_pct_intpl[["c_sandpct_intpl_grp"]],
                clay_pct_intpl[["c_claypct_intpl_grp"]],
                cf_pct_intpl[["c_cfpct_intpl_grp"]],
                compname,
                cokey,
                comppct,
            ],
            axis=1,
        )
        getProfile_cokey_temp2.columns = [
            "sandpct_intpl",
            "claypct_intpl",
            "rfv_intpl",
            "compname",
            "cokey",
            "comppct",
        ]
        mucompdata_pd_group = mucompdata_pd[
            mucompdata_pd["cokey"].isin(getProfile_cokey_temp2["cokey"])
        ]

        if (
            getProfile_cokey_temp2.sandpct_intpl.isnull().values.all()
            or getProfile_cokey_temp2.claypct_intpl.isnull().values.all()
        ) and (mucompdata_pd_group["compkind"].isin(OSD_compkind).any()):
            OSD_text_int.append("Yes")
        else:
            OSD_text_int.append("No")

        if (getProfile_cokey_temp2.rfv_intpl.isnull().values.all()) and (
            mucompdata_pd_group["compkind"].isin(OSD_compkind).any()
        ):
            OSD_rfv_int.append("Yes")
        else:
            OSD_rfv_int.append("No")

        cokeyB = getProfile_cokey_temp2["cokey"].iloc[0]
        compnameB = getProfile_cokey_temp2["compname"].iloc[0]
        c_bottom_depths_temp = pd.DataFrame([cokeyB, compnameB, int(c_very_bottom)]).T
        c_bottom_depths_temp.columns = ["cokey", "compname", "c_very_bottom"]

        c_bottom_depths.append(c_bottom_depths_temp)
        getProfile_cokey.append(getProfile_cokey_temp2)
        comp_texture_list = group_sorted.texture.str.lower().tolist()
        comp_texture_list = [x for x in comp_texture_list if x is not None]
        if any("clay" in string for string in comp_texture_list):
            clay_texture_temp = pd.DataFrame([compnameB, "Yes"]).T.dropna()
        else:
            clay_texture_temp = pd.DataFrame([compnameB, "No"]).T.dropna()
        clay_texture_temp.columns = ["compname", "clay"]
        clay_texture.append(clay_texture_temp)

        # extract horizon data
        hz_depb = group_sorted["hzdepb_r"]
        snd_d = group_sorted["sandtotal_r"]
        cly_d = group_sorted["claytotal_r"]
        txt_d = group_sorted["texture"]
        rf_d = group_sorted["total_frag_volume"]
        cec_d = group_sorted["CEC"]
        ec_d = group_sorted["EC"]
        ph_d = group_sorted["pH"]

        hz_depb = hz_depb.fillna("")
        snd_d = snd_d.fillna("")
        cly_d = cly_d.fillna("")
        rf_d = rf_d.fillna("")
        cec_d = cec_d.fillna("")
        ph_d = ph_d.fillna("")
        ec_d = ec_d.fillna("")
        txt_d = txt_d.fillna("")
        snd_lyrs.append(dict(zip(snd_d.index, snd_d)))
        cly_lyrs.append(dict(zip(cly_d.index, cly_d)))
        txt_lyrs.append(dict(zip(txt_d.index, txt_d)))
        rf_lyrs.append(dict(zip(rf_d.index, rf_d)))
        cec_lyrs.append(dict(zip(cec_d.index, cec_d)))
        ph_lyrs.append(dict(zip(ph_d.index, ph_d)))
        ec_lyrs.append(dict(zip(ec_d.index, ec_d)))
        hz_lyrs.append(dict(zip(hz_depb.index, hz_depb)))

    c_bottom_depths = pd.concat(c_bottom_depths, axis=0)
    clay_texture = pd.concat(clay_texture, axis=0)

    # Filter main dataframes based on cokeys in c_bottom_depths
    valid_cokeys = c_bottom_depths["cokey"]
    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(valid_cokeys)]
    muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(valid_cokeys)]

    # Add OSD infilling indicators to mucompdata_pd
    mucompdata_pd["OSD_text_int"] = OSD_text_int
    mucompdata_pd["OSD_rfv_int"] = OSD_rfv_int

    # Merge component bottom depth and clay texture information into mucompdata_pd
    mucompdata_pd = mucompdata_pd.merge(
        c_bottom_depths[["compname", "c_very_bottom"]], on="compname", how="left"
    )
    mucompdata_pd = mucompdata_pd.merge(clay_texture, on="compname", how="left")

    # Create a mapping from cokey to index
    comp_key = mucompdata_pd["cokey"].unique().tolist()
    cokey_Index = {key: index for index, key in enumerate(comp_key)}

    # ------------------------------------------------------------------------

    """
    Simulation modeling Steps:
      Step 1. Calculate a local soil property correlation matrix uisng the
              representative values from SSURGO.

      Step 2. Steps performed on each row:
        a. Simulate sand/silt/clay percentages using the 'simulate_correlated_triangular'
           function, using the global correlation matrix and the local l,r,h values for
           each particle fraction. Format as a composition using 'acomp'
        b. Perform the isometric log-ratio transformation.
        c. Extract l,r,h values (min, median, max for ilr1 and ilr2) and format into a
           params object for simiulation.
        d. Simulate all properties and then permorm inverse transform on ilr1 and ilr2
           to obtain sand, silt, and clay values.
        e. Append simulated values to dataframe

      Step 3. Run Rosetta and other Van Genuchten equations to calcuate AWS in top 50
              cm using simulated dataframe.
    """

    # Step 1. Calculate a local soil property correlation matrix

    # Subset data based on required soil inputs
    sim_columns = [
        "compname_grp",
        "distance_score",
        "hzdept_r",
        "hzdepb_r",
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
    ]

    sim_data_columns = [
        "hzdept_r",
        "hzdepb_r",
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
    ]
    sim = muhorzdata_pd[sim_columns]

    # infill missing data
    sim = infill_soil_data(sim)

    agg_data = []

    # Group data by compname_grp
    sim_group_compname = [group for _, group in sim.groupby("compname_grp", sort=False)]
    for group in sim_group_compname:
        # aggregate data into 0-20 and 20-50 or bottom depth
        group_ag = slice_and_aggregate_soil_data(group[sim_data_columns])
        group_ag["compname_grp"] = group["compname_grp"].unique()[0]
        group_ag["distance_score"] = group["distance_score"].unique()[0]
        group_ag = group_ag.drop("Depth", axis=1)
        agg_data.append(group_ag)

    # Concatenate the results for each column into a single dataframe
    agg_data_df = pd.concat(agg_data, axis=0, ignore_index=True).dropna().reset_index(drop=True)

    # return(jsonify(agg_data_df.to_dict()))
    # return(json.dumps(agg_data_df[["sandtotal_r", "silttotal_r", "claytotal_r"]].values.tolist()))

    # Extract columns with names ending in '_r'
    agg_data_r = agg_data_df[[col for col in agg_data_df.columns if col.endswith("_r")]]

    # Compute the local correlation matrix (Spearman correlation matrix)
    rep_columns = agg_data_r.drop(columns=["sandtotal_r", "silttotal_r", "claytotal_r"])
    # correlation_matrix, _ = spearmanr(selected_columns, axis=0)

    ilr_site_txt = ilr(agg_data_df[["sandtotal_r", "silttotal_r", "claytotal_r"]].values)
    # return(json.dumps(ilr_site_txt[:, 0].tolist()))
    rep_columns["ilr1"] = pd.Series(ilr_site_txt[:, 0])
    rep_columns["ilr2"] = pd.Series(ilr_site_txt[:, 1])

    correlation_matrix_data = rep_columns[
        [
            "ilr1",
            "ilr2",
            "dbovendry_r",
            "wthirdbar_r",
            "wfifteenbar_r",
        ]
    ]
    local_correlation_matrix, _ = spearmanr(correlation_matrix_data, axis=0)
    """
    Step 2. Simulate data for each row, with the number of simulations equal
            to the (distance_score*100)*10
    """
    # Global soil texture correlation matrix (used for initial simulation)
    texture_correlation_matrix = np.array(
        [
            [1.0000000, -0.76231798, -0.67370589],
            [-0.7623180, 1.00000000, 0.03617498],
            [-0.6737059, 0.03617498, 1.00000000],
        ]
    )

    sim_data_out = []

    for _, row in agg_data_df.iterrows():
        # 2a. Simulate sand/silt/clay percentages
        # 1. Extract and format data params
        sand_params = [row["sandtotal_l"], row["sandtotal_r"], row["sandtotal_h"]]
        silt_params = [row["silttotal_l"], row["silttotal_r"], row["silttotal_h"]]
        clay_params = [row["claytotal_l"], row["claytotal_r"], row["claytotal_h"]]

        params_txt = [sand_params, silt_params, clay_params]

        # 2. Perform processing steps on data
        #    Convert simulated data using the acomp function and then compute the isometric
        #    log-ratio transformation.
        simulated_txt = acomp(
            simulate_correlated_triangular(
                (int(row["distance_score"] * 1000)), params_txt, texture_correlation_matrix
            )
        )
        simulated_txt_ilr = ilr(simulated_txt)

        # Extract min, median, and max for the first two ilr transformed columns.
        ilr1_values = simulated_txt_ilr[:, 0]
        ilr2_values = simulated_txt_ilr[:, 1]

        ilr1_l, ilr1_r, ilr1_h = (
            ilr1_values.min(),
            np.median(ilr1_values),
            ilr1_values.max(),
        )
        ilr2_l, ilr2_r, ilr2_h = (
            ilr2_values.min(),
            np.median(ilr2_values),
            ilr2_values.max(),
        )

        # Create the list of parameters.
        params = [
            [ilr1_l, ilr1_r, ilr1_h],
            [ilr2_l, ilr2_r, ilr2_h],
            [row["dbovendry_l"], row["dbovendry_r"], row["dbovendry_h"]],
            [row["wthirdbar_l"], row["wthirdbar_r"], row["wthirdbar_h"]],
            [row["wfifteenbar_l"], row["wfifteenbar_r"], row["wfifteenbar_h"]],
        ]

        sim_data = simulate_correlated_triangular(
            int(row["distance_score"] * 1000), params, local_correlation_matrix
        )
        sim_data = pd.DataFrame(
            sim_data,
            columns=[
                "ilr1",
                "ilr2",
                "bulk_density_third_bar",
                "water_retention_third_bar",
                "water_retention_15_bar",
            ],
        )
        sim_data["water_retention_third_bar"] = sim_data["water_retention_third_bar"].div(100)
        sim_data["water_retention_15_bar"] = sim_data["water_retention_15_bar"].div(100)
        sim_txt = ilr_inv(sim_data[["ilr1", "ilr2"]])
        sim_txt = pd.DataFrame(sim_txt, columns=["sand_total", "silt_total", "clay_total"])
        sim_txt = sim_txt.multiply(100)
        multi_sim = pd.concat([sim_data.drop(columns=["ilr1", "ilr2"]), sim_txt], axis=1)
        multi_sim["compname_grp"] = row["compname_grp"]
        multi_sim["hzdept_r"] = row["hzdept_r"]
        multi_sim["hzdepb_r"] = row["hzdepb_r"]
        sim_data_out.append(multi_sim)
    sim_data_df = pd.concat(sim_data_out, axis=0, ignore_index=True)

    # Convert NaN values to None
    sim_data_df = sim_data_df.where(pd.notna(sim_data_df), None)

    # Convert the DataFrame to the desired format
    # rossetta_input_data = sim_data_df.values.tolist()
    variables = [
        "sand_total",
        "silt_total",
        "clay_total",
        "bulk_density_third_bar",
        "water_retention_third_bar",
        "water_retention_15_bar",
    ]
    rosetta_data = process_data_with_rosetta(sim_data_df, vars=variables, v="3")

    # Create layerID
    sim_data_df["layerID"] = sim_data_df["compname_grp"] + "_" + sim_data_df["hzdept_r"].astype(str)
    rosetta_data["layerID"] = sim_data_df["layerID"]

    awc = calculate_vwc_awc(rosetta_data)
    return jsonify(awc)

    # ----------------------------------------------------------------------------
    # This extracts OSD color, texture, and CF data
    if data_source == "STATSGO":
        # If the condition is met, we perform the series of operations, otherwise,
        # we set OSDhorzdata_pd to None
        if mucompdata_pd["compkind"].isin(OSD_compkind).any():
            try:
                # Generate series names
                series_name = [
                    re.sub("[0-9]+", "", compname)
                    for compname in mucompdata_pd["compname"]
                    if compname in OSD_compkind
                ]

                params = {"q": "site_hz", "s": series_name}

                # Fetch data from URL using requests
                series_url = "https://casoilresource.lawr.ucdavis.edu/api/soil-series.php"
                response = requests.get(series_url, params=params, timeout=3)
                seriesDict = response.json()

                # Normalize the data and perform data manipulations
                OSDhorzdata_pd = pd.json_normalize(seriesDict["hz"])[
                    [
                        "series",
                        "top",
                        "bottom",
                        "hzname",
                        "matrix_dry_color_hue",
                        "matrix_dry_color_value",
                        "matrix_dry_color_chroma",
                        "texture_class",
                        "cf_class",
                    ]
                ]
                OSDhorzdata_pd["texture_class"] = (
                    OSDhorzdata_pd["texture_class"]
                    .str.lower()
                    .str.replace(r"(fine|medium|coarse) ", "")
                )
                OSDhorzdata_pd.loc[
                    OSDhorzdata_pd["matrix_dry_color_hue"] == "N",
                    "matrix_dry_color_value",
                ] = 1

                # Color conversion
                munsell_RGB = []
                for hue, value, chroma in zip(
                    OSDhorzdata_pd["matrix_dry_color_hue"],
                    OSDhorzdata_pd["matrix_dry_color_value"],
                    OSDhorzdata_pd["matrix_dry_color_chroma"],
                ):
                    if pd.isnull(hue) or pd.isnull(value) or pd.isnull(chroma):
                        munsell_RGB.append([np.nan, np.nan, np.nan])
                    else:
                        munsell = pd.Series([hue, value, chroma])
                        RGB = munsell2rgb(color_ref, munsell_ref, munsell)
                        munsell_RGB.append(RGB)

                munsell_RGB_sim = pd.DataFrame(munsell_RGB, columns=["r", "g", "b"])
                OSDhorzdata_pd = pd.concat([OSDhorzdata_pd, munsell_RGB_df], axis=1)

                # Merge with another dataframe
                mucompdata_pd_merge = mucompdata_pd[["mukey", "cokey", "compname", "compkind"]]
                mucompdata_pd_merge["series"] = mucompdata_pd_merge["compname"].str.replace(
                    r"\d+", ""
                )
                OSDhorzdata_pd["series"] = OSDhorzdata_pd["series"].str.lower().str.capitalize()
                OSDhorzdata_pd = pd.merge(
                    mucompdata_pd_merge, OSDhorzdata_pd, on="series", how="left"
                )

                # Data type conversions and value assignments
                columns_to_str = [
                    "series",
                    "hzname",
                    "texture_class",
                    "cf_class",
                    "matrix_dry_color_hue",
                    "matrix_dry_color_value",
                    "matrix_dry_color_chroma",
                ]
                OSDhorzdata_pd[columns_to_str] = OSDhorzdata_pd[columns_to_str].astype(str)
                OSDhorzdata_pd[["top", "bottom"]] = (
                    OSDhorzdata_pd[["top", "bottom"]].fillna(0).astype(int)
                )
                OSDhorzdata_pd["cf_class"] = OSDhorzdata_pd["cf_class"].astype(str)
                OSDhorzdata_pd["total_frag_volume"] = OSDhorzdata_pd["cf_class"].apply(getOSDCF)
                OSDhorzdata_pd["claytotal_r"] = OSDhorzdata_pd["texture_class"].apply(getClay)
                OSDhorzdata_pd["sandtotal_r"] = OSDhorzdata_pd["texture_class"].apply(getSand)
            except Exception as err:
                OSDhorzdata_pd = None
                print("An error occurred:", err)
        else:
            OSDhorzdata_pd = None

    elif data_source == "SSURGO":
        # Check if 'OSD_morph' from SoilWeb API is False
        if not out.get("OSD_morph"):
            OSDhorzdata_pd = None
        else:
            # Normalize the data
            OSDhorzdata_pd = pd.json_normalize(out["OSD_morph"])

            # Prepare for merge
            mucompdata_pd_merge = mucompdata_pd[["mukey", "cokey", "compname", "compkind"]]
            mucompdata_pd_merge["series"] = mucompdata_pd_merge["compname"].str.replace(r"\d+", "")

            # Filter and merge the dataframes
            OSDhorzdata_pd = OSDhorzdata_pd[
                [
                    "series",
                    "top",
                    "bottom",
                    "hzname",
                    "texture_class",
                    "cf_class",
                    "matrix_dry_color_hue",
                    "matrix_dry_color_value",
                    "matrix_dry_color_chroma",
                    "r",
                    "g",
                    "b",
                ]
            ]
            OSDhorzdata_pd = pd.merge(mucompdata_pd_merge, OSDhorzdata_pd, on="series", how="left")

            # Set data types for specific columns
            columns_to_str = [
                "series",
                "hzname",
                "texture_class",
                "cf_class",
                "matrix_dry_color_hue",
                "matrix_dry_color_value",
                "matrix_dry_color_chroma",
            ]
            OSDhorzdata_pd[columns_to_str] = OSDhorzdata_pd[columns_to_str].astype(str)

            # Remove texture modifiers for sand fraction using a regex pattern for efficiency
            OSDhorzdata_pd["texture_class"] = (
                OSDhorzdata_pd["texture_class"]
                .str.lower()
                .str.replace(r"(fine|medium|coarse) ", "")
            )

            # For horizons missing all depth data, replace NaN with 0
            OSDhorzdata_pd[["top", "bottom"]] = (
                OSDhorzdata_pd[["top", "bottom"]].fillna(0).astype(int)
            )

            # Update specific columns based on functions
            OSDhorzdata_pd["total_frag_volume"] = OSDhorzdata_pd["cf_class"].apply(getOSDCF)
            OSDhorzdata_pd["claytotal_r"] = OSDhorzdata_pd["texture_class"].apply(getClay)
            OSDhorzdata_pd["sandtotal_r"] = OSDhorzdata_pd["texture_class"].apply(getSand)

    # Initial setup for OSDnarrative_pd dataframe
    base_columns = ["mukey", "cokey", "compname"]
    base_df = mucompdata_pd[base_columns]
    nan_columns = pd.DataFrame(
        np.nan,
        index=np.arange(len(mucompdata_pd.cokey)),
        columns=["series", "brief_narrative"],
    )

    # Conditions for 'STATSGO' or absence of 'OSD_narrative' in 'SSURGO'
    if data_source == "STATSGO" or (data_source == "SSURGO" and not out["OSD_narrative"]):
        OSDnarrative_pd = pd.concat([base_df, nan_columns], axis=1)
    else:
        OSDnarrative_pd = json_normalize(out["OSD_narrative"])
        mucompdata_pd_merge = base_df.copy()
        mucompdata_pd_merge["series"] = mucompdata_pd_merge["compname"].str.replace(r"\d+", "")
        OSDnarrative_pd = pd.merge(mucompdata_pd_merge, OSDnarrative_pd, on="series", how="left")

    # Merge with the main dataframe
    mucompdata_pd = pd.merge(mucompdata_pd, OSDnarrative_pd, on=base_columns, how="left")

    if OSDhorzdata_pd is not None:
        # Replace "NULL" strings with actual NaNs
        OSDhorzdata_pd.replace("NULL", np.nan, inplace=True)

        # Filter rows based on 'cokey' and remove duplicates
        OSDhorzdata_pd = (
            OSDhorzdata_pd[OSDhorzdata_pd["cokey"].isin(comp_key)]
            .drop_duplicates()
            .reset_index(drop=True)
        )

        # Sort dataframe based on a ranking provided by cokey_Index for 'cokey' and then by 'top'
        OSDhorzdata_pd.sort_values(
            by=["cokey", "top"],
            key=lambda col: col.map(cokey_Index) if col.name == "cokey" else col,
            inplace=True,
        )

        # Remove bedrock by filtering out 'R|r' in hzname
        OSDhorzdata_pd = OSDhorzdata_pd[
            ~OSDhorzdata_pd["hzname"].str.contains("R", case=False, na=False)
        ]

        if mucompdata_pd["compkind"].isin(OSD_compkind).any():
            # Group data by cokey
            OSDhorzdata_group_cokey = [
                group for _, group in OSDhorzdata_pd.groupby("cokey", sort=False)
            ]

            # Initialize empty lists
            lab_lyrs = []
            munsell_lyrs = []
            lab_intpl_lyrs = []

            # Define helper functions outside the loop
            def create_new_layer(row, top, bottom):
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
                    "r",
                    "g",
                    "b",
                    "total_frag_volume",
                    "claytotal_r",
                    "sandtotal_r",
                ]:
                    new_row[col] = np.nan
                return new_row

            for index, group in enumerate(OSDhorzdata_group_cokey):
                group_sorted = group.sort_values(by="top").drop_duplicates().reset_index(drop=True)

                # Remove invalid horizons where top depth is greater than bottom depth
                group_sorted = group_sorted[
                    group_sorted["top"] <= group_sorted["bottom"]
                ].reset_index(drop=True)

                if (
                    group_sorted["compkind"].isin(OSD_compkind).any()
                    and group_sorted["bottom"].iloc[-1] != 0
                ):
                    # Check for missing surface horizon
                    if group_sorted["top"].iloc[0] != 0:
                        new_layer = create_new_layer(
                            group_sorted.iloc[0], 0, group_sorted["top"].iloc[0]
                        )
                        group_sorted = pd.concat(
                            [pd.DataFrame([new_layer]), group_sorted], ignore_index=True
                        )

                    # Check for missing subsurface horizons
                    for j in range(len(group_sorted) - 1):
                        if group_sorted["top"].iloc[j + 1] > group_sorted["bottom"].iloc[j]:
                            new_layer = create_new_layer(
                                group_sorted.iloc[j],
                                group_sorted["bottom"].iloc[j],
                                group_sorted["top"].iloc[j + 1],
                            )
                            group_sorted = pd.concat(
                                [
                                    group_sorted.iloc[: j + 1],
                                    pd.DataFrame([new_layer]),
                                    group_sorted.iloc[j + 1 :],
                                ],
                                ignore_index=True,
                            )

                    group[:] = group_sorted

                    # Initialize flags to indicate if OSD depth adjustment is needed
                    OSD_depth_add = False
                    OSD_depth_remove = False

                    # Extract OSD Color Data
                    lab_intpl = getProfileLAB(group_sorted, color_ref)
                    lab_intpl.columns = ["l", "a", "b"]
                    OSD_very_bottom = group_sorted["bottom"].iloc[-1]

                    # subset c_bottom_depth by cokey
                    c_bottom_depths_group = c_bottom_depths[
                        c_bottom_depths["cokey"].isin(group_sorted["cokey"])
                    ]
                    muhorzdata_pd_group = muhorzdata_pd[
                        muhorzdata_pd["cokey"].isin(group_sorted["cokey"])
                    ]

                    # Check if OSD depth adjustment is needed
                    if OSD_very_bottom < c_bottom_depths_group["c_very_bottom"].iloc[0]:
                        OSD_depth_add = True
                        depth_difference = (
                            c_bottom_depths_group["c_very_bottom"].iloc[0] - OSD_very_bottom
                        )
                        lab_values = [
                            lab_intpl.loc[OSD_very_bottom - 1].values.tolist()
                        ] * depth_difference

                        pd_add = pd.DataFrame(lab_values, columns=["l", "a", "b"])

                        # Adjust LAB values for the OSD depth
                        lab_intpl = pd.concat(
                            [lab_intpl.loc[: OSD_very_bottom - 1], pd_add], axis=0
                        ).reset_index(drop=True)
                        OSD_very_bottom = c_bottom_depths_group["c_very_bottom"].iloc[0]

                    elif 0 < c_bottom_depths_group["c_very_bottom"].iloc[0] < OSD_very_bottom:
                        OSD_depth_remove = True

                        # Adjust LAB values for the component depth
                        lab_intpl = lab_intpl.loc[
                            : c_bottom_depths_group["c_very_bottom"].iloc[0]
                        ].reset_index(drop=True)
                        OSD_very_bottom = c_bottom_depths_group["c_very_bottom"].iloc[0]

                    # Set column names for lab_intpl
                    lab_intpl.columns = ["l", "a", "b"]
                    lab_intpl_lyrs.append(lab_intpl)

                    # If all values in lab_intpl are null, append default values to lists
                    if lab_intpl.isnull().values.all():
                        lab_lyrs.append(["", "", ""])
                        munsell_lyrs.append("")
                    else:
                        # Aggregate data for each color dimension
                        l_d = aggregate_data(
                            data=lab_intpl["l"],
                            bottom_depths=muhorzdata_pd_group["hzdepb_r"].tolist(),
                            sd=2,
                        ).fillna("")
                        a_d = aggregate_data(
                            data=lab_intpl["a"],
                            bottom_depths=muhorzdata_pd_group["hzdepb_r"].tolist(),
                            sd=2,
                        ).fillna("")
                        b_d = aggregate_data(
                            data=lab_intpl["b"],
                            bottom_depths=muhorzdata_pd_group["hzdepb_r"].tolist(),
                            sd=2,
                        ).fillna("")

                        # Convert LAB values to a list of triplets
                        lab_parse = [[L, A, B] for L, A, B in zip(l_d, a_d, b_d)]
                        lab_lyrs.append(dict(zip(l_d.index, lab_parse)))

                        # Convert LAB triplets to Munsell values
                        munsell_values = [
                            lab2munsell(color_ref, LAB_ref, LAB=lab)
                            if lab[0] and lab[1] and lab[2]
                            else ""
                            for lab in lab_parse
                        ]
                        munsell_lyrs.append(dict(zip(l_d.index, munsell_values)))
                    # return(jsonify(OSD_text_int  = str(OSD_text_int[index] )))
                    # return(jsonify(group_sorted.to_dict(orient='records')))
                    # Extract OSD Texture and Rock Fragment Data
                    if OSD_text_int[index] == "Yes" or OSD_rfv_int[index] == "Yes":
                        group_sorted[["hzdept_r", "hzdepb_r", "texture"]] = group_sorted[
                            ["top", "bottom", "texture_class"]
                        ]
                        OSD_very_bottom_int, OSD_clay_intpl = getProfile(
                            group_sorted, "claytotal_r", c_bot=True
                        )
                        OSD_clay_intpl.columns = [
                            "c_claypct_intpl",
                            "c_claypct_intpl_grp",
                        ]
                        OSD_sand_intpl = getProfile(group_sorted, "sandtotal_r")
                        OSD_sand_intpl.columns = [
                            "c_sandpct_intpl",
                            "c_sandpct_intpl_grp",
                        ]
                        OSD_rfv_intpl = getProfile(group_sorted, "total_frag_volume")
                        OSD_rfv_intpl.columns = ["c_cfpct_intpl", "c_cfpct_intpl_grp"]

                        # Helper function to update dataframes based on depth conditions
                        def update_intpl_data(df, col_names, values, very_bottom):
                            if OSD_depth_add:
                                layer_add = very_bottom - OSD_very_bottom_int
                                pd_add = pd.DataFrame([values] * layer_add, columns=col_names)
                                df = pd.concat(
                                    [df.loc[: OSD_very_bottom_int - 1], pd_add], axis=0
                                ).reset_index(drop=True)
                            elif OSD_depth_remove:
                                df = df.loc[:very_bottom].reset_index(drop=True)
                            return df

                        # Update data based on depth conditions
                        sand_values = OSD_sand_intpl.iloc[OSD_very_bottom_int - 1].tolist()
                        OSD_sand_intpl = update_intpl_data(
                            OSD_sand_intpl,
                            ["c_sandpct_intpl", "c_sandpct_intpl_grp"],
                            sand_values,
                            OSD_very_bottom,
                        )

                        clay_values = OSD_clay_intpl.iloc[OSD_very_bottom_int - 1].tolist()
                        OSD_clay_intpl = update_intpl_data(
                            OSD_clay_intpl,
                            ["c_claypct_intpl", "c_claypct_intpl_grp"],
                            clay_values,
                            OSD_very_bottom,
                        )

                        rfv_values = OSD_rfv_intpl.iloc[OSD_very_bottom_int - 1].tolist()
                        OSD_rfv_intpl = update_intpl_data(
                            OSD_rfv_intpl,
                            ["c_cfpct_intpl", "c_cfpct_intpl_grp"],
                            rfv_values,
                            OSD_very_bottom,
                        )

                        # If OSD bottom depth is greater than component depth and component depth
                        # is <120cm
                        if OSD_depth_remove:
                            # Remove data based on c_bottom_depths
                            OSD_sand_intpl = OSD_sand_intpl.loc[: c_bottom_depths.iloc[index, 2]]
                            OSD_clay_intpl = OSD_clay_intpl.loc[: c_bottom_depths.iloc[index, 2]]
                            OSD_rfv_intpl = OSD_rfv_intpl.loc[: c_bottom_depths.iloc[index, 2]]

                        # Create the compname and cokey dataframes
                        compname_df = pd.DataFrame(
                            [group_sorted.compname.unique()] * len(OSD_sand_intpl)
                        )
                        cokey_df = pd.DataFrame([group_sorted.cokey.unique()] * len(OSD_sand_intpl))

                        # Concatenate the dataframes
                        group_sorted2 = pd.concat(
                            [
                                OSD_sand_intpl[["c_sandpct_intpl_grp"]],
                                OSD_clay_intpl[["c_claypct_intpl_grp"]],
                                OSD_rfv_intpl[["c_cfpct_intpl_grp"]],
                                compname_df,
                                cokey_df,
                            ],
                            axis=1,
                        )
                        group_sorted2.columns = [
                            "c_sandpct_intpl",
                            "c_claypct_intpl",
                            "c_cfpct_intpl",
                            "compname",
                            "cokey",
                        ]

                        # Update getProfile_mod based on conditions
                        getProfile_mod = getProfile_cokey[index]
                        compname_check = (
                            getProfile_mod["compname"]
                            .isin(group_sorted2[["compname"]].iloc[0])
                            .any()
                        )

                        if (
                            compname_check
                            and OSD_text_int[index] == "Yes"
                            and not group_sorted2["c_sandpct_intpl"].isnull().all()
                        ):
                            getProfile_mod["sandpct_intpl"] = group_sorted2["c_sandpct_intpl"]
                            getProfile_mod["claypct_intpl"] = group_sorted2["c_claypct_intpl"]

                        if (
                            compname_check
                            and OSD_rfv_int[index] == "Yes"
                            and not group_sorted2["c_cfpct_intpl"].isnull().all()
                        ):
                            getProfile_mod["rfv_intpl"] = group_sorted2["c_cfpct_intpl"]

                        getProfile_cokey[index] = getProfile_mod

                        # Aggregate sand data
                        snd_d_osd, hz_depb_osd = agg_data_layer(
                            data=OSD_sand_intpl.iloc[:, 0],
                            bottom=OSD_very_bottom,
                            depth=True,
                        )

                        # Aggregate clay data
                        cly_d_osd = agg_data_layer(
                            data=OSD_clay_intpl.iloc[:, 1], bottom=OSD_very_bottom
                        )

                        # Calculate texture data based on sand and clay data
                        txt_d_osd = [
                            getTexture(row=None, sand=s, silt=(100 - (s + c)), clay=c)
                            for s, c in zip(snd_d_osd, cly_d_osd)
                        ]
                        txt_d_osd = pd.Series(txt_d_osd, index=snd_d_osd.index)

                        # Aggregate rock fragment data
                        rf_d_osd = agg_data_layer(
                            data=OSD_rfv_intpl.c_cfpct_intpl, bottom=OSD_very_bottom
                        )

                        # Fill NaN values
                        snd_d_osd.fillna("", inplace=True)
                        cly_d_osd.fillna("", inplace=True)
                        txt_d_osd.fillna("", inplace=True)
                        rf_d_osd.fillna("", inplace=True)
                        hz_depb_osd.fillna("", inplace=True)

                        # Store aggregated data in dictionaries based on conditions
                        if OSD_text_int[index] == "Yes":
                            snd_lyrs[index] = snd_d_osd.to_dict()
                            cly_lyrs[index] = cly_d_osd.to_dict()
                            txt_lyrs[index] = txt_d_osd.to_dict()

                        if OSD_rfv_int[index] == "Yes":
                            rf_lyrs[index] = rf_d_osd.to_dict()

                        # Update horizon layers if bottom depth is zero
                        if c_bottom_depths.iloc[index, 2] == 0:
                            hz_lyrs[index] = hz_depb_osd.to_dict()
                            mucompdata_pd.loc[index, "c_very_bottom"] = OSD_very_bottom

                        # Update cec, ph, and ec layers if they contain only a single empty string
                        for lyr in [cec_lyrs, ph_lyrs, ec_lyrs]:
                            if len(lyr[index]) == 1 and lyr[index][0] == "":
                                lyr[index] = dict(zip(hz_depb_osd.index, [""] * len(hz_depb_osd)))

                else:
                    OSDhorzdata_group_cokey[index] = group_sorted

                    # Create an empty dataframe with NaNs for lab_intpl
                    lab_intpl = pd.DataFrame(
                        np.nan,
                        index=np.arange(c_bottom_depths.iloc[index, 2]),
                        columns=["l", "a", "b"],
                    )
                    lab_intpl_lyrs.append(lab_intpl)

                    # Create dummy data for lab_lyrs
                    lab_dummy = [["", "", ""] for _ in range(len(hz_lyrs[index]))]
                    lab_lyrs.append(dict(zip(hz_lyrs[index].keys(), lab_dummy)))

                    # Create dummy data for munsell_lyrs
                    munsell_dummy = [""] * len(hz_lyrs[index])
                    munsell_lyrs.append(dict(zip(hz_lyrs[index].keys(), munsell_dummy)))

            # Series URL Generation
            # Initialize lists to store series URLs
            SDE_URL = []
            SEE_URL = []

            # Group data by 'cokey'
            OSDhorzdata_group_cokey = [g for _, g in OSDhorzdata_pd.groupby("cokey", sort=False)]

            for index, group in enumerate(OSDhorzdata_group_cokey):
                # Check if compkind is not in OSD_compkind or if series contains any null values
                if (
                    mucompdata_pd.loc[index]["compkind"] not in OSD_compkind
                    or group["series"].isnull().any()
                ):
                    SDE_URL.append("")
                    SEE_URL.append("")
                else:
                    # Extract compname, convert to lowercase, remove trailing numbers, and replace
                    # spaces with underscores
                    comp = group["compname"].iloc[0].lower()
                    comp = re.sub(r"\d+$", "", comp)
                    comp = comp.replace(" ", "_")

                    # Create and append URLs
                    SDE_URL.append(f"https://casoilresource.lawr.ucdavis.edu/sde/?series={comp}")
                    SEE_URL.append(f"https://casoilresource.lawr.ucdavis.edu/see/#{comp}")

        else:
            # Initialize lists to store data layers and URLs
            lab_lyrs = []
            lab_intpl_lyrs = []
            munsell_lyrs = []
            SDE_URL = []
            SEE_URL = []

            # Iterate over each entry in mucompdata_pd
            for i in range(len(mucompdata_pd)):
                # Initialize a DataFrame filled with NaNs
                lab_intpl = pd.DataFrame(
                    np.nan,
                    index=np.arange(c_bottom_depths.iloc[i, 2]),
                    columns=["l", "a", "b"],
                )
                lab_intpl_lyrs.append(lab_intpl)

                # Create dummy data for lab and munsell layers
                keys = list(hz_lyrs[i].keys())
                lab_dummy = [{"", "", ""} for _ in range(len(keys))]
                munsell_dummy = [""] * len(keys)

                # Append dummy data to lists
                lab_lyrs.append(dict(zip(keys, lab_dummy)))
                munsell_lyrs.append(dict(zip(keys, munsell_dummy)))

                # Append empty URLs
                SDE_URL.append("")
                SEE_URL.append("")

    else:
        # Initialize lists to store data layers and URLs
        lab_lyrs = []
        lab_intpl_lyrs = []
        munsell_lyrs = []
        SDE_URL = []
        SEE_URL = []

        # Iterate over each entry in mucompdata_pd
        for i in range(len(mucompdata_pd)):
            # Initialize a DataFrame filled with NaNs
            lab_intpl = pd.DataFrame(
                np.nan,
                index=np.arange(c_bottom_depths.iloc[i, 2]),
                columns=["l", "a", "b"],
            )
            lab_intpl_lyrs.append(lab_intpl)

            # Create dummy data for lab and munsell layers
            keys = list(hz_lyrs[i].keys())
            lab_dummy = [{"", "", ""} for _ in range(len(keys))]
            munsell_dummy = [""] * len(keys)

            # Append dummy data to lists
            lab_lyrs.append(dict(zip(keys, lab_dummy)))
            munsell_lyrs.append(dict(zip(keys, munsell_dummy)))

            # Append empty URLs
            SDE_URL.append("")
            SEE_URL.append("")

    # Subset datasets to exclude pedons without any depth information
    cokeys_with_depth = mucompdata_pd[mucompdata_pd["c_very_bottom"] > 0].cokey.unique()

    # If there are cokeys with depth
    if len(cokeys_with_depth) > 0:
        # Subset based on cokeys with depth data
        mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(cokeys_with_depth)].reset_index(
            drop=True
        )
        muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(cokeys_with_depth)].reset_index(
            drop=True
        )
        c_bottom_depths = c_bottom_depths[
            c_bottom_depths["cokey"].isin(cokeys_with_depth)
        ].reset_index(drop=True)

        # Update comp_key and cokey_Index
        comp_key = mucompdata_pd["cokey"].unique().tolist()
        cokey_Index = dict(zip(comp_key, range(len(comp_key))))

        # Get indices of components with depth data to exclude profiles with no depth information
        indices_with_depth = mucompdata_pd.index.tolist()

        # Subset layers based on indices with depth
        layer_lists = [
            lab_intpl_lyrs,
            getProfile_cokey,
            snd_lyrs,
            cly_lyrs,
            txt_lyrs,
            rf_lyrs,
            cec_lyrs,
            ph_lyrs,
            ec_lyrs,
            hz_lyrs,
            lab_lyrs,
            munsell_lyrs,
        ]

        for i, lst in enumerate(layer_lists):
            layer_lists[i] = [lst[index] for index in indices_with_depth]

        # Unpack the layer lists
        (
            lab_intpl_lyrs,
            getProfile_cokey,
            snd_lyrs,
            cly_lyrs,
            txt_lyrs,
            rf_lyrs,
            cec_lyrs,
            ph_lyrs,
            ec_lyrs,
            hz_lyrs,
            lab_lyrs,
            munsell_lyrs,
        ) = layer_lists

    # Create a new column 'soilID_rank' which will be True for the first row in each group sorted
    # by 'distance' and False for other rows
    mucompdata_pd = mucompdata_pd.sort_values(["compname_grp", "distance"])
    mucompdata_pd["soilID_rank"] = ~mucompdata_pd.duplicated("compname_grp", keep="first")

    # Assign the minimum distance for each group to a new column 'min_dist'
    mucompdata_pd["min_dist"] = mucompdata_pd.groupby("compname_grp")["distance"].transform("first")

    mucompdata_pd = mucompdata_pd.reset_index(drop=True)

    # ---------------------------------------------------------------------------------------------
    # Ecological site
    if data_source == "SSURGO":
        if not out["ESD"]:
            ESDcompdata_pd = None
        else:
            ESD = pd.json_normalize(out["ESD"])
            ESD[["cokey", "ecoclassid", "ecoclassname"]] = ESD[
                ["cokey", "ecoclassid", "ecoclassname"]
            ].astype(str)

            # Check if any cokey in ESD matches with mucompdata_pd
            if any(ESD["cokey"].isin(mucompdata_pd["cokey"])):
                ESDcompdata_pd = pd.merge(
                    mucompdata_pd[["mukey", "cokey", "compname"]],
                    ESD,
                    on="cokey",
                    how="left",
                )
            else:
                ESDcompdata_pd = None

    elif data_source == "STATSGO":
        ESDcompdataQry = (
            "SELECT cokey,  coecoclasskey, ecoclassname FROM coecoclass WHERE cokey IN ("
            + ",".join(map(str, comp_key))
            + ")"
        )
        ESDcompdata_out = sda_return(propQry=ESDcompdataQry)

        if not ESDcompdata_out:
            ESDcompdata_pd = None
        else:
            ESDcompdata_sda = pd.DataFrame(
                ESDcompdata_out["Table"][1:], columns=ESDcompdata_out["Table"][0]
            )
            ESDcompdata_sda[["cokey", "ecoclassid", "ecoclassname"]] = ESDcompdata_sda[
                ["cokey", "ecoclassid", "ecoclassname"]
            ].astype(str)

            # Check if any cokey in ESDcompdata_sda matches with mucompdata_pd
            if any(ESDcompdata_sda["cokey"].isin(mucompdata_pd["cokey"])):
                ESDcompdata_pd = pd.merge(
                    mucompdata_pd[["mukey", "cokey", "compname"]],
                    ESDcompdata_sda,
                    on="cokey",
                    how="left",
                )
            else:
                ESDcompdata_pd = None

    if ESDcompdata_pd is not None:
        # Clean and process the dataframe
        ESDcompdata_pd = ESDcompdata_pd.replace("NULL", np.nan)
        ESDcompdata_pd = ESDcompdata_pd.drop_duplicates(keep="first").reset_index(drop=True)
        ESDcompdata_pd = ESDcompdata_pd[ESDcompdata_pd["cokey"].isin(comp_key)]
        ESDcompdata_pd["Comp_Rank"] = ESDcompdata_pd["cokey"].map(cokey_Index)
        ESDcompdata_pd.sort_values(["Comp_Rank"], ascending=True, inplace=True)
        ESDcompdata_pd.drop(columns="Comp_Rank", inplace=True)

        # Update ecoclassid based on MLRA update by querying 'ESD_class_synonym_list' table
        ecositeID = ESDcompdata_pd["ecoclassid"].dropna().tolist()

        # old code
        ESD_geo = []
        ESD_geo.extend(ecositeID)
        ESD_geo = [ESD_geo for ESD_geo in ESD_geo if str(ESD_geo) != "nan"]
        ESD_geo = ESD_geo[0][1:5]

        class_url = "https://edit.jornada.nmsu.edu/services/downloads/esd/%s/class-list.json" % (
            ESD_geo
        )

        try:
            response = requests.get(class_url, timeout=4)
            response.raise_for_status()  # Raise an exception for any HTTP error

            ESD_list = response.json()
            ESD_list_pd = json_normalize(ESD_list["ecoclasses"])[["id", "legacyId"]]
            ESD_URL = []

            if isinstance(ESD_list, list):
                ESD_URL.append("")
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
                        ES_URL_t = "https://edit.jornada.nmsu.edu/catalogs/esd/%s/%s" % (
                            ESD_geo,
                            ecosite_edit_id,
                        )
                        ESD_URL.append(ES_URL_t)
                    else:
                        ESD_URL.append("")

            ESDcompdata_pd = ESDcompdata_pd.assign(esd_url=ESD_URL)
        except requests.exceptions.RequestException as err:
            ESDcompdata_pd["esd_url"] = pd.Series(np.repeat("", len(ecositeID))).values
            print("An error occurred:", err)

        # Assign missing ESD for components that have other instances with an assigned ESD
        if ESDcompdata_pd is not None:
            if (
                ESDcompdata_pd.ecoclassid.isnull().any()
                or ESDcompdata_pd.ecoclassname.isnull().any()
            ):
                ESDcompdata_pd["compname_grp"] = ESDcompdata_pd.compname.str.replace(r"[0-9]+", "")
                ESDcompdata_pd_comp_grps = [
                    g for _, g in ESDcompdata_pd.groupby(["compname_grp"], sort=False)
                ]
                ecoList_out = []
                for i in range(len(ESDcompdata_pd_comp_grps)):
                    comp_grps_temp = ESDcompdata_pd_comp_grps[i]
                    if len(comp_grps_temp) == 1:
                        ecoList_out.append(comp_grps_temp)
                    elif (
                        comp_grps_temp.ecoclassid.isnull().all()
                        or comp_grps_temp.ecoclassname.isnull().all()
                    ):
                        ecoList_out.append(comp_grps_temp)
                    elif (
                        comp_grps_temp.ecoclassid.isnull().any()
                        and len(comp_grps_temp.ecoclassid.dropna().unique()) == 1
                    ) and (
                        comp_grps_temp.ecoclassname.isnull().any()
                        and len(comp_grps_temp.ecoclassname.dropna().unique()) == 1
                    ):
                        comp_grps_temp["ecoclassid"] = pd.Series(
                            np.tile(
                                comp_grps_temp.ecoclassid.dropna().unique().tolist(),
                                len(comp_grps_temp),
                            )
                        ).values
                        comp_grps_temp["ecoclassname"] = pd.Series(
                            np.tile(
                                comp_grps_temp.ecoclassname.dropna().unique().tolist(),
                                len(comp_grps_temp),
                            )
                        ).values
                        url = comp_grps_temp.esd_url.unique().tolist()
                        url = [x for x in url if x != ""]
                        if not url:
                            comp_grps_temp["esd_url"] = pd.Series(
                                np.tile("", len(comp_grps_temp))
                            ).values
                        else:
                            comp_grps_temp["esd_url"] = pd.Series(
                                np.tile(url, len(comp_grps_temp))
                            ).values
                        ecoList_out.append(comp_grps_temp)
                    else:
                        ecoList_out.append(comp_grps_temp)
                ESDcompdata_pd = pd.concat(ecoList_out)

        ESDcompdata_group_cokey = [g for _, g in ESDcompdata_pd.groupby(["cokey"], sort=False)]
        esd_comp_list = []
        for i in range(len(ESDcompdata_group_cokey)):
            if ESDcompdata_group_cokey[i]["ecoclassname"].isnull().values.any():
                esd_comp_list.append({"ESD": {"ecoclassid": "", "ecoclassname": "", "esd_url": ""}})
            else:
                esd_comp_list.append(
                    {
                        "ESD": {
                            "ecoclassid": ESDcompdata_group_cokey[i]["ecoclassid"].tolist(),
                            "ecoclassname": ESDcompdata_group_cokey[i]["ecoclassname"].tolist(),
                            "esd_url": ESDcompdata_group_cokey[i]["esd_url"].tolist(),
                        }
                    }
                )
        else:
            esd_comp_list = []
            for i in range(len(mucompdata_pd)):
                esd_comp_list.append({"ESD": {"ecoclassid": "", "ecoclassname": "", "esd_url": ""}})

        # Add ecosite data to mucompdata_pd for testing output. In cases with multiple ESDs per
        # component, only take the first.
        if ESDcompdata_pd is None:
            mucompdata_pd["ecoclassid"] = pd.Series(np.tile(np.nan, len(mucompdata_pd))).values
            mucompdata_pd["ecoclassname"] = pd.Series(np.tile(np.nan, len(mucompdata_pd))).values
        else:
            mucompdata_pd = pd.merge(
                mucompdata_pd,
                ESDcompdata_pd[["cokey", "ecoclassid", "ecoclassname"]]
                .drop_duplicates("cokey", keep="first")
                .reset_index(drop=True),
                on="cokey",
                how="left",
            )

    # --------------------------------------------------------------------------------------------
    # SoilIDList output

    # ------------------------------------------------------------
    # Define model version
    model_version = "3.0"
    # Create the soilIDRank_output list by combining the dataframes of various data sources
    soilIDRank_output = [
        pd.concat(
            [
                getProfile_cokey[i][["compname", "sandpct_intpl", "claypct_intpl", "rfv_intpl"]],
                lab_intpl_lyrs[i],
            ],
            axis=1,
        )
        for i in range(len(getProfile_cokey))
    ]

    # Convert the list to a DataFrame and reset the index
    soilIDRank_output_pd = pd.concat(soilIDRank_output).reset_index(drop=True)

    # Sort mucompdata_pd based on normalized distance score in descending order
    mucompdata_cond_prob = mucompdata_pd.sort_values(
        "distance_score_norm", ascending=False
    ).reset_index(drop=True)

    # Generate the Rank_Loc column values
    rank_id = 1
    Rank_Loc = []
    for rank, displayed in mucompdata_cond_prob["soilID_rank"].items():
        if displayed:
            Rank_Loc.append(str(rank_id))
            rank_id += 1
        else:
            Rank_Loc.append("Not Displayed")

    mucompdata_cond_prob["Rank_Loc"] = Rank_Loc

    # ------------------------------------------------------------

    # Sort mucompdata_cond_prob by soilID_rank and distance_score_norm
    mucompdata_cond_prob = mucompdata_cond_prob.sort_values(
        ["soilID_rank", "distance_score_norm"], ascending=[False, False]
    )
    mucomp_index = mucompdata_cond_prob.index

    # Generate the ID list
    ID = [
        {
            "name": site.capitalize(),
            "component": comp.capitalize(),
            "score_loc": sc,
            "rank_loc": rank,
        }
        for site, comp, sc, rank in zip(
            mucompdata_cond_prob["compname"],
            mucompdata_cond_prob["compname_grp"],
            mucompdata_cond_prob["distance_score_norm"].round(3),
            mucompdata_cond_prob["Rank_Loc"],
        )
    ]

    # Replace NaN values with an empty string
    mucompdata_cond_prob = mucompdata_cond_prob.fillna("")

    # Generate the Site list
    Site = [
        {
            "siteData": {
                "mapunitID": row["mukey"],
                "componentID": row["cokey"],
                "componentKind": row["compkind"],
                "dataSource": row["data_source"],
                "textureInfill": row["OSD_text_int"],
                "rfvInfill": row["OSD_rfv_int"],
                "componentPct": row["comppct_r"],
                "distance": row["distance"],
                "minCompDistance": row["min_dist"],
                "slope": row["slope_r"],
                "nirrcapcl": row["nirrcapcl"],
                "nirrcapscl": row["nirrcapscl"],
                "nirrcapunit": row["nirrcapunit"],
                "irrcapcl": row["irrcapcl"],
                "irrcapscl": row["irrcapscl"],
                "irrcapunit": row["irrcapunit"],
                "taxsubgrp": row["taxsubgrp"],
                "sdeURL": SDE_URL[idx],
                "seeURL": SEE_URL[idx],
            },
            "siteDescription": row["brief_narrative"],
        }
        for idx, row in mucompdata_cond_prob.iterrows()
    ]

    # Reordering lists using list comprehension and mucomp_index
    lists_to_reorder = [
        esd_comp_list,
        hz_lyrs,
        snd_lyrs,
        cly_lyrs,
        txt_lyrs,
        rf_lyrs,
        cec_lyrs,
        ph_lyrs,
        ec_lyrs,
        lab_lyrs,
        munsell_lyrs,
    ]
    reordered_lists = [[lst[i] for i in mucomp_index] for lst in lists_to_reorder]

    # Destructuring reordered lists for clarity
    (
        esd_comp_list,
        hz_lyrs,
        snd_lyrs,
        cly_lyrs,
        txt_lyrs,
        rf_lyrs,
        cec_lyrs,
        ph_lyrs,
        ec_lyrs,
        lab_lyrs,
        munsell_lyrs,
    ) = reordered_lists

    # Generating output_SoilList
    output_SoilList = [
        dict(
            zip(
                [
                    "id",
                    "site",
                    "esd",
                    "bottom_depth",
                    "sand",
                    "clay",
                    "texture",
                    "rock_fragments",
                    "cec",
                    "ph",
                    "ec",
                    "lab",
                    "munsell",
                ],
                row,
            )
        )
        for row in zip(
            ID,
            Site,
            esd_comp_list,
            hz_lyrs,
            snd_lyrs,
            cly_lyrs,
            txt_lyrs,
            rf_lyrs,
            cec_lyrs,
            ph_lyrs,
            ec_lyrs,
            lab_lyrs,
            munsell_lyrs,
        )
    ]

    # Writing out list of data needed for soilIDRank
    if plot_id is None:
        soilIDRank_output_pd.to_csv(
            "%s/soilIDRank_ofile1.csv" % current_app.config["DATA_BACKEND"],
            index=None,
            header=True,
        )
        mucompdata_cond_prob.to_csv(
            "%s/soilIDRank_ofile2.csv" % current_app.config["DATA_BACKEND"],
            index=None,
            header=True,
        )
    else:
        output_data = json.dumps(
            {
                "metadata": {
                    "location": "us",
                    "model": "v3",
                    "unit_measure": {
                        "distance": "m",
                        "depth": "cm",
                        "cec": "cmol(c)/kg",
                        "clay": "%",
                        "rock_fragments": "cm3/100cm3",
                        "sand": "%",
                        "ec": "ds/m",
                    },
                },
                "soilList": output_SoilList,
            }
        )
        save_model_output(
            plot_id,
            model_version,
            output_data,
            soilIDRank_output_pd.to_csv(index=None, header=True),
            mucompdata_cond_prob.to_csv(index=None, header=True),
        )

    # Return the final output
    return {
        "metadata": {
            "location": "us",
            "model": "v3",
            "unit_measure": {
                "distance": "m",
                "depth": "cm",
                "cec": "cmol(c)/kg",
                "clay": "%",
                "rock_fragments": "cm3/100cm3",
                "sand": "%",
                "ec": "ds/m",
            },
        },
        "soilList": output_SoilList,
    }


##############################################################################################
#                                   rankPredictionUS                                         #
##############################################################################################
def rankPredictionUS(
    lon,
    lat,
    soilHorizon,
    horizonDepth,
    rfvDepth,
    lab_Color,
    pSlope,
    pElev,
    bedrock,
    cracks,
    plot_id=None,
):
    """
    TODO: Future testing to see if deltaE2000 values should be incorporated
    into site data use 'getColor_deltaE2000_OSD_pedon' and helper functions
    located in local_functions_SoilID_v3.py
    """
    # ---------------------------------------------------------------------------------------
    # ------ Load in user data --------#
    # Initialize the DataFrame from the input data
    soil_df = pd.DataFrame(
        {
            "soilHorizon": soilHorizon,
            "horizonDepth": horizonDepth,
            "rfvDepth": rfvDepth,
            "lab_Color": lab_Color,
        }
    )

    # Drop rows where all values are NaN
    soil_df.dropna(how="all", inplace=True)

    # Set the bottom of each horizon
    soil_df["bottom"] = soil_df["horizonDepth"]

    # Replace NaNs with None for consistency
    soil_df.fillna(value=None, inplace=True)

    # Calculate the top depth for each horizon
    soil_df["top"] = [0] + soil_df["horizonDepth"].iloc[:-1].tolist()

    # Adjust the bottom depth based on bedrock depth
    if bedrock is not None and soil_df["bottom"].iloc[-1] > bedrock:
        last_valid_index = soil_df.loc[soil_df["bottom"] <= bedrock].index[-1]
        soil_df = soil_df.loc[:last_valid_index]
        soil_df["bottom"].iloc[-1] = bedrock

    # Drop the original horizonDepth column
    soil_df.drop(columns=["horizonDepth"], inplace=True)

    # Filter out rows without valid horizon data
    relevant_columns = ["soilHorizon", "rfvDepth", "lab_Color"]
    soil_df_slice = soil_df.dropna(how="all", subset=relevant_columns)

    if soil_df_slice.empty:
        soil_df_slice = None

    if soil_df_slice is not None:
        soil_df_slice = soil_df_slice.reset_index(drop=True)

        # Create index list of soil slices where user data exists
        pedon_slice_index = [
            j
            for i in range(len(soil_df_slice))
            for j in range(int(soil_df_slice.top[i]), int(soil_df_slice.bottom[i]))
        ]
        pedon_slice_index = [x for x in pedon_slice_index if x < 120]

        if bedrock is not None:
            pedon_slice_index.extend(range(bedrock, 120))

        # Convert soil properties to lists
        soilHorizon = soil_df.soilHorizon.tolist()
        rfvDepth = soil_df.rfvDepth.tolist()
        horizonDepthB = [int(x) for x in soil_df.bottom.tolist()]
        horizonDepthT = [int(x) for x in soil_df.top.tolist()]
        lab_Color = soil_df.lab_Color

        # Generate user specified percent clay, sand, and rfv distributions
        spt = [getSand(sh) for sh in soilHorizon]
        cpt = [getClay(sh) for sh in soilHorizon]
        p_cfg = [getCF_fromClass(rf) for rf in rfvDepth]

        p_sandpct_intpl = [
            spt[i]
            for i in range(len(soilHorizon))
            for _ in range(horizonDepthT[i], horizonDepthB[i])
        ]
        p_claypct_intpl = [
            cpt[i]
            for i in range(len(soilHorizon))
            for _ in range(horizonDepthT[i], horizonDepthB[i])
        ]
        p_cfg_intpl = [
            p_cfg[i]
            for i in range(len(soilHorizon))
            for _ in range(horizonDepthT[i], horizonDepthB[i])
        ]

        # Length of interpolated texture and RF depth
        p_bottom_depth = pd.DataFrame([-999, "sample_pedon", soil_df_slice.bottom.iloc[-1]]).T
        p_bottom_depth.columns = ["cokey", "compname", "bottom_depth"]

        # Pedon color data
        if not lab_Color.isnull().all():
            lab_Color = [[np.nan, np.nan, np.nan] if x is None else x for x in lab_Color]
            p_lab_intpl = [
                lab_Color.iloc[i]
                for i in range(len(lab_Color))
                for _ in range(horizonDepthT[i], horizonDepthB[i])
            ]
            p_lab_intpl = pd.DataFrame(p_lab_intpl).reset_index(drop=True)
        else:
            lab_Color = pd.DataFrame([x for x in lab_Color if x is not None])
            p_lab_intpl = pd.DataFrame(
                np.nan, index=np.arange(120), columns=np.arange(3)
            ).reset_index(drop=True)

        def adjust_depth_interval(data, target_length=120, add_columns=1):
            """Adjusts the depth interval of user data."""
            length = len(data)
            if length > target_length:
                data = data.iloc[:target_length]
            elif length < target_length:
                add_length = target_length - length
                if add_columns == 1:
                    add_data = pd.Series(np.nan, index=np.arange(add_length))
                else:
                    add_data = pd.DataFrame(
                        np.nan, index=np.arange(add_length), columns=np.arange(add_columns)
                    )
                data = pd.concat([data, add_data])
            return data.reset_index(drop=True)

        # Adjust depth interval for each dataset
        p_sandpct_intpl = adjust_depth_interval(p_sandpct_intpl)
        p_claypct_intpl = adjust_depth_interval(p_claypct_intpl)
        p_cfg_intpl = adjust_depth_interval(p_cfg_intpl)
        p_lab_intpl = adjust_depth_interval(p_lab_intpl, add_columns=3)

        # Construct final dataframe with adjusted data
        p_compname = pd.Series("sample_pedon", index=np.arange(len(p_sandpct_intpl)))
        p_hz_data = pd.concat(
            [p_compname, p_sandpct_intpl, p_claypct_intpl, p_cfg_intpl, p_lab_intpl], axis=1
        )
        p_hz_data.columns = [
            "compname",
            "sandpct_intpl",
            "claypct_intpl",
            "rfv_intpl",
            "l",
            "a",
            "b",
        ]

        # Clean up the final data
        p_hz_data = p_hz_data.loc[:, ~p_hz_data.isnull().all()]
        p_hz_data = p_hz_data[p_hz_data.index.isin(pedon_slice_index)]

        # List of user entered variables
        p_hz_data_names = p_hz_data.columns.tolist()

    else:
        # Initialize primary data containers to default or empty values
        p_hz_data = None

        # Initialize interpolated soil property lists
        p_sandpct_intpl = []
        p_claypct_intpl = []
        p_cfg_intpl = []

        # Initialize lab interpolation data with NaNs
        p_lab_intpl = pd.DataFrame(np.nan, index=np.arange(1), columns=np.arange(3))

        # Set default bottom depth data
        p_bottom_depth = pd.DataFrame([-999, "sample_pedon", 0]).T
        p_bottom_depth.columns = ["cokey", "compname", "bottom_depth"]

    if pElev is None:
        try:
            # Construct the URL for fetching elevation data based on latitude and longitude
            elev_url = (
                f"https://nationalmap.gov/epqs/pqs.php?x={lon}&y={lat}&units=Meters&output=json"
            )

            # Fetch data from the URL
            response = requests.get(elev_url, timeout=2)
            elev_data = response.json()

            # Extract elevation value and round it to 3 decimal places
            pElev = round(
                elev_data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"], 3
            )
        except Exception as err:
            print(f"Error fetching elevation data: {err}")
            pElev = None

    # Generate data completeness score
    def compute_completeness(length, thresholds, scores):
        for thres, score in zip(thresholds, scores):
            if length <= thres:
                return score
        return scores[-1]

    # Compute text completeness
    text_len = len(p_sandpct_intpl.dropna())
    text_thresholds = [1, 10, 20, 50, 70, 100]
    text_scores = [3, 8, 15, 25, 30, 35, 40]
    text_comp = compute_completeness(text_len, text_thresholds, text_scores)

    # Compute rf completeness
    rf_len = len(p_cfg_intpl.dropna())
    rf_thresholds = [1, 10, 20, 50, 70, 100, 120]
    rf_scores = [3, 6, 10, 15, 20, 23, 25]
    rf_comp = compute_completeness(rf_len, rf_thresholds, rf_scores)

    # Compute lab completeness
    lab_len = len(p_lab_intpl.dropna())
    lab_thresholds = [1, 10, 20, 50, 70, 100, 120]
    lab_scores = [1, 3, 6, 9, 12, 14, 15]
    lab_comp = compute_completeness(lab_len, lab_thresholds, lab_scores)

    # Compute slope and crack completeness
    slope_comp = 15 if pSlope is not None else 0
    crack_comp = 5 if cracks is not None else 0

    # Compute total data completeness
    data_completeness = slope_comp + crack_comp + text_comp + rf_comp + lab_comp

    # Generate completeness message
    missing_data = []
    if text_comp < 40:
        missing_data.append("soil texture")
    if rf_comp < 25:
        missing_data.append("soil rock fragments")
    if lab_comp < 15:
        missing_data.append("soil color")
    if slope_comp < 15:
        missing_data.append("slope")
    if crack_comp < 5:
        missing_data.append("soil cracking")

    if missing_data:
        missing_text = ", ".join(missing_data)
        text_completeness = (
            f"To improve predictions, complete data entry for: {missing_text} and re-sync."
        )
    else:
        text_completeness = "SoilID data entry for this site is complete."

    # -------------------------------------------------------------------------------------------
    # Load in component data from soilIDList
    if plot_id is None:
        # Reading from file
        soilIDRank_output_pd = pd.read_csv(
            f"{current_app.config['DATA_BACKEND']}/soilIDRank_ofile1.csv"
        )
        mucompdata_pd = pd.read_csv(f"{current_app.config['DATA_BACKEND']}/soilIDRank_ofile2.csv")
        record_id = None
    else:
        # Read from database
        modelRun = load_model_output(plot_id)
        if modelRun:
            record_id = modelRun[0]
            soilIDRank_output_pd = pd.read_csv(io.StringIO(modelRun[2]))
            mucompdata_pd = pd.read_csv(io.StringIO(modelRun[3]))
        else:
            print("Cannot find a plot with this ID")

    # Modify mucompdata_pd DataFrame
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
    mucompdata_pd["nirrcapcl"] = mucompdata_pd["nirrcapcl"].apply(trim_fraction)
    mucompdata_pd["irrcapcl"] = mucompdata_pd["irrcapcl"].apply(trim_fraction)

    # Create soil depth DataFrame and subset component depths based on max user
    # depth if no bedrock specified
    c_bottom_depths = mucompdata_pd[["cokey", "compname", "c_very_bottom"]]
    c_bottom_depths.columns = ["cokey", "compname", "bottom_depth"]
    slices_of_soil = pd.concat([p_bottom_depth, c_bottom_depths], axis=0).reset_index(drop=True)
    compnames = mucompdata_pd[["compname", "compname_grp"]]

    # Generate a matrix storing a flag describing soil (1) vs. non-soil (0) at each slice
    # note that this will truncate a profile to the max depth of actual data

    # Determine the maximum depth based on bedrock and user input
    if bedrock is None:
        max_depth = min(p_bottom_depth.bottom_depth.values[0], 200)
    else:
        max_depth = 200

    # Truncate depths in slices_of_soil that exceed max_depth
    slices_of_soil.loc[slices_of_soil.bottom_depth > max_depth, "bottom_depth"] = max_depth

    # Initialize the soil_matrix
    soil_matrix = pd.DataFrame(
        np.nan, index=np.arange(max_depth), columns=np.arange(len(slices_of_soil))
    )

    # Populate the soil_matrix
    for i, bottom_depth in enumerate(slices_of_soil.bottom_depth):
        # Check if the bottom_depth is less than max_depth
        if bottom_depth < max_depth:
            soil_matrix.iloc[:bottom_depth, i] = 1
            soil_matrix.iloc[bottom_depth:, i] = 0
        else:
            soil_matrix.iloc[:, i] = 1

    # Check if user has entered horizon data
    if not p_hz_data or p_bottom_depth.bottom_depth.le(0).any():
        soilIDRank_output_pd = None
    else:
        # Subset component soil properties to match user measured properties
        soilIDRank_output_pd = soilIDRank_output_pd[p_hz_data_names]

        # Subset soil_matrix to user measured slices
        soil_matrix = soil_matrix.loc[pedon_slice_index]

    # Horizon Data Similarity
    if soilIDRank_output_pd is not None:
        groups = [group for _, group in soilIDRank_output_pd.groupby(["compname"], sort=False)]

        Comp_Rank_Status = []
        Comp_Missing_Status = []
        Comp_name = []

        for group in groups:
            data_subgroup = group[p_hz_data_names].drop("compname", 1)
            all_nan = data_subgroup.isnull().all().all()
            any_nan = data_subgroup.isnull().all().any()

            if all_nan:
                Comp_Rank_Status.append("Not Ranked")
                Comp_Missing_Status.append("No Data")
            elif any_nan:
                Comp_Rank_Status.append("Ranked")
                Comp_Missing_Status.append("Missing Data")
            else:
                Comp_Rank_Status.append("Ranked")
                Comp_Missing_Status.append("Data Complete")

            Comp_name.append(group["compname"].unique()[0])

        # Consolidate the results into a DataFrame
        Rank_Filter = pd.DataFrame(
            {
                "compname": Comp_name,
                "rank_status": Comp_Rank_Status,
                "missing_status": Comp_Missing_Status,
            }
        )

        # Horizon Data Matrix

        # Subset depth intervals to match user measured intervals
        horz_vars = [p_hz_data]
        horz_vars.extend([group.reset_index(drop=True).loc[pedon_slice_index] for group in groups])

        # Calculate similarity for each depth slice
        dis_mat_list = []
        for (
            i
        ) in (
            soil_matrix.index
        ):  # i should be an index of p_hz_data depth slices, e.g. if user only enters
            # 100-120cm data, then i = 100:120
            slice_data = [horz.loc[i] for horz in horz_vars]
            sliceT = pd.concat(slice_data, axis=1).T
            """
            Not all depth slices have the same user recorded data. Here we filter out data columns
            with missing data and use that to subset the component data.
            If certain components are missing lots of data and the filering results in less than 2
            soil properties, than we filter out data columns with missing USER data and components
            with missing data will later be assigned the max dissimilarity across all horizons
            """
            # Filter columns based on available data
            if i < bedrock:
                sample_pedon_slice_vars = (
                    sliceT.dropna(axis="columns").drop(["compname"], 1).columns.tolist()
                )
                if len(sample_pedon_slice_vars) < 2:
                    sample_pedon_slice_vars = (
                        sliceT[sliceT["compname"] == "sample_pedon"]
                        .dropna(axis="columns")
                        .drop(["compname"], 1)
                        .columns.tolist()
                    )
                sliceT = sliceT[sample_pedon_slice_vars]

            D = gower_distances(sliceT)  # Equal weighting given to all soil variables
            dis_mat_list.append(D)

        # Determine if any components have all NaNs at every slice
        dis_mat_nan_check = np.ma.MaskedArray(dis_mat_list, mask=np.isnan(dis_mat_list))
        D_check = np.ma.average(dis_mat_nan_check, axis=0)
        Rank_Filter["rank_status"] = [
            "Not Ranked" if np.ma.is_masked(x) else "Ranked" for x in D_check[0][1:]
        ]

        # Calculate maximum dissimilarity
        dis_max_slice = [
            np.nanmax(matrix) if not np.isnan(matrix).all() else np.nan for matrix in dis_mat_list
        ]
        dis_max = np.nanmax(dis_max_slice)
        dis_max_slice = [dis_max if np.isnan(x) else x for x in dis_max_slice]

        # Apply depth weight
        depth_weight = np.concatenate((np.repeat(0.2, 20), np.repeat(1.0, 80)), axis=0)
        depth_weight = depth_weight[pedon_slice_index]

        # Infill Nan data

        # Update dis_mat_list using numpy operations
        for i, dis_mat in enumerate(dis_mat_list):
            soil_slice = soil_matrix.iloc[i, :]

            # Identify where NaN values exist and where the corresponding soil_slice is 1
            nan_and_soil_slice_is_one = np.isnan(dis_mat) & np.isin(
                np.arange(dis_mat.shape[1]), np.where(soil_slice == 1)[0]
            )

            # Identify where NaN values exist and both corresponding values in soil_slice are 0
            rows, cols = np.where(np.isnan(dis_mat))
            both_zero_rows_cols = [
                (row, col)
                for row, col in zip(rows, cols)
                if soil_slice[row] == 0 and soil_slice[col] == 0
            ]

            # Assign max dissimilarity or 0 based on conditions
            dis_mat[nan_and_soil_slice_is_one] = dis_max
            for row, col in both_zero_rows_cols:
                dis_mat[row, col] = 0

            dis_mat_list[i] = dis_mat

        # Weighted average of depth-wise dissimilarity matrices
        dis_mat_list = np.ma.MaskedArray(dis_mat_list, mask=np.isnan(dis_mat_list))
        D_sum = (
            np.ma.average(dis_mat_list, axis=0, weights=depth_weight)
            if depth_weight is not None
            else np.ma.average(dis_mat_list, axis=0)
        )
        D_sum = np.ma.filled(D_sum, fill_value=np.nan)
        D_horz = 1 - D_sum

    else:
        D_horz = None

    # ---Site Data Similarity---
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

    # Initialize variables for site similarity
    p_slope = pd.DataFrame(["sample_pedon", pSlope, pElev]).T
    p_slope.columns = ["compname", "slope_r", "elev_r"]

    # Check conditions to determine the data columns and feature weights
    if (pSlope is not None) and (pElev is not None) and (p_bottom_depth.bottom_depth.any() > 0):
        D_site = compute_site_similarity(
            p_slope,
            mucompdata_pd,
            slices_of_soil,
            ["slope_r", "elev_r", "bottom_depth"],
            feature_weight=np.array([1.0, 0.5, 0.5]),
        )
    else:
        D_site = compute_site_similarity(
            p_slope, mucompdata_pd, slices_of_soil, feature_weight=np.array([1.0, 0.5])
        )

    # Adjust the distances and apply weight
    site_wt = 0.5
    D_site = (1 - D_site) * site_wt

    # Create the D_final dataframe based on the availability of D_horz and D_site data

    # When both D_horz and D_site are available
    if D_horz is not None and D_site is not None:
        D_site_hz = np.sum([D_site, D_horz], axis=0) / (1 + site_wt)
        D_final = pd.concat(
            [compnames, pd.Series(D_site_hz[0][1:]), pd.Series(np.repeat(1.0, len(compnames)))],
            axis=1,
        )

    # When only D_horz is available
    elif D_horz is not None:
        D_final = pd.concat(
            [compnames, pd.Series(D_horz[0][1:]), pd.Series(np.repeat(1.0, len(compnames)))], axis=1
        )

    # When only D_site is available
    elif D_site is not None:
        D_final = pd.concat(
            [compnames, pd.Series(D_site[0][1:]), pd.Series(np.repeat(1.0, len(compnames)))], axis=1
        )
        # Set rank status to 'Ranked with site data'
        Rank_Filter = pd.concat(
            [
                mucompdata_pd.compname,
                pd.Series(np.tile(["Ranked with site data"], len(mucompdata_pd))),
                pd.Series(np.tile(["Site data only"], len(mucompdata_pd))),
            ],
            axis=1,
        )
        Rank_Filter.columns = ["compname", "rank_status", "missing_status"]

    # When neither D_horz nor D_site is available
    else:
        D_final = pd.concat(
            [
                compnames,
                pd.Series(np.repeat(0.0, len(compnames))),
                pd.Series(np.repeat(0.0, len(compnames))),
            ],
            axis=1,
        )
        # Set rank status to 'Ranked with location data'
        Rank_Filter = pd.concat(
            [
                mucompdata_pd.compname,
                pd.Series(np.tile(["Ranked with location data"], len(mucompdata_pd))),
                pd.Series(np.tile(["Location data only"], len(mucompdata_pd))),
            ],
            axis=1,
        )
        Rank_Filter.columns = ["compname", "rank_status", "missing_status"]

    # Setting the columns for D_final dataframe
    D_final.columns = ["compname", "compname_grp", "Score_Data", "data_weight"]

    # Assigning the D_horz and D_site values to the D_final dataframe
    D_final["D_horz"] = np.nan if D_horz is None else D_horz[0][1:]
    D_final["D_site"] = np.nan if D_site is None else D_site[0][1:]

    # Refactoring the code for sorting, reindexing, and ranking the Data-only score

    # Sort and rank the components within each group
    soilIDList_data = []
    for _, group in D_final.groupby(["compname_grp"], sort=False):
        # Sort by score, and then by compname
        group = group.sort_values(by=["Score_Data", "compname"], ascending=[False, True])

        # The top component in each group gets a True rank, others get False
        group["soilID_rank_data"] = [True if i == 0 else False for i in range(len(group))]

        soilIDList_data.append(group)

    # Concatenate the sorted and ranked groups
    D_final = pd.concat(soilIDList_data).reset_index(drop=True)

    # Merge with the Rank_Filter data
    D_final = pd.merge(D_final, Rank_Filter, on="compname", how="left")

    # Assigning rank based on the soilID rank and rank status
    rank_id = 1
    Rank_Data = []
    for idx, row in D_final.iterrows():
        if not row["soilID_rank_data"]:
            Rank_Data.append("Not Displayed")
        elif row["rank_status"] == "Not Ranked":
            Rank_Data.append("Not ranked")
        else:
            Rank_Data.append(str(rank_id))
            rank_id += 1

    D_final["Rank_Data"] = Rank_Data
    """
    Code options for production API/testing output
    # ----------------------------------------------------------------
    #Data output for testing
    D_final_loc = pd.merge(D_final, mucompdata_pd[['compname', 'cokey', 'mukey',
    'distance_score', 'distance_score_norm', 'clay', 'taxorder', 'taxsubgrp', 'OSD_text_int',
    'OSD_rfv_int', 'data_source', 'Rank_Loc', 'majcompflag', 'comppct_r', 'distance',
    'nirrcapcl', 'nirrcapscl', 'nirrcapunit', 'irrcapcl', 'irrcapscl', 'irrcapunit',
    'ecoclassid_update', 'ecoclassname']], on='compname', how='left')
    """
    # Refactoring the code for data output

    # Merge D_final with additional data from mucompdata_pd
    D_final_loc = pd.merge(
        D_final,
        mucompdata_pd[
            [
                "compname",
                "cokey",
                "distance_score",
                "distance_score_norm",
                "clay",
                "taxorder",
                "taxsubgrp",
                "OSD_text_int",
                "OSD_rfv_int",
                "data_source",
                "Rank_Loc",
            ]
        ],
        on="compname",
        how="left",
    )

    # If Score_Data is NaN, assign 0 to data_weight
    D_final_loc.loc[D_final_loc["Score_Data"].isnull(), "data_weight"] = 0

    # Calculate the final score incorporating the location score
    location_weight = 1

    # Check if both D_horz and D_site are None
    if D_horz is None and D_site is None:
        Score_Data_Loc = [0.0 for _ in range(len(D_final_loc))]
    else:
        # Scale location and data scores for equal weighting
        D_final_loc["Score_Data_scale"] = D_final_loc["Score_Data"] / np.nanmax(
            D_final_loc["Score_Data"]
        )
        D_final_loc["distance_score_scale"] = D_final_loc["distance_score"] / np.nanmax(
            D_final_loc["distance_score"]
        )

        # Calculate the combined score
        Score_Data_Loc = (D_final_loc["Score_Data_scale"] + D_final_loc["distance_score_scale"]) / (
            D_final_loc["data_weight"] + location_weight
        )
        Score_Data_Loc /= np.nanmax(Score_Data_Loc)

    # Assign the final combined score to the dataframe
    D_final_loc["Score_Data_Loc"] = Score_Data_Loc

    # Rule-based code to identify vertisols or soils with vertic properties

    # Identify vertisols based on cracks, clay texture, and taxonomic presence of "ert"
    # Compute the condition for rows that meet the criteria
    condition = (
        D_final_loc["cracks"]
        & (D_final_loc["clay"] == "Yes")
        & (
            D_final_loc["taxorder"].str.contains("ert", case=False)
            | D_final_loc["taxsubgrp"].str.contains("ert", case=False)
        )
        & D_final_loc["soilID_rank_data"]
    )

    # Sum the number of components that meet the criteria
    vert = condition.sum()

    # If only one component meets these criteria, assign weight of 1 to that component
    if vert == 1:
        D_final_loc.loc[condition, "Score_Data_Loc"] = 1.001

    # Sorting and reindexing of final dataframe based on component groups
    soilIDList_out = []

    for _, group in D_final_loc.groupby("compname_grp", sort=False):
        group = group.sort_values("Score_Data_Loc", ascending=False).reset_index(drop=True)
        group["soilID_rank_final"] = [True if idx == 0 else False for idx in range(len(group))]
        soilIDList_out.append(group)

    # Concatenate the list of dataframes
    D_final_loc = pd.concat(soilIDList_out)

    # Sort dataframe based on Score_Data_Loc and compname
    D_final_loc = D_final_loc.sort_values(
        ["Score_Data_Loc", "compname"], ascending=[False, True]
    ).reset_index(drop=True)

    # Create a Rank_Data_Loc column
    rank_id = 1
    Rank_DataLoc = []

    for i, row in D_final_loc.iterrows():
        if not row["soilID_rank_final"]:
            Rank_DataLoc.append("Not Displayed")
        elif row["rank_status"] == "Not Ranked":
            Rank_DataLoc.append("Not ranked")
        else:
            Rank_DataLoc.append(str(rank_id))
            rank_id += 1

    D_final_loc["Rank_Data_Loc"] = Rank_DataLoc

    # Sort dataframe based on soilID_rank_final and Score_Data_Loc
    D_final_loc = D_final_loc.sort_values(
        ["soilID_rank_final", "Score_Data_Loc"], ascending=[False, False]
    ).reset_index(drop=True)

    # Uncomment code for testing output
    """
    # ----------------------------------------------------------------
    # Data formatting for testing

    # # Update LCC_I column
    D_final_loc['LCC_I'] = np.where(
        (D_final_loc['irrcapcl'] == 'nan') | (D_final_loc['irrcapscl'] == 'nan'),
        None, D_final_loc['irrcapcl'] + "-" + D_final_loc['irrcapscl']
    )

    # Update LCC_NI column
    D_final_loc['LCC_NI'] = np.where(
        (D_final_loc['nirrcapcl'] == 'nan') | (D_final_loc['nirrcapscl'] == 'nan'),
        None,
        D_final_loc['nirrcapcl'] + "-" + D_final_loc['nirrcapscl']
    )
    # ----------------------------------------------------------------
    """

    # Replace NaN values in the specified columns with 0.0
    D_final_loc[
        [
            "Score_Data",
            "D_horz",
            "D_site",
            "Score_Data_Loc",
            "distance_score",
            "distance_score_norm",
        ]
    ] = D_final_loc[
        [
            "Score_Data",
            "D_horz",
            "D_site",
            "Score_Data_Loc",
            "distance_score",
            "distance_score_norm",
        ]
    ].fillna(
        0.0
    )

    # Adjust the 'Score_Data_Loc' column based on 'data_completeness'
    min_scaling_factor = 0.25 if data_completeness < 25 else float(data_completeness) / 100
    D_final_loc["Score_Data_Loc"] *= min_scaling_factor

    # Construct the output format
    Rank = [
        {
            "name": row.compname.capitalize(),
            "component": row.compname_grp.capitalize(),
            "componentID": row.cokey,
            "score_data_loc": ""
            if row.missing_status == "Location data only"
            else round(row.Score_Data_Loc, 3),
            "rank_data_loc": ""
            if row.missing_status == "Location data only"
            else row.Rank_Data_Loc,
            "score_data": ""
            if row.missing_status == "Location data only"
            else round(row.Score_Data, 3),
            "rank_data": "" if row.missing_status == "Location data only" else row.Rank_Data,
            "score_loc": round(row.distance_score_norm, 3),
            "rank_loc": row.Rank_Loc,
            "componentData": row.missing_status,
        }
        for _, row in D_final_loc.iterrows()
    ]

    output_data = {
        "metadata": {
            "location": "us",
            "model": "v2",
            "dataCompleteness": {"score": data_completeness, "text": text_completeness},
        },
        "soilRank": Rank,
    }

    # If 'record_id' is provided, save the output data
    if record_id is not None:
        model_version = 3
        save_rank_output(record_id, model_version, json.dumps(output_data))

    return output_data
    """
    # Data return for testing
    return(D_final_loc[['compname', 'compname_grp', 'Rank_Loc', 'distance_score_norm',
    'Rank_Data', 'Score_Data', 'Rank_Data_Loc', 'Score_Data_Loc','ecoclassid_update',
    'ecoclassname', 'LCC_I', 'LCC_NI', 'taxorder', 'taxsubgrp', 'majcompflag',
    'comppct_r', 'distance']])
    # ----------------------------------------------------------------
    """
