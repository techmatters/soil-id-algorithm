# Standard libraries
import collections
import csv
import json
import re

# Third-party libraries
import geopandas as gpd
import numpy as np
import pandas as pd
import requests
import scipy.stats
import shapely

# Flask
from flask import current_app

# Import local fucntions
from model.local_functions_SoilID_v3 import *
from osgeo import ogr
from pandas.io.json import json_normalize
from shapely.geometry import Point


#####################################################################################################
#                                       Database and API Functions                                  #
#####################################################################################################
def findSoilLocation(lon, lat):
    """
    Determines the location type (US, Global, or None) of the given longitude and latitude based on soil datasets.

    Args:
    - lon (float): Longitude of the point.
    - lat (float): Latitude of the point.

    Returns:
    - str or None: 'US' if point is in US soil dataset, 'Global' if in global dataset, None otherwise.
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


#####################################################################################################
#                                 getSoilLocationBasedGlobal                                        #
#####################################################################################################
def getSoilLocationBasedGlobal(lon, lat, plot_id):
    # Extract HWSD-WISE Data
    # Note: Need to convert HWSD shp to gpkg file
    wise_data = extract_WISE_data(
        lon,
        lat,
        file_path="%s/wise30sec_poly_simp_soil.shp" % current_app.config["DATA_BACKEND"],
        layer_name=None,
        buffer_size=0.5,
    )

    # Component Data
    mucompdata_pd = wise_data[["MUGLB_NEW", "SU_name", "distance", "PROP", "COMPID", "FAO_SYS"]]
    mucompdata_pd.columns = ["mukey", "compname", "distance", "share", "cokey", "fss"]
    mucompdata_pd["distance"] = pd.to_numeric(mucompdata_pd["distance"])
    mucompdata_pd["share"] = pd.to_numeric(mucompdata_pd["share"])
    mucompdata_pd = mucompdata_pd.drop_duplicates().reset_index(drop=True)

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    #########################################################################################################################################
    # Individual probability
    # Based on Fan et al 2018 EQ 1, the conditional probability for each component is calculated by taking the sum of all occurances of a component
    # in the home and adjacent mapunits and dividing this by the sum of all map units and components. We have modified this approach so that each
    # instance of a component occurance is evaluated separately and assinged a weight and the max distance score for each component group is assigned
    # to all component instances.
    #########################################################################################################################################
    ExpCoeff = -0.00036888  # Decays to 0.25 @ 10km
    loc_scores = []
    mucompdata_grouped = mucompdata_pd.groupby(["mukey", "cokey"], sort=False)

    for (mukey, cokey), group in mucompdata_grouped:
        loc_score = calculate_location_score(group, ExpCoeff)
        loc_scores.append({"Cokey": cokey, "Mukey": mukey, "distance_score": round(loc_score, 3)})

    loc_top_pd = pd.DataFrame(loc_scores)
    loc_top_comp_prob = loc_top_pd.groupby("Cokey").distance_score.sum()
    loc_bot_prob_sum = loc_top_pd.distance_score.sum()
    cond_prob = (loc_top_comp_prob / loc_bot_prob_sum).reset_index(name="distance_score")

    mucompdata_pd = pd.merge(mucompdata_pd, cond_prob, on="cokey", how="left")
    mucompdata_pd = mucompdata_pd.sort_values("distance_score", ascending=False)
    mucompdata_pd["distance_score_norm"] = (
        mucompdata_pd.distance_score / mucompdata_pd.distance_score.max()
    ) * 0.25
    mucompdata_pd = mucompdata_pd.reset_index(drop=True)
    mucompdata_pd["distance"] = mucompdata_pd["distance"].round(4)
    mucompdata_pd["Index"] = mucompdata_pd.index

    # Group by component name
    mucompdata_grouped = mucompdata_pd.groupby("compname", sort=False)

    # Take at most 12 groups
    mucompdata_comp_grps = [group for _, group in mucompdata_grouped][:12]

    # Assign max distance scores to all members within each group
    soilIDList_out = [assign_max_distance_scores(group) for group in mucompdata_comp_grps]

    mucompdata_pd = pd.concat(soilIDList_out).reset_index(drop=True)
    index = mucompdata_pd["Index"].tolist()
    comp_key = mucompdata_pd["cokey"].tolist()

    # -----------------------------------------------------------------------------------------------------------------
    # Create horizon data table
    columns_to_select = [
        "COMPID",
        "TopDep",
        "BotDep",
        "id",
        "Layer",
        "SDTO",
        "STPC",
        "CLPC",
        "CFRAG",
        "CECS",
        "PHAQ",
        "ELCO",
        "PROP",
        "SU_name",
        "FAO_SYS",
    ]
    new_column_names = [
        "cokey",
        "hzdept_r",
        "hzdepb_r",
        "chkey",
        "hzname",
        "sandtotal_r",
        "silttotal_r",
        "claytotal_r",
        "total_frag_volume",
        "CEC",
        "pH",
        "EC",
        "comppct_r",
        "compname",
        "fss",
    ]

    muhorzdata_pd = wise_data[columns_to_select]
    muhorzdata_pd.columns = new_column_names
    muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(comp_key)]
    muhorzdata_pd[["hzdept_r", "hzdepb_r"]] = (
        muhorzdata_pd[["hzdept_r", "hzdepb_r"]].fillna(0).astype(int)
    )
    muhorzdata_pd["texture"] = muhorzdata_pd.apply(getTexture, axis=1)

    # Rank components and sort by rank and depth
    cokey_Index = {key: rank for rank, key in enumerate(comp_key)}
    muhorzdata_pd["Comp_Rank"] = muhorzdata_pd["cokey"].map(cokey_Index)
    muhorzdata_pd.sort_values(["Comp_Rank", "hzdept_r"], inplace=True)
    muhorzdata_pd.drop(columns="Comp_Rank", inplace=True)
    muhorzdata_pd = muhorzdata_pd.drop_duplicates().reset_index(drop=True)

    # Check for duplicate component instances
    hz_drop = drop_cokey_horz(muhorzdata_pd)
    if hz_drop is not None:
        muhorzdata_pd = muhorzdata_pd[~muhorzdata_pd.cokey.isin(hz_drop)]

    # Update comp_key
    comp_key = muhorzdata_pd["cokey"].unique().tolist()

    # Subset mucompdata_pd by new comp_key and add suffix to name if there are duplicates
    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(comp_key)]
    mucompdata_pd.sort_values(["distance_score", "distance"], ascending=[False, True], inplace=True)
    mucompdata_pd.reset_index(drop=True, inplace=True)

    # Add suffix to duplicate names
    name_counts = collections.Counter(mucompdata_pd["compname"])
    for name, count in name_counts.items():
        if count > 1:
            for suffix in range(1, count + 1):
                mucompdata_pd.loc[mucompdata_pd["compname"] == name, "compname"] = name + str(
                    suffix
                )

    # Add modified compname to muhorzdata
    muhorzdata_name = muhorzdata_pd[["cokey"]].merge(
        mucompdata_pd[["cokey", "compname"]], on="cokey"
    )
    muhorzdata_pd["compname"] = muhorzdata_name["compname"]

    # Group data by cokey for texture
    muhorzdata_group_cokey = list(muhorzdata_pd.groupby("cokey", sort=False))

    # Initialize lists for storing data
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

    for group in muhorzdata_group_cokey:
        profile = (
            group.sort_values(by="hzdept_r").drop_duplicates(keep="first").reset_index(drop=True)
        )

        c_very_bottom, sand_pct_intpl = getProfile(profile, "sandtotal_r", c_bot=True)
        sand_pct_intpl.columns = ["c_sandpct_intpl", "c_sandpct_intpl_grp"]

        def process_profile(param):
            result = getProfile(profile, param)
            result.columns = [f"c_{param}_intpl", f"c_{param}_intpl_grp"]
            return result

        clay_pct_intpl = process_profile("claytotal_r")
        cf_pct_intpl = process_profile("total_frag_volume")
        cec_intpl = process_profile("CEC")
        ph_intpl = process_profile("pH")
        ec_intpl = process_profile("EC")

        combined_data = pd.concat(
            [
                sand_pct_intpl[["c_sandpct_intpl_grp"]],
                clay_pct_intpl[["c_claypct_intpl_grp"]],
                cf_pct_intpl[["c_cfpct_intpl_grp"]],
                profile.compname.unique(),
                profile.cokey.unique(),
                profile.comppct_r.unique(),
            ],
            axis=1,
        )

        combined_data.columns = [
            "sandpct_intpl",
            "claypct_intpl",
            "rfv_intpl",
            "compname",
            "cokey",
            "comppct",
        ]

        c_bottom_temp = pd.DataFrame(
            {
                "cokey": [combined_data["cokey"].iloc[0]],
                "compname": [combined_data["compname"].iloc[0]],
                "c_very_bottom": [int(c_very_bottom)],
            }
        )

        def aggregated_data_layer(data, column_name):
            return agg_data_layer(
                data=data[column_name],
                bottom=c_bottom_temp["c_very_bottom"].iloc[0],
                depth=True,
            )

        snd_d, hz_depb = aggregated_data_layer(sand_pct_intpl, "c_sandpct_intpl")
        cly_d = aggregated_data_layer(clay_pct_intpl, "c_claypct_intpl")
        txt_d = [
            getTexture(row=None, sand=s, silt=(100 - (s + c)), clay=c) for s, c in zip(snd_d, cly_d)
        ]
        txt_d = pd.Series(txt_d, index=snd_d.index)

        rf_d = aggregated_data_layer(cf_pct_intpl, "c_cfpct_intpl_grp")
        cec_d = aggregated_data_layer(cec_intpl, "c_cec_intpl")
        ph_d = aggregated_data_layer(ph_intpl, "c_ph_intpl")
        ec_d = aggregated_data_layer(ec_intpl, "c_ec_intpl")

        # Fill NaN values and append to lists
        for data_list, data in zip(
            [
                snd_lyrs,
                cly_lyrs,
                txt_lyrs,
                rf_lyrs,
                cec_lyrs,
                ph_lyrs,
                ec_lyrs,
                hz_lyrs,
            ],
            [snd_d, cly_d, txt_d, rf_d, cec_d, ph_d, ec_d, hz_depb],
        ):
            data_list.append(dict(data.fillna("")))

        c_bottom_depths.append(c_bottom_temp)
        getProfile_cokey.append(combined_data)

        comp_texture_list = [x for x in profile.texture.str.lower() if x]
        clay_val = "Yes" if any("clay" in string for string in comp_texture_list) else "No"
        clay_texture_temp = pd.DataFrame(
            {"compname": [combined_data["compname"].iloc[0]], "clay": [clay_val]}
        )
        clay_texture.append(clay_texture_temp)

    # Concatenate lists to form DataFrames
    c_bottom_depths = pd.concat(c_bottom_depths, axis=0)
    clay_texture = pd.concat(clay_texture, axis=0)

    # Subset mucompdata and muhorzdata DataFrames
    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(c_bottom_depths.cokey)]
    muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(c_bottom_depths.cokey)]

    # Merge c_bottom_depth and clay_texture with mucompdata
    mucompdata_pd = pd.merge(
        mucompdata_pd,
        c_bottom_depths[["compname", "c_very_bottom"]],
        on="compname",
        how="left",
    )
    mucompdata_pd = pd.merge(mucompdata_pd, clay_texture, on="compname", how="left")

    # Create index for component instance display
    mucompdata_comp_grps_list = []
    mucompdata_comp_grps = [group for _, group in mucompdata_pd.groupby("compname_grp", sort=False)]

    for group in mucompdata_comp_grps:
        group = group.sort_values("distance").reset_index(drop=True)
        soilID_rank = [True if idx == 0 else False for idx in range(len(group))]

        group["soilID_rank"] = soilID_rank
        group["min_dist"] = group.distance.iloc[0]

        mucompdata_comp_grps_list.append(group)

    mucompdata_pd = pd.concat(mucompdata_comp_grps_list).reset_index(drop=True)

    # --------------------------------------------------------------------------------------------------------------------------
    # SoilIDList output

    # ------------------------------------------------------------
    # Format output data
    soilIDRank_output = [
        group[["compname", "sandpct_intpl", "claypct_intpl", "rfv_intpl"]]
        for group in getProfile_cokey
    ]
    soilIDRank_output_pd = pd.concat(soilIDRank_output, axis=0).reset_index(drop=True)

    mucompdata_cond_prob = mucompdata_pd.sort_values(
        "distance_score_norm", ascending=False
    ).reset_index(drop=True)

    # Determine rank location
    rank_id = 1
    Rank_Loc = []
    for rank in mucompdata_cond_prob["soilID_rank"]:
        Rank_Loc.append(str(rank_id) if rank else "Not Displayed")
        rank_id += rank  # Increase rank_id only if rank is True
    mucompdata_cond_prob["Rank_Loc"] = Rank_Loc

    # Handle NaN values
    mucompdata_cond_prob.replace({np.nan: "", "nan": "", "None": "", None: ""}, inplace=True)

    # Merge component descriptions
    WRB_Comp_Desc = getWRB_descriptions(
        mucompdata_cond_prob["compname_grp"].drop_duplicates().tolist()
    )
    mucompdata_cond_prob = pd.merge(
        mucompdata_cond_prob,
        WRB_Comp_Desc,
        left_on="compname_grp",
        right_on="WRB_tax",
        how="left",
    )

    mucomp_index = mucompdata_cond_prob.sort_values(
        ["soilID_rank", "distance_score_norm"], ascending=[False, False]
    ).index

    # Extract site information
    Site = [
        {
            "siteData": {
                "mapunitID": row.mukey,
                "componentID": row.cokey,
                "fao": row.fss,
                "share": row.share,
                "distance": round(row.distance, 3),
                "minCompDistance": row.min_dist,
                "soilDepth": row.c_very_bottom,
            },
            "siteDescription": {
                key: row[key]
                for key in [
                    "Description_en",
                    "Management_en",
                    "Description_es",
                    "Management_es",
                    "Description_ks",
                    "Management_ks",
                    "Description_fr",
                    "Management_fr",
                ]
            },
        }
        for _, row in mucompdata_cond_prob.iterrows()
    ]

    # Reorder lists based on mucomp_index
    hz_lyrs = [hz_lyrs[i] for i in mucomp_index]
    snd_lyrs = [snd_lyrs[i] for i in mucomp_index]
    cly_lyrs = [cly_lyrs[i] for i in mucomp_index]
    txt_lyrs = [txt_lyrs[i] for i in mucomp_index]
    rf_lyrs = [rf_lyrs[i] for i in mucomp_index]
    cec_lyrs = [cec_lyrs[i] for i in mucomp_index]
    ph_lyrs = [ph_lyrs[i] for i in mucomp_index]
    ec_lyrs = [ec_lyrs[i] for i in mucomp_index]

    output_SoilList = [
        dict(
            zip(
                [
                    "id",
                    "site",
                    "bottom_depth",
                    "sand",
                    "clay",
                    "texture",
                    "rock_fragments",
                    "cec",
                    "ph",
                    "ec",
                ],
                item,
            )
        )
        for item in zip(
            ID,
            Site,
            hz_lyrs,
            snd_lyrs,
            cly_lyrs,
            txt_lyrs,
            rf_lyrs,
            cec_lyrs,
            ph_lyrs,
            ec_lyrs,
        )
    ]

    # Save data
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
        saveModelOutput(
            plot_id,
            model_version,
            json.dumps(
                {
                    "metadata": {
                        "location": "global",
                        "model": "v2",
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
            ),
            soilIDRank_output_pd.to_csv(index=None, header=True),
            mucompdata_cond_prob.to_csv(index=None, header=True),
        )

    # Return the JSON output
    return {
        "metadata": {
            "location": "global",
            "model": "v2",
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


#####################################################################################################
#                                   rankPredictionGlobal                                            #
#####################################################################################################
def rankPredictionGlobal(
    lon,
    lat,
    soilHorizon,
    horizonDepth,
    rfvDepth,
    lab_Color,
    bedrock,
    cracks,
    plot_id=None,
):
    # ------------------------------------------------------------------------------------------------
    # ------ Load in user data --------#

    # Create the dataframe
    soil_df = pd.DataFrame(
        {
            "soilHorizon": soilHorizon,
            "horizonDepth": horizonDepth,
            "rfvDepth": rfvDepth,
            "lab_Color": lab_Color,
        }
    )

    # Cleanup the dataframe
    soil_df.dropna(how="all", inplace=True)
    soil_df["bottom"] = soil_df["horizonDepth"]

    # Replace NaN values with None for consistency
    soil_df = soil_df.where(pd.notna(soil_df), None)

    # Compute the 'top' column
    top = [0] + [int(depth) for depth in soil_df["horizonDepth"][:-1]]
    soil_df["top"] = top

    # Set 'horizonDepth' as the index
    soil_df.set_index("horizonDepth", inplace=True)

    # Drop rows where 'soilHorizon' and 'rfvDepth' are both missing
    soil_df_slice = soil_df.dropna(subset=["soilHorizon", "rfvDepth"], how="all")

    # If the slice is not empty, reset its index
    if not soil_df_slice.empty:
        soil_df_slice.reset_index(inplace=True)
        soil_df_slice.rename(columns={"horizonDepth": "index"}, inplace=True)
        soil_df_slice.set_index("index", inplace=True)

    if soil_df_slice is not None:
        # If bedrock has been recorded and the lowest soil depth associated with data is greater than bedrock,
        # then change lowest soil depth to bedrock depth
        if bedrock and soil_df_slice["bottom"].iloc[-1] > bedrock:
            soil_df_slice["bottom"].iloc[-1] = bedrock
            soil_df["bottom"][soil_df_slice.index[-1]] = bedrock
            soil_df.reset_index(drop=True, inplace=True)

            # Infill missing horizons
            for j in range(len(soil_df) - 1):
                if soil_df["top"].iloc[j + 1] > soil_df["bottom"].iloc[j]:
                    layer_add = pd.DataFrame(
                        {
                            "soilHorizon": [None],
                            "rfvDepth": [None],
                            "lab_Color": [None],
                            "bottom": [soil_df["top"].iloc[j + 1]],
                            "top": [soil_df["bottom"].iloc[j]],
                        }
                    )
                    soil_df = (
                        pd.concat([soil_df, layer_add], axis=0)
                        .sort_values("top")
                        .reset_index(drop=True)
                    )

            soil_df.where(pd.notnull(soil_df), None, inplace=True)

        # Create index list of soil slices where user data exists
        soil_df_slice.reset_index(drop=True, inplace=True)
        pedon_slice_index = [
            j
            for i in range(len(soil_df_slice))
            for j in range(int(soil_df_slice["top"].iloc[i]), int(soil_df_slice["bottom"].iloc[i]))
            if j < 120
        ]
        if bedrock:
            pedon_slice_index.extend(range(bedrock, 120))

        # Soil properties to lists
        soilHorizon, rfvDepth = (
            soil_df["soilHorizon"].tolist(),
            soil_df["rfvDepth"].tolist(),
        )
        horizonDepthB, horizonDepthT = list(map(int, soil_df["bottom"])), list(
            map(int, soil_df["top"])
        )
        lab_Color = soil_df["lab_Color"]

        # Format and generate user data
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

        # Adjust depth interval of user data
        for data_list in [p_sandpct_intpl, p_claypct_intpl, p_cfg_intpl]:
            if len(data_list) > 120:
                data_list = data_list[:120]
            else:
                data_list.extend([np.nan] * (120 - len(data_list)))

        p_hz_data = pd.DataFrame(
            {
                "compname": "sample_pedon",
                "sandpct_intpl": p_sandpct_intpl,
                "claypct_intpl": p_claypct_intpl,
                "rfv_intpl": p_cfg_intpl,
            }
        )

        # Drop empty data columns and depth slices
        p_hz_data.dropna(axis=1, how="all", inplace=True)
        p_hz_data = p_hz_data[p_hz_data.index.isin(pedon_slice_index)]
        p_hz_data_names = p_hz_data.columns.tolist()

    else:
        p_hz_data, p_sandpct_intpl, p_claypct_intpl, p_cfg_intpl = None, [], [], []
        p_bottom_depth = pd.DataFrame(
            {"cokey": [-999], "compname": ["sample_pedon"], "bottom_depth": [0]}
        )

    lab_Color = pd.Series(lab_Color)
    if not lab_Color.isnull().all():
        lab_Color = [[np.nan, np.nan, np.nan] if x is None else x for x in lab_Color]
        lab_Color = pd.DataFrame(lab_Color)
        pedon_LAB = pedon_color(lab_Color, horizonDepth)

        if not np.isnan(pedon_LAB).all():
            refs = {
                "white": [100.0, 0.0, 0.0],
                "red": [53.2, 80.1, 67.2],
                "yellow": [97.1, -21.6, 94.5],
            }

            cr_df = [calculate_deltaE2000(pedon_LAB, refs[color]) for color in refs]
            cr_df = pd.Series(cr_df)
        else:
            cr_df = pd.Series([np.nan])
    else:
        cr_df = pd.Series([np.nan])

    # Calculate Data Completeness Score
    data_completeness, text_completeness = compute_data_completeness(
        bedrock, p_sandpct_intpl, soilHorizon, p_cfg_intpl, rfvDepth, cracks, cr_df
    )

    # --------------------------------------------------------------------------------------------------------------------------------------
    # Load in component data from soilIDList
    # Initialize
    record_id = None

    # If no plot_id is provided, load data from file
    if plot_id is None:
        soilIDRank_output = pd.read_csv(
            "{}/soilIDRank_ofile1.csv".format(current_app.config["DATA_BACKEND"])
        )
        mucompdata = pd.read_csv(
            "{}/soilIDRank_ofile2.csv".format(current_app.config["DATA_BACKEND"])
        )

    # If plot_id is provided, load data from the database
    else:
        modelRun = loadModelOutput(plot_id)

        # Check if modelRun data was successfully fetched
        if modelRun:
            record_id = modelRun[0]
            soilIDRank_output = pd.read_csv(io.StringIO(modelRun[2]))
            mucompdata = pd.read_csv(io.StringIO(modelRun[3]))
        else:
            return "Cannot find a plot with this ID"

    # Group the soilIDRank_output dataframe by 'compname' and return
    grouped_soil_data = [group for _, group in soilIDRank_output.groupby("compname", sort=False)]

    # Create soil depth DataFrame and subset component depths based on max user depth if no bedrock specified
    c_bottom_depths = mucompdata_pd[["compname", "c_very_bottom"]].rename(
        columns={"c_very_bottom": "bottom_depth"}
    )
    slices_of_soil = pd.concat([p_bottom_depth, c_bottom_depths], axis=0).reset_index(drop=True)

    compnames = mucompdata_pd[["compname", "compname_grp"]]

    # If bedrock is not specified, determine max_depth based on user recorded depth (limited to 120 cm)
    max_depth = (
        120
        if bedrock is None and p_bottom_depth.bottom_depth.values[0] > 120
        else p_bottom_depth.bottom_depth.values[0]
    )

    # If bedrock is specified, truncate to the app's max depth of 120 cm
    max_depth = 120 if bedrock is not None else max_depth

    # Adjust slices of soil if they exceed the determined max_depth
    slices_of_soil.loc[slices_of_soil.bottom_depth > max_depth, "bottom_depth"] = max_depth
    slices_of_non_soil = max_depth - slices_of_soil.bottom_depth

    # Generate a matrix describing soil (1) vs. non-soil (0) for each slice
    soil_matrix = pd.DataFrame(
        np.nan, index=np.arange(max_depth), columns=np.arange(len(slices_of_soil))
    )
    for i in range(len(slices_of_soil)):
        slice_end = slices_of_soil.bottom_depth.iloc[i]
        soil_matrix.iloc[:slice_end, i] = 1
        soil_matrix.iloc[slice_end:max_depth, i] = 0

    # Determine if user has entered horizon data and if so, subset component horizon data based on user input data
    if p_hz_data:
        # Subset component soil properties to match user measured properties
        soilIDRank_output_pd = soilIDRank_output_pd[p_hz_data_names]
        # Subset soil_matrix to user measured slices
        soil_matrix = soil_matrix.loc[pedon_slice_index]

        # Horizon Data Similarity
        if p_bottom_depth.bottom_depth.any() > 0:
            horz_vars_group_cokey = [
                group for _, group in soilIDRank_output_pd.groupby("compname", sort=False)
            ]

            # Create lists to store component statuses
            Comp_Rank_Status, Comp_Missing_Status, Comp_name = [], [], []

            # Check component status
            for group in horz_vars_group_cokey:
                subset_group = group[p_hz_data_names].drop(columns="compname")
                if subset_group.isnull().values.all():
                    Comp_Rank_Status.append("Not Ranked")
                    Comp_Missing_Status.append("No Data")
                else:
                    Comp_Rank_Status.append("Ranked")
                    Comp_Missing_Status.append(
                        "Missing Data" if subset_group.isnull().values.any() else "Data Complete"
                    )
                Comp_name.append(group["compname"].unique()[0])

            Rank_Filter = pd.DataFrame(
                {
                    "compname": Comp_name,
                    "rank_status": Comp_Rank_Status,
                    "missing_status": Comp_Missing_Status,
                }
            )

            horz_vars = [p_hz_data]
            for group in horz_vars_group_cokey:
                horz_vars.append(group.reset_index(drop=True).loc[pedon_slice_index])

            dis_mat_list = []
            for i in soil_matrix.index:
                slice_temp = [var.loc[i] for var in horz_vars]
                sliceT = pd.concat(slice_temp, axis=1).T
                slice_mat = sliceT.drop(columns="compname")

                if i < bedrock:
                    sample_pedon_slice_vars = (
                        sliceT.dropna(axis="columns").drop(columns="compname").columns.to_list()
                    )
                    if len(sample_pedon_slice_vars) < 2:
                        sample_pedon_slice_vars = (
                            sliceT[sliceT["compname"] == "sample_pedon"]
                            .dropna(axis="columns")
                            .drop(columns="compname")
                            .columns.to_list()
                        )
                    slice_mat = sliceT[sample_pedon_slice_vars]

                D = gower_distances(slice_mat)  # Equal weighting given to all soil variables
                dis_mat_list.append(D)

            # Check if any components have all NaNs at every slice
            dis_mat_nan_check = np.ma.MaskedArray(dis_mat_list, mask=np.isnan(dis_mat_list))
            D_check = np.ma.average(dis_mat_nan_check, axis=0)
            rank_status = ["Not Ranked" if np.ma.is_masked(x) else "Ranked" for x in D_check[0][1:]]
            Rank_Filter["rank_status"] = rank_status

            # Calculate max dissimilarity per depth slice
            dis_max = max(map(np.nanmax, dis_mat_list))

            # Apply depth weight
            depth_weight = np.concatenate([np.repeat(0.2, 20), np.repeat(1.0, 80)])
            depth_weight = depth_weight[: len(soil_matrix)]

            # Infill NaN data
            for idx, dis_mat in enumerate(dis_mat_list):
                soil_slice = soil_matrix.iloc[idx]
                for j in range(len(dis_mat)):
                    for k in range(len(dis_mat[j])):
                        if np.isnan(dis_mat[j, k]):
                            if (soil_slice[j] and not soil_slice[k]) or (
                                not soil_slice[j] and soil_slice[k]
                            ):
                                dis_mat[j, k] = dis_max
                            elif not (soil_slice[j] or soil_slice[k]):
                                dis_mat[j, k] = 0

            # Weighted average of depth-wise dissimilarity matrices
            dis_mat_list_masked = np.ma.MaskedArray(dis_mat_list, mask=np.isnan(dis_mat_list))
            D_sum = np.ma.average(dis_mat_list_masked, axis=0, weights=depth_weight)
            D_sum = np.ma.filled(D_sum, fill_value=np.nan)
            D_horz = 1 - D_sum

            D_final_horz = pd.concat(
                [
                    compnames.reset_index(drop=True),
                    pd.Series(D_horz[0][1:]),
                    pd.Series(np.repeat(1.0, len(compnames))),
                ],
                axis=1,
            )
            D_final_horz.columns = ["compname", "compname_grp", "horz_score", "weight"]
            D_final_horz = pd.merge(
                D_final_horz,
                mucompdata_pd[["compname", "mukey", "cokey", "distance_score", "Rank_Loc", "fss"]],
                on="compname",
                how="left",
            )
    else:
        D_final_horz = pd.concat(
            [
                compnames.reset_index(drop=True),
                pd.Series(np.repeat(0.0, len(compnames))),
                pd.Series(np.repeat(0.0, len(compnames))),
            ],
            axis=1,
        )
        D_final_horz.columns = ["compname", "compname_grp", "horz_score", "weight"]
        D_final_horz = pd.merge(
            D_final_horz,
            mucompdata_pd[["compname", "mukey", "cokey", "distance_score", "Rank_Loc", "fss"]],
            on="compname",
            how="left",
        )

    # ----------------------------------------------------------------------------------------
    """
    ##--Possibly add site disimilarity matrix in the future

    if lab_Color is not None:
        p_slope = pd.DataFrame(["sample_pedon", pSlope]).T
        p_slope.columns = ["compname", "slope_r"]
        soil_slope = pd.concat([p_slope, mucompdata_pd[['compname', 'slope_r']]], axis=0)
        site_vars = pd.merge(slices_of_soil, soil_slope, on='compname', how='left')
        compnames = site_vars[['compname']]
        site_mat = site_vars[['slope_r', 'bottom_depth']]
        site_mat = site_mat.set_index(compnames.compname.values)
        D_site = gower_distances(site_mat, feature_weight=np.array([1.,1.]))
        D_site = np.where(np.isnan(D_site), np.nanmin(D_site), D_site)
        D_site = D_site/np.nanmax(D_site)
        site_wt = 0.5
        D_site = (1 - D_site)*site_wt
        D_site_hz = np.sum([D_site, D_horz],axis=0)/(1+site_wt)
        D_site_hz = D_site_hz/np.nanmax(D_site_hz)
        D_final = pd.concat([compnames.reset_index(drop=True), pd.Series(D_site_hz[0][1:])], axis=1)
    else:
        D_final = pd.concat([compnames.reset_index(drop=True), pd.Series(D_horz[0][1:])], axis=1)

    """
    # --------------------------------------------------------------------------------------------

    # Start of soil color

    # Load in SRG color distribution data
    dist1_values = [wmf1, wsf1, rmf1, rsf1, ymf1, ysf1]
    dist2_values = [wmf2, wsf2, rmf2, rsf2, ymf2, ysf2]

    data_path = current_app.config["DATA_BACKEND"]

    with open(f"{data_path}/NormDist1.csv", "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for idx, row in enumerate(readCSV):
            dist1_values[idx].extend(row)

    with open(f"{data_path}/NormDist2.csv", "r") as csvfile:
        readCSV = csv.reader(csvfile, delimiter=",")
        for idx, row in enumerate(readCSV):
            dist2_values[idx].extend(row)

    fao74 = [
        "Acrisols",
        "Andosols",
        "Arenosols",
        "Cambisols",
        "Chernozems",
        "Ferralsols",
        "Fluvisols",
        "Gleysols",
        "Greyzems",
        "Histosols",
        "Kastanozems",
        "Luvisols",
        "Nitosols",
        "Phaeozems",
        "Planosols",
        "Podzols",
        "Podzoluvisols",
        "Regosols",
        "Solonchaks",
        "Solonetz",
        "Vertisols",
        "Xerosols",
        "Yermosols",
    ]

    fao90 = [
        "Acrisols",
        "Alisols",
        "Andosols",
        "Anthrosols",
        "Arenosols",
        "Calcisols",
        "Cambisols",
        "Chernozems",
        "Ferralsols",
        "Fluvisols",
        "Gleysols",
        "Greyzems",
        "Gypsisols",
        "Histosols",
        "Kastanozems",
        "Leptosols",
        "Lixisols",
        "Luvisols",
        "Nitisols",
        "Phaeozems",
        "Planosols",
        "Plinthosols",
        "Podzols",
        "Podzoluvisols",
        "Regosols",
        "Solonchaks",
        "Solonetz",
        "Vertisols",
    ]

    # Calculate color similarity
    if not cr_df.isnull().values.any():
        color_sim = []
        w_df, r_df, y_df = cr_df.iloc[0], cr_df.iloc[1], cr_df.iloc[2]
        fao74 = [item.lower() for item in fao74]
        fao90 = [item.lower() for item in fao90]

        for compname, fss in zip(D_final_horz.compname, D_final_horz.fss):
            soilgroup = re.sub(r"\d+$", "", " ".join(compname.split()[1:])).lower()

            prob_w, prob_r, prob_y = [], [], []

            if fss.lower() == "fao74":
                fao_list, wmf, wsf, rmf, rsf, ymf, ysf = (
                    fao74,
                    wmf1,
                    wsf1,
                    rmf1,
                    rsf1,
                    ymf1,
                    ysf1,
                )
            else:
                fao_list, wmf, wsf, rmf, rsf, ymf, ysf = (
                    fao90,
                    wmf2,
                    wsf2,
                    rmf2,
                    rsf2,
                    ymf2,
                    ysf2,
                )

            idx = fao_list.index(soilgroup) if soilgroup in fao_list else -1

            for mw, sw, mr, sr, my, sy in zip(wmf, wsf, rmf, rsf, ymf, ysf):
                prob_w.append(scipy.stats.norm(float(mw), float(sw)).pdf(float(w_df)))
                prob_r.append(scipy.stats.norm(float(mr), float(sr)).pdf(float(r_df)))
                prob_y.append(scipy.stats.norm(float(my), float(sy)).pdf(float(y_df)))

            max_prob_w, min_prob_w = max(prob_w), min(prob_w)
            max_prob_r, min_prob_r = max(prob_r), min(prob_r)
            max_prob_y, min_prob_y = max(prob_y), min(prob_y)

            for j in range(len(fao_list)):
                prob_w[j] = (prob_w[j] - min_prob_w) / (max_prob_w - min_prob_w)
                prob_r[j] = (prob_r[j] - min_prob_r) / (max_prob_r - min_prob_r)
                prob_y[j] = (prob_y[j] - min_prob_y) / (max_prob_y - min_prob_y)

            crsr = (prob_w[idx] + prob_r[idx] + prob_y[idx]) / 3.0 if idx != -1 else 1.0
            color_sim.append(crsr)

        color_sim = pd.Series(color_sim)

    # Calculate Data score
    global color_weight
    color_weight = 0.3

    if D_final_horz.horz_score.any() > 0 and cr_df.notnull().values.any():
        D_final_horz["Score_Data"] = (D_final_horz.horz_score + (color_sim * color_weight)) / (
            color_weight + D_final_horz.weight
        )
    elif D_final_horz.horz_score.all() == 0 and cr_df.notnull().values.any():
        D_final_horz["Score_Data"] = color_sim
        rank_status = ["Ranked with color data"] * len(mucompdata_pd)
        missing_status = ["Color data only"] * len(mucompdata_pd)
        Rank_Filter = pd.DataFrame(
            {
                "compname": mucompdata_pd.compname,
                "rank_status": rank_status,
                "missing_status": missing_status,
            }
        )
    elif D_final_horz.horz_score.any() > 0 and cr_df.isnull().values.any():
        D_final_horz["Score_Data"] = D_final_horz.horz_score
    else:
        D_final_horz["Score_Data"] = D_final_horz.horz_score
        rank_status = ["Ranked with location data"] * len(mucompdata_pd)
        missing_status = ["Location data only"] * len(mucompdata_pd)
        Rank_Filter = pd.DataFrame(
            {
                "compname": mucompdata_pd.compname,
                "rank_status": rank_status,
                "missing_status": missing_status,
            }
        )

    # Sorting and reindexing of Data-only score
    soilIDList_data = []
    D_final_comp_grps = [g for _, g in D_final_horz.groupby("compname_grp", sort=False)]

    for comp_grps_temp in D_final_comp_grps:
        comp_grps_temp = comp_grps_temp.sort_values("Score_Data", ascending=False).reset_index(
            drop=True
        )
        SID_data = [True] + [False] * (len(comp_grps_temp) - 1)
        comp_grps_temp["soilID_rank_data"] = SID_data
        soilIDList_data.append(comp_grps_temp)

    D_final_horz = pd.concat(soilIDList_data)

    # Generate Rank_Data
    D_final_horz = D_final_horz.sort_values(
        ["Score_Data", "compname"], ascending=[False, True]
    ).reset_index(drop=True)
    rank_id = 1
    Rank_Data = []

    for is_ranked in D_final_horz["soilID_rank_data"]:
        if is_ranked:
            Rank_Data.append(str(rank_id))
            rank_id += 1
        else:
            Rank_Data.append("Not Displayed")

    D_final_horz["Rank_Data"] = Rank_Data

    D_final_loc = pd.merge(
        D_final_horz,
        mucompdata_pd[["compname", "clay", "distance_score_norm"]],
        on="compname",
        how="left",
    )
    D_final_loc = pd.merge(D_final_loc, Rank_Filter, on="compname", how="left")

    # If Score_Data returns NA, assign 0 to weight
    D_final_loc.loc[D_final_loc.Score_Data.isnull(), "weight"] = 0

    # Adjust Data Score weight by the data completeness
    Score_Data_Loc = (D_final_loc[["Score_Data", "distance_score"]].sum(axis=1)) / (
        D_final_loc.weight + 1
    )
    Score_Data_Loc = Score_Data_Loc / np.nanmax(Score_Data_Loc)
    D_final_loc["Score_Data_Loc"] = Score_Data_Loc

    # Rule-based final score adjustment
    for i, row in D_final_loc.iterrows():
        if cracks and row["clay"] == "Yes" and "vert" in row["compname"].lower():
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (
            0 <= bedrock <= 10
            and row["fss"].lower() == "fao74"
            and "lithosols" in row["compname"].lower()
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (
            0 <= bedrock <= 10
            and row["fss"].lower() == "fao90"
            and "lithic leptosols" in row["compname"].lower()
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (
            10 < bedrock <= 30
            and row["fss"].lower() == "fao74"
            and ("rendzinas" in row["compname"].lower() or "rankers" in row["compname"].lower())
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (
            10 < bedrock <= 30
            and row["fss"].lower() == "fao90"
            and "leptosols" in row["compname"].lower()
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (bedrock is None or bedrock > 50) and any(
            term in row["compname"].lower()
            for term in ["lithosols", "leptosols", "rendzinas", "rankers"]
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 0.001

    D_final_loc["Score_Data_Loc"] = D_final_loc["Score_Data_Loc"] / np.nanmax(
        D_final_loc["Score_Data_Loc"]
    )
    D_final_loc = D_final_loc.sort_values("Score_Data_Loc", ascending=False)

    # Sorting and reindexing of final dataframe
    soilIDList_out = []

    # Group by 'compname_grp'
    for _, comp_grps_temp in D_final_loc.groupby("compname_grp", sort=False):
        comp_grps_temp = comp_grps_temp.sort_values("Score_Data_Loc", ascending=False).reset_index(
            drop=True
        )

        if len(comp_grps_temp) > 1:
            # Mark the first entry as True, all others as False
            comp_grps_temp["soilID_rank_final"] = [True] + [False] * (len(comp_grps_temp) - 1)
        else:
            comp_grps_temp["soilID_rank_final"] = [True]

        soilIDList_out.append(comp_grps_temp)

    # Reassemble the dataframe
    D_final_loc = pd.concat(soilIDList_out)

    # Sort dataframe by score and compname
    D_final_loc = D_final_loc.sort_values(
        ["Score_Data_Loc", "compname"], ascending=[False, True]
    ).reset_index(drop=True)

    # Assign final rank
    rank_id = 1
    Rank_DataLoc = []
    for is_rank_final in D_final_loc["soilID_rank_final"]:
        if is_rank_final:
            Rank_DataLoc.append(str(rank_id))
            rank_id += 1
        else:
            Rank_DataLoc.append("Not Displayed")

    D_final_loc["Rank_Data_Loc"] = Rank_DataLoc
    D_final_loc = D_final_loc.sort_values(
        ["soilID_rank_final", "Score_Data_Loc"], ascending=[False, False]
    ).reset_index(drop=True)
    D_final_loc[
        [
            "Score_Data",
            "horz_score",
            "Score_Data_Loc",
            "distance_score",
            "distance_score_norm",
        ]
    ] = D_final_loc[
        [
            "Score_Data",
            "horz_score",
            "Score_Data_Loc",
            "distance_score",
            "distance_score_norm",
        ]
    ].fillna(
        0.0
    )

    # Final scaling of similarity score by 'data_completeness'
    D_final_loc.Score_Data_Loc = D_final_loc.Score_Data_Loc * (
        0.25 if data_completeness < 25 else float(data_completeness) / 100
    )

    # Constructing rank list
    rank_list = []

    is_location_only = D_final_loc.missing_status.all() == "Location data only"
    for idx, row in D_final_loc.iterrows():
        if is_location_only:
            rank_entry = {
                "name": row.compname.capitalize(),
                "component": row.compname_grp.capitalize(),
                "mapunitID": row.mukey,
                "componentID": row.cokey,
                "score_data_loc": "",
                "rank_data_loc": "",
                "score_data": "",
                "rank_data": "",
                "score_loc": round(row.distance_score_norm, 3),
                "rank_loc": row.Rank_Loc,
                "componentData": row.missing_status,
            }
        else:
            rank_entry = {
                "name": row.compname.capitalize(),
                "component": row.compname_grp.capitalize(),
                "mapunitID": row.mukey,
                "componentID": row.cokey,
                "score_data_loc": round(row.Score_Data_Loc, 3),
                "rank_data_loc": row.Rank_Data_Loc,
                "score_data": round(row.Score_Data, 3),
                "rank_data": row.Rank_Data,
                "score_loc": round(row.distance_score_norm, 3),
                "rank_loc": row.Rank_Loc,
                "componentData": row.missing_status,
            }
        rank_list.append(rank_entry)

    # Setting up the return data structure
    model_version = 3
    metadata = {
        "location": "global",
        "model": "v3",
        "dataCompleteness": {"score": data_completeness, "text": text_completeness},
    }

    result = {"metadata": metadata, "soilRank": rank_list}

    # Save data if record_id is provided
    if record_id:
        saveRankOutput(record_id, model_version, json.dumps(result))

    return result


#####################################################################################################
#                                   getSoilLocationBasedUS                                          #
#####################################################################################################
def getSoilLocationBasedUS(lon, lat, plot_id):
    # Load in LAB to Munsell conversion look-up table
    color_ref = pd.read_csv("%s/LandPKS_munsell_rgb_lab.csv" % current_app.config["DATA_BACKEND"])

    # Load in SSURGO data from SoilWeb
    # soilweb_url = f"https://casoilresource.lawr.ucdavis.edu/api/landPKS.php?q=spn&lon={lon}&lat={lat}&r=1000" # current production API
    soilweb_url = f"https://soilmap2-1.lawr.ucdavis.edu/dylan/soilweb/api/landPKS.php?q=spn&lon={lon}&lat={lat}&r=1000"  # testing API
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
        mucompdataQry = f"SELECT component.mukey, component.cokey, component.compname, component.comppct_r, component.compkind, component.majcompflag, component.slope_r, component.elev_r, component.nirrcapcl, component.nirrcapscl, component.nirrcapunit, component.irrcapcl, component.irrcapscl, component.irrcapunit, component.taxorder, component.taxsubgrp FROM component WHERE mukey IN ({','.join(map(str, mukey_list))})"
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

    # --------------------------------------------------------------------------------------------------
    # Location based calculation
    # -----------------------------

    # --------------------------------------------------------------------------------------------------------------------------------------------------------
    #########################################################################################################################################
    # Individual probability
    # Based on Fan et al 2018 EQ 1, the conditional probability for each component is calculated by taking the sum of all occurances of a component
    # in the home and adjacent mapunits and dividing this by the sum of all map units and components. We have modified this approach so that each
    # instance of a component occurance is evaluated separately and assinged a weight and the max distance score for each component group is assigned
    # to all component instances.
    #########################################################################################################################################

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

    # --------------------------------------------------------------------------------------------------
    # Extracts horizon data
    # --------------------------------------------------------------------------------------------------
    if data_source == "SSURGO":
        # Convert JSON data to DataFrame
        muhorzdata_pd = pd.json_normalize(out["hz"])[
            [
                "cokey",
                "hzdept_r",
                "hzdepb_r",
                "chkey",
                "hzname",
                "sandtotal_r",
                "silttotal_r",
                "claytotal_r",
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

                # --------------------------------------------------------------------------------------------------
                # Location based calculation - Run on STATSGO b/c SSURGO horizon data was NULL
                # -----------------------------
                #########################################################################################################################################
                # Individual probability
                # Based on Fan et al 2018 EQ 1, the conditional probability for each component is calculated by taking the sum of all occurances of a component
                # in the home and adjacent mapunits and dividing this by the sum of all map units and components. We have modified this approach so that each
                # instance of a component occurance is evaluated separately and assinged a weight and the max distance score for each component group is assigned
                # to all component instances.
                #########################################################################################################################################

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
        mucompdata_pd[["cokey", "comppct_r", "compname", "slope_r"]],
        on="cokey",
        how="left",
    )

    # Replace "NULL" strings with numpy NaN
    muhorzdata_pd.replace("NULL", np.nan, inplace=True)

    # Filter out components with missing horizon depth data that aren't either a Series, Variant, or Family
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
        comppctB = getProfile_cokey_temp2["comppct"].iloc[0]
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

        # calculate LPKS depth aggregated data based on original depth sliced horizons
        snd_d, hz_depb = agg_data_layer(
            data=sand_pct_intpl["c_sandpct_intpl"],
            bottom=c_bottom_depths_temp["c_very_bottom"].iloc[0],
            depth=True,
        )
        cly_d = agg_data_layer(
            data=clay_pct_intpl["c_claypct_intpl"],
            bottom=c_bottom_depths_temp["c_very_bottom"].iloc[0],
        )
        txt_d = []
        for l in range(len(snd_d)):
            text_T = getTexture(
                row=None,
                sand=snd_d[l],
                silt=(100 - (snd_d[l] + cly_d[l])),
                clay=cly_d[l],
            )
            txt_d.append(text_T)
        txt_d = pd.Series(txt_d)
        txt_d.index = snd_d.index
        rf_d = agg_data_layer(
            data=cf_pct_intpl["c_cfpct_intpl"],
            bottom=c_bottom_depths_temp["c_very_bottom"].iloc[0],
        )
        cec_d = agg_data_layer(
            data=cec_intpl["c_cec_intpl"],
            bottom=c_bottom_depths_temp["c_very_bottom"].iloc[0],
        )
        ph_d = agg_data_layer(
            data=ph_intpl["c_ph_intpl"],
            bottom=c_bottom_depths_temp["c_very_bottom"].iloc[0],
        )
        ec_d = agg_data_layer(
            data=ec_intpl["c_ec_intpl"],
            bottom=c_bottom_depths_temp["c_very_bottom"].iloc[0],
        )
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

    # -------------------------------------------------------------------------------------------------------------
    # This extracts OSD color, texture, and CF data
    if data_source == "STATSGO":
        # If the condition is met, we perform the series of operations, otherwise, we set OSDhorzdata_pd to None
        if mucompdata_pd["compkind"].isin(OSD_compkind).any():
            try:
                # Generate series names
                series_name = [
                    urllib.parse.quote(re.sub("[0-9]+", "", compname))
                    for compname in mucompdata_pd["compname"]
                    if compname in OSD_compkind
                ]
                series_join = ",".join(series_name)

                # Fetch data from URL using requests
                series_url = f"https://casoilresource.lawr.ucdavis.edu/api/soil-series.php?q=site_hz&s={series_join}"
                response = requests.get(series_url, timeout=3)
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

                munsell_RGB_df = pd.DataFrame(munsell_RGB, columns=["r", "g", "b"])
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

        if mucompdata_pd["compkind"].isin(OSD_compkind).any():
            # Group data by cokey
            OSDhorzdata_group_cokey = [
                group for _, group in OSDhorzdata_pd.groupby("cokey", sort=False)
            ]

            # Initialize empty lists
            lab_lyrs = []
            munsell_lyrs = []
            lab_intpl_lyrs = []
            osd_hz_lyrs = []

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

            for group in OSDhorzdata_group_cokey:
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
                    # return(jsonify(c_bottom_depths_group.to_dict(orient='records')))
                    # Check if OSD depth adjustment is needed
                    if (
                        OSD_very_bottom < c_bottom_depths_group["c_very_bottom"].iloc[0]
                        and OSD_very_bottom < 120
                    ):
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

                    elif 0 < c_bottom_depths_group["c_very_bottom"].iloc[0] < 120 < OSD_very_bottom:
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
                        osd_hz_lyrs.append("")
                        munsell_lyrs.append("")
                    else:
                        # Aggregate data for each color dimension
                        l_d, osd_hz_d = agg_data_layer(
                            data=lab_intpl["l"],
                            bottom=OSD_very_bottom,
                            sd=4,
                            depth=True,
                        )
                        a_d = agg_data_layer(
                            data=lab_intpl["a"], bottom=OSD_very_bottom, sd=4
                        ).fillna("")
                        b_d = agg_data_layer(
                            data=lab_intpl["b"], bottom=OSD_very_bottom, sd=4
                        ).fillna("")

                        # Convert LAB values to a list of triplets
                        lab_parse = [[l, a, b] for l, a, b in zip(l_d, a_d, b_d)]
                        lab_lyrs.append(dict(zip(l_d.index, lab_parse)))
                        osd_hz_lyrs.append(dict(zip(osd_hz_d.index, osd_hz_d)))

                        # Convert LAB triplets to Munsell values
                        munsell_values = [
                            lab2munsell(color_ref, LAB_ref, LAB=lab)
                            if lab[0] and lab[1] and lab[2]
                            else ""
                            for lab in lab_parse
                        ]
                        munsell_lyrs.append(dict(zip(l_d.index, munsell_values)))

                    # Extract OSD Texture and Rock Fragment Data
                    if OSD_text_int[i] == "Yes" or OSD_rfv_int[i] == "Yes":
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

                        # If OSD bottom depth is greater than component depth and component depth is <120cm
                        if OSD_depth_remove:
                            # Remove data based on c_bottom_depths
                            OSD_sand_intpl = OSD_sand_intpl.loc[: c_bottom_depths.iloc[i, 2]]
                            OSD_clay_intpl = OSD_clay_intpl.loc[: c_bottom_depths.iloc[i, 2]]
                            OSD_rfv_intpl = OSD_rfv_intpl.loc[: c_bottom_depths.iloc[i, 2]]

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
                        getProfile_mod = getProfile_cokey[i]
                        compname_check = (
                            getProfile_mod["compname"]
                            .isin(group_sorted2[["compname"]].iloc[0])
                            .any()
                        )

                        if (
                            compname_check
                            and OSD_text_int[i] == "Yes"
                            and not group_sorted2["c_sandpct_intpl"].isnull().all()
                        ):
                            getProfile_mod["sandpct_intpl"] = group_sorted2["c_sandpct_intpl"]
                            getProfile_mod["claypct_intpl"] = group_sorted2["c_claypct_intpl"]

                        if (
                            compname_check
                            and OSD_rfv_int[i] == "Yes"
                            and not group_sorted2["c_cfpct_intpl"].isnull().all()
                        ):
                            getProfile_mod["rfv_intpl"] = group_sorted2["c_cfpct_intpl"]

                        getProfile_cokey[i] = getProfile_mod

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
                        if OSD_text_int[i] == "Yes":
                            snd_lyrs[i] = snd_d_osd.to_dict()
                            cly_lyrs[i] = cly_d_osd.to_dict()
                            txt_lyrs[i] = txt_d_osd.to_dict()

                        if OSD_rfv_int[i] == "Yes":
                            rf_lyrs[i] = rf_d_osd.to_dict()

                        # Update horizon layers if bottom depth is zero
                        if c_bottom_depths.iloc[i, 2] == 0:
                            hz_lyrs[i] = hz_depb_osd.to_dict()
                            mucompdata_pd.loc[i, "c_very_bottom"] = OSD_very_bottom

                        # Update cec, ph, and ec layers if they contain only a single empty string
                        for lyr in [cec_lyrs, ph_lyrs, ec_lyrs]:
                            if len(lyr[i]) == 1 and lyr[i][0] == "":
                                lyr[i] = dict(zip(hz_depb_osd.index, [""] * len(hz_depb_osd)))

                else:
                    OSDhorzdata_group_cokey[i] = group_sorted

                    # Create an empty dataframe with NaNs for lab_intpl
                    lab_intpl = pd.DataFrame(
                        np.nan,
                        index=np.arange(c_bottom_depths.iloc[i, 2]),
                        columns=["l", "a", "b"],
                    )
                    lab_intpl_lyrs.append(lab_intpl)

                    # Create dummy data for lab_lyrs
                    lab_dummy = [["", "", ""] for _ in range(len(hz_lyrs[i]))]
                    lab_lyrs.append(dict(zip(hz_lyrs[i].keys(), lab_dummy)))

                    # Empty osd_hz_lyrs entry
                    osd_hz_lyrs.append("")

                    # Create dummy data for munsell_lyrs
                    munsell_dummy = [""] * len(hz_lyrs[i])
                    munsell_lyrs.append(dict(zip(hz_lyrs[i].keys(), munsell_dummy)))

            # Series URL Generation
            # Initialize lists to store series URLs
            SDE_URL = []
            SEE_URL = []

            # Group data by 'cokey'
            OSDhorzdata_group_cokey = [g for _, g in OSDhorzdata_pd.groupby("cokey", sort=False)]

            for group in OSDhorzdata_group_cokey:
                # Check if compkind is not in OSD_compkind or if series contains any null values
                if (
                    mucompdata_pd.loc[i]["compkind"] not in OSD_compkind
                    or group["series"].isnull().any()
                ):
                    SDE_URL.append("")
                    SEE_URL.append("")
                else:
                    # Extract compname, convert to lowercase, remove trailing numbers, and replace spaces with underscores
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
            osd_hz_lyrs = []
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
                osd_hz_lyrs.append("")
                munsell_lyrs.append(dict(zip(keys, munsell_dummy)))

                # Append empty URLs
                SDE_URL.append("")
                SEE_URL.append("")

    else:
        # Initialize lists to store data layers and URLs
        lab_lyrs = []
        lab_intpl_lyrs = []
        osd_hz_lyrs = []
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
            osd_hz_lyrs.append("")
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

    # Create a new column 'soilID_rank' which will be True for the first row in each group sorted by 'distance' and False for other rows
    mucompdata_pd = mucompdata_pd.sort_values(["compname_grp", "distance"])
    mucompdata_pd["soilID_rank"] = ~mucompdata_pd.duplicated("compname_grp", keep="first")

    # Assign the minimum distance for each group to a new column 'min_dist'
    mucompdata_pd["min_dist"] = mucompdata_pd.groupby("compname_grp")["distance"].transform("first")

    mucompdata_pd = mucompdata_pd.reset_index(drop=True)

    # -------------------------------------------------------------------------------------------------------------
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
            + ",".join(map(str, cokey_list))
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

        conn = getDataStore_Connection()
        cur = conn.cursor()
        sql1 = (
            "SELECT lu.ES_ID, lu.Synonym FROM ESD_class_synonym_list AS lu WHERE lu.Synonym IN (%s)"
            % ", ".join(map(str, ecositeID))
        )
        cur.execute(sql1)
        results = cur.fetchall()

        if not results:
            ESDcompdata_pd["ecoclassid_update"] = ESDcompdata_pd["ecoclassid"]
        else:
            esd_lookup = pd.DataFrame(results, columns=["ecoclassid_update", "ecoclassid"])
            ESDcompdata_pd = pd.merge(ESDcompdata_pd, esd_lookup, on="ecoclassid", how="left")
            ESDcompdata_pd["ecoclassid_update"].fillna(ESDcompdata_pd["ecoclassid"], inplace=True)

        ecositeID = ESDcompdata_pd["ecoclassid_update"].tolist()
        ESD_geo = [ecosite[:4] for ecosite in ESDcompdata_pd["ecoclassid_update"].unique()]

        # Filter out 'nan' from ESD_geo
        ESD_geo = [geo for geo in ESD_geo if str(geo) != "nan"]

        # Extract class URL data
        class_url_list = []
        for geo in ESD_geo:
            class_url = (
                f"https://edit.jornada.nmsu.edu/services/downloads/esd/{geo}/class-list.json"
            )
            try:
                with urllib.request.urlopen(class_url, timeout=4) as response:
                    class_data = json.load(response)
                    class_url_list.append(
                        pd.json_normalize(class_data["ecoclasses"])[["id", "legacyId"]]
                    )
            except:
                pass

        # Concatenate all the class URL data into a single dataframe
        ESD_list_pd = (
            pd.concat(class_url_list)
            if class_url_list
            else pd.DataFrame(columns=["id", "legacyID"])
        )

        # Extract ESD URL data
        ESD_URL = []
        for eco_id in ecositeID:
            if eco_id in ESD_list_pd["id"].tolist() or eco_id in ESD_list_pd["legacyId"].tolist():
                ecosite_edit_id = ESD_list_pd[
                    ESD_list_pd.apply(lambda r: r.str.contains(eco_id, case=False).any(), axis=1)
                ]["id"].values[0]
                ES_URL_t = f"https://edit.jornada.nmsu.edu/catalogs/esd/{ecosite_edit_id[1:5]}/{ecosite_edit_id}"
                ESD_URL.append(ES_URL_t)
            else:
                ESD_URL.append("")

        ESDcompdata_pd["esd_url"] = ESD_URL

        # Assign missing ESD for components that have other instances with an assinged ESD
        if ESDcompdata_pd is not None:
            if (
                ESDcompdata_pd.ecoclassid_update.isnull().any()
                or ESDcompdata_pd.ecoclassname.isnull().any()
            ):
                ESDcompdata_pd["compname_grp"] = ESDcompdata_pd.compname.str.replace(r"[0-9]+", "")

                # Group by component name without numbers
                ESDcompdata_pd_comp_grps = [
                    group for _, group in ESDcompdata_pd.groupby("compname_grp", sort=False)
                ]

                ecoList_out = []
                for group in ESDcompdata_pd_comp_grps:
                    if (
                        len(group) == 1
                        or group.ecoclassid_update.isnull().all()
                        or group.ecoclassname.isnull().all()
                    ):
                        ecoList_out.append(group)
                    elif (
                        group.ecoclassid_update.isnull().any()
                        and len(group.ecoclassid_update.dropna().unique()) == 1
                    ) and (
                        group.ecoclassname.isnull().any()
                        and len(group.ecoclassname.dropna().unique()) == 1
                    ):
                        unique_ecoclassid = group.ecoclassid_update.dropna().unique()[0]
                        unique_ecoclassname = group.ecoclassname.dropna().unique()[0]
                        unique_url = next((url for url in group.esd_url.unique() if url), "")

                        group["ecoclassid_update"] = unique_ecoclassid
                        group["ecoclassname"] = unique_ecoclassname
                        group["esd_url"] = unique_url

                        ecoList_out.append(group)
                    else:
                        ecoList_out.append(group)

                ESDcompdata_pd = pd.concat(ecoList_out)

            ESDcompdata_pd = ESDcompdata_pd.drop_duplicates(subset="cokey", keep="first")

            esd_comp_list = []
            for _, group in ESDcompdata_pd.groupby("cokey", sort=False):
                if group["ecoclassname"].isnull().values.any():
                    esd_comp_list.append({"ecoclassid": "", "ecoclassname": "", "esd_url": ""})
                else:
                    esd_comp_list.append(
                        {
                            "ecoclassid": group["ecoclassid_update"].tolist()[0],
                            "ecoclassname": group["ecoclassname"].tolist()[0],
                            "esd_url": group["esd_url"].tolist()[0],
                        }
                    )

    else:
        # Initialize the ecosite data list
        esd_comp_list = [
            {"ecoclassid": "", "ecoclassname": "", "esd_url": ""} for _ in range(len(mucompdata_pd))
        ]

    # Add ecosite data to mucompdata_pd for testing output.
    # In cases with multiple ESDs per component, only take the first.
    if ESDcompdata_pd is None:
        mucompdata_pd["ecoclassid_update"] = np.nan
        mucompdata_pd["ecoclassname"] = np.nan
    else:
        # Merge data with the first unique ecosite data per component
        mucompdata_pd = mucompdata_pd.merge(
            ESDcompdata_pd[["cokey", "ecoclassid_update", "ecoclassname"]].drop_duplicates(
                "cokey", keep="first"
            ),
            on="cokey",
            how="left",
        )

    # Note: Need to devise a way to improve the identification of bedrock or restrictive layers using both comp data and OSD.
    #      There are many instances where the last one or two component horizons are missing texture data and the OSD bottom depth
    #      is shallower than the component bottom depth, indicating a shallower profile. But the horizon designations are generic, e.g.,
    #      H1, H2, H3; thus not allowing their use in indentifying bedrock.

    # --------------------------------------------------------------------------------------------------------------------------
    # SoilIDList output

    # ------------------------------------------------------------
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
        osd_hz_lyrs,
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
        osd_hz_lyrs,
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
                    "model": "v2",
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
        saveModelOutput(
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
            "model": "v2",
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


#####################################################################################################
#                                   rankPredictionUS                                                #
#####################################################################################################
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
    # TODO: Future testing to see if deltaE2000 values should be incorporated into site data
    # use 'getColor_deltaE2000_OSD_pedon' and helper functions located in local_functions_SoilID_v3.py

    # ------------------------------------------------------------------------------------------------
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
    soil_df = soil_df.dropna(how="all", subset=relevant_columns)

    if soil_df.empty:
        soil_df = None

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
                        np.nan,
                        index=np.arange(add_length),
                        columns=np.arange(add_columns),
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
            [p_compname, p_sandpct_intpl, p_claypct_intpl, p_cfg_intpl, p_lab_intpl],
            axis=1,
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
                elev_data["USGS_Elevation_Point_Query_Service"]["Elevation_Query"]["Elevation"],
                3,
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

    # --------------------------------------------------------------------------------------------------------------------------------------
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
        modelRun = loadModelOutput(plot_id)
        if modelRun:
            record_id = modelRun[0]
            soilIDRank_output_pd = pd.read_csv(cStringIO.StringIO(modelRun[2]))
            mucompdata_pd = pd.read_csv(cStringIO.StringIO(modelRun[3]))
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

    # Create soil depth DataFrame and subset component depths based on max user depth if no bedrock specified
    c_bottom_depths = mucompdata_pd[["cokey", "compname", "c_very_bottom"]]
    c_bottom_depths.columns = ["cokey", "compname", "bottom_depth"]
    slices_of_soil = pd.concat([p_bottom_depth, c_bottom_depths], axis=0).reset_index(drop=True)
    compnames = mucompdata_pd[["compname", "compname_grp"]]

    # Generate a matrix storing a flag describing soil (1) vs. non-soil (0) at each slice
    # note that this will truncate a profile to the max depth of actual data

    # Determine the maximum depth based on bedrock and user input
    if bedrock is None:
        max_depth = min(p_bottom_depth.bottom_depth.values[0], 120)
    else:
        max_depth = 120

    # Truncate depths in slices_of_soil that exceed max_depth
    slices_of_soil.loc[slices_of_soil.bottom_depth > max_depth, "bottom_depth"] = max_depth

    # Calculate the non-soil slices for each entry in slices_of_soil
    slices_of_non_soil = max_depth - slices_of_soil.bottom_depth

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
        horz_vars.extend(
            [group.reset_index(drop=True).loc[pedon_slice_index] for group in horz_vars_group_cokey]
        )

        # Calculate similarity for each depth slice
        dis_mat_list = []
        for (
            i
        ) in (
            soil_matrix.index
        ):  # i should be an index of p_hz_data depth slices, e.g. if user only enters 100-120cm data, then i = 100:120
            slice_data = [horz.loc[i] for horz in horz_vars]
            sliceT = pd.concat(slice_data, axis=1).T
            """
            Not all depth slices have the same user recorded data. Here we filter out data columns with missing data and use that to subset the component data.
            If certain components are missing lots of data and the filering results in less than 2 soil properties, than we filter out data columns with missing USER data
            and components with missing data will later be assigned the max dissimilarity across all horizons
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

    ##---Site Data Similarity
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
            [
                compnames,
                pd.Series(D_site_hz[0][1:]),
                pd.Series(np.repeat(1.0, len(compnames))),
            ],
            axis=1,
        )

    # When only D_horz is available
    elif D_horz is not None:
        D_final = pd.concat(
            [
                compnames,
                pd.Series(D_horz[0][1:]),
                pd.Series(np.repeat(1.0, len(compnames))),
            ],
            axis=1,
        )

    # When only D_site is available
    elif D_site is not None:
        D_final = pd.concat(
            [
                compnames,
                pd.Series(D_site[0][1:]),
                pd.Series(np.repeat(1.0, len(compnames))),
            ],
            axis=1,
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

    # Code options for production API/testing output
    # ----------------------------------------------------------------
    # #Data output for testing
    # D_final_loc = pd.merge(D_final, mucompdata_pd[['compname', 'cokey', 'mukey', 'distance_score', 'distance_score_norm', 'clay', 'taxorder', 'taxsubgrp', 'OSD_text_int', 'OSD_rfv_int', 'data_source', 'Rank_Loc', 'majcompflag', 'comppct_r', 'distance', 'nirrcapcl', 'nirrcapscl',
    #    'nirrcapunit', 'irrcapcl', 'irrcapscl', 'irrcapunit', 'ecoclassid_update', 'ecoclassname']], on='compname', how='left')

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
    # ----------------------------------------------------------------
    # Data formatting for testing

    # # Update LCC_I column
    # D_final_loc['LCC_I'] = np.where(
    #     (D_final_loc['irrcapcl'] == 'nan') | (D_final_loc['irrcapscl'] == 'nan'),
    #     None,
    #     D_final_loc['irrcapcl'] + "-" + D_final_loc['irrcapscl']
    # )
    #
    # # Update LCC_NI column
    # D_final_loc['LCC_NI'] = np.where(
    #     (D_final_loc['nirrcapcl'] == 'nan') | (D_final_loc['nirrcapscl'] == 'nan'),
    #     None,
    #     D_final_loc['nirrcapcl'] + "-" + D_final_loc['nirrcapscl']
    # )
    # ----------------------------------------------------------------

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
        model_version = 2
        saveRankOutput(record_id, model_version, json.dumps(output_data))

    return output_data

    # Data return for testing
    # return(D_final_loc[['compname', 'compname_grp', 'Rank_Loc', 'distance_score_norm',  'Rank_Data', 'Score_Data', 'Rank_Data_Loc', 'Score_Data_Loc','ecoclassid_update', 'ecoclassname', 'LCC_I', 'LCC_NI', 'taxorder', 'taxsubgrp', 'majcompflag', 'comppct_r', 'distance']])
    # ----------------------------------------------------------------


#####################################################################################################
#                                          getSoilGridsGlobal                                       #
#####################################################################################################
def getSoilGridsGlobal(lon, lat, plot_id=None):
    # -------------------------------------------------------------------------------------------

    ########### SoilGrids250  ############################################
    # Construct the SoilGrids API v2 URL
    sg_api = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=cfvo&property=cec&property=clay&property=phh2o&property=sand&value=mean"

    try:
        # Make the API request using the requests library
        response = requests.get(sg_api, timeout=160)
        response.raise_for_status()  # Check for unsuccessful status codes
        sg_out = response.json()

    except requests.RequestException as e:
        # Log the error and set the status to unavailable
        if plot_id is not None:
            saveSoilGridsOutput(plot_id, 1, json.dumps({"status": "unavailable"}))
        sg_out = {"status": "unavailable"}

    # Use the 'extract_values' function to extract specific keys from the JSON
    top_depths = extract_values(sg_out, "top_depth")
    bottom_depths = extract_values(sg_out, "bottom_depth")
    values = extract_values(sg_out, "mean")
    names = extract_values(sg_out, "name")

    # Convert extracted lists to DataFrames
    df_top_depth = pd.DataFrame(top_depths, columns=["top_depth"])
    df_bottom_depth = pd.DataFrame(bottom_depths, columns=["bottom_depth"])
    df_values = pd.DataFrame(values, columns=["value"])

    # Repeat names for each 6 rows since there are 6 depths
    df_names = pd.DataFrame(names * 6, columns=["prop"])

    # Concatenate the DataFrames to form a single DataFrame
    sg_data = pd.concat([df_names, df_top_depth, df_bottom_depth, df_values], axis=1)

    # Pivot the data based on bottom depth and property name, setting values to be the 'value' column
    sg_data_w = sg_data.pivot(index="bottom_depth", columns="prop", values="value")

    # Add the top and bottom depth columns to the wide DataFrame
    sg_data_w["hzdept_r"] = df_top_depth["top_depth"].head(6).tolist()
    sg_data_w["hzdepb_r"] = df_bottom_depth["bottom_depth"].head(6).tolist()

    # Reset the index for the resulting DataFrame
    sg_data_w.reset_index(drop=True, inplace=True)

    # Check if all values in the specified columns are NaN
    if sg_data_w[["sand", "clay", "cfvo"]].isnull().all().all():
        if plot_id is not None:
            saveSoilGridsOutput(plot_id, 1, json.dumps({"status": "unavailable"}))
        return {"status": "unavailable"}
    else:
        # Apply the factor to the specific columns
        cols_to_multiply = ["sand", "clay", "cfvo", "phh2o", "cec"]
        sg_data_w[cols_to_multiply] = sg_data_w[cols_to_multiply].multiply(0.1)
        # Calculate silt and texture for the DataFrame
        sg_data_w["silt"] = sg_data_w.apply(silt_calc, axis=1)
        sg_data_w["texture"] = sg_data_w.apply(getTexture, axis=1)

        # Fetch SG wRB Taxonomy
        # Construct the API URL for fetching soil data
        api_url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?lon={lon}&lat={lat}&number_classes=3"

        # Fetch data from the API
        try:
            with request.urlopen(api_url, timeout=6) as response:
                sg_tax = json.load(response)
        except Exception as e:
            # Handle data fetch failure
            if plot_id is not None:
                # Assuming the function `saveSoilGridsOutput` exists elsewhere in the code
                saveSoilGridsOutput(plot_id, 1, json.dumps({"status": "unavailable"}))
            sg_tax = None

        # If data was successfully fetched, process it
        if sg_tax:
            # Create DataFrame from fetched data
            sg_tax_prob = pd.DataFrame(sg_tax["wrb_class_probability"], columns=["WRB_tax", "Prob"])

            # Sort DataFrame by probability
            sg_tax_prob.sort_values("Prob", ascending=False, inplace=True)

            # Fetch descriptions for the soil classifications
            # Assuming the function `getSG_descriptions` exists elsewhere in the code
            WRB_Comp_Desc = getSG_descriptions(sg_tax_prob["WRB_tax"].tolist())

            # Merge the descriptions with the data
            TAXNWRB_pd = pd.merge(sg_tax_prob, WRB_Comp_Desc, on="WRB_tax", how="left")
            TAXNWRB_pd.index = ["Rank1", "Rank2", "Rank3"]
        else:
            # If data wasn't fetched, create an unavailable status DataFrame
            TAXNWRB_pd = pd.DataFrame(
                {
                    "WRB_tax": [""],
                    "Prob": [""],
                    "Description_en": [""],
                    "Management_en": [""],
                    "Description_es": [""],
                    "Management_es": [""],
                    "Description_ks": [""],
                    "Management_ks": [""],
                    "Description_fr": [""],
                    "Management_fr": [""],
                }
            )

        # Avoid repetitive code by creating a function to get data and aggregate
        def get_and_agg(variable, sg_data_w, bottom, return_depth=False):
            pd_int = getProfile_SG(sg_data_w, variable, c_bot=False)
            if return_depth == True:
                pd_lpks, lpks_depths = agg_data_layer(
                    data=pd_int.var_pct_intpl, bottom=bottom, depth=True
                )
                return (pd_lpks.replace(np.nan, ""), lpks_depths)
            else:
                pd_lpks = agg_data_layer(data=pd_int.var_pct_intpl, bottom=bottom, depth=False)
                return pd_lpks.replace(np.nan, "")

        sand_pd_lpks, lpks_depths = get_and_agg("sand", sg_data_w, bottom, return_depth=True)

        variables = ["clay", "cfvo", "phh2o", "cec"]
        dataframes = {var: get_and_agg(var, sg_data_w, bottom) for var in variables}

        clay_pd_lpks = dataframes["clay"]
        rfv_pd_lpks = dataframes["cfvo"]
        pH_pd_lpks = dataframes["phh2o"]
        cec_pd_lpks = dataframes["cec"]

        # Additional texture calculation
        texture_pd = pd.DataFrame([sand_pd_lpks, clay_pd_lpks]).T
        texture_pd.columns = ["sand", "clay"]
        texture_pd["silt"] = texture_pd.apply(silt_calc, axis=1)
        texture_pd_lpks = texture_pd.apply(getTexture, axis=1).replace([None], "")

        # SoilGrids API call version 1
        model_version = 1

        # Define keys and corresponding values for the 'components' dictionary
        component_keys = [
            "compname",
            "probability",
            "descriptionEN",
            "managementEN",
            "descriptionES",
            "managementES",
            "descriptionKS",
            "managementKS",
            "descriptionFR",
            "managementFR",
        ]
        component_values = [
            TAXNWRB_pd.WRB_tax,
            TAXNWRB_pd.Prob,
            TAXNWRB_pd.Description_en,
            TAXNWRB_pd.Management_en,
            TAXNWRB_pd.Description_es,
            TAXNWRB_pd.Management_es,
            TAXNWRB_pd.Description_ks,
            TAXNWRB_pd.Management_ks,
            TAXNWRB_pd.Description_fr,
            TAXNWRB_pd.Management_fr,
        ]

        # Create 'components' dictionary using a loop
        components_dict = {
            k: dict(zip(TAXNWRB_pd.index, v)) for k, v in zip(component_keys, component_values)
        }

        # Create remaining dictionaries for the SoilGrids dictionary
        remaining_keys = [
            "bottom_depth",
            "texture",
            "sand",
            "clay",
            "rock_fragments",
            "ph",
            "cec",
        ]
        remaining_values = [
            lpks_depths,
            texture_pd_lpks,
            sand_pd_lpks,
            clay_pd_lpks,
            rfv_pd_lpks,
            pH_pd_lpks,
            cec_pd_lpks,
        ]

        # Create the overall SoilGrids dictionary
        SoilGrids = {
            "components": components_dict,
            "bottom_depth": dict(list(zip(lpks_depths.index, lpks_depths))),
            "bedrock": bottom,
        }
        SoilGrids.update(
            {k: dict(list(zip(v.index, v))) for k, v in zip(remaining_keys, remaining_values)}
        )

        # Define the metadata dictionary
        metadata = {
            "location": "global",
            "model": "v1",
            "unit_measure": {
                "depth": "cm",
                "cec": "cmol(c)/kg",
                "clay": "%",
                "rock_fragments": "cm3/100cm3",
                "sand": "%",
            },
        }

        # If a plot_id is provided, save the SoilGrids output
        if plot_id is not None:
            saveSoilGridsOutput(
                plot_id,
                model_version,
                json.dumps({"metadata": metadata, "soilGrids": SoilGrids}),
            )

        # Return the final result
        return {"metadata": metadata, "soilGrids": SoilGrids}


#####################################################################################################
#                                          getSoilGridsUS                                           #
#####################################################################################################
def getSoilGridsUS(lon, lat, plot_id=None):
    ########### SoilGrids250  ############################################
    # Construct the SoilGrids API v2 URL
    sg_api = f"https://rest.isric.org/soilgrids/v2.0/properties/query?lon={lon}&lat={lat}&property=cfvo&property=cec&property=clay&property=phh2o&property=sand&value=mean"

    try:
        # Make the API request using the requests library
        response = requests.get(sg_api, timeout=60)
        response.raise_for_status()  # Check for unsuccessful status codes
        sg_out = response.json()

    except requests.RequestException as e:
        # Log the error and set the status to unavailable
        if plot_id is not None:
            saveSoilGridsOutput(plot_id, 1, json.dumps({"status": "unavailable"}))
        sg_out = {"status": "unavailable"}

    # Use the 'extract_values' function to extract specific keys from the JSON
    top_depths = extract_values(sg_out, "top_depth")
    bottom_depths = extract_values(sg_out, "bottom_depth")
    values = extract_values(sg_out, "mean")
    names = extract_values(sg_out, "name")

    # Convert extracted lists to DataFrames
    df_top_depth = pd.DataFrame(top_depths, columns=["top_depth"])
    df_bottom_depth = pd.DataFrame(bottom_depths, columns=["bottom_depth"])
    df_values = pd.DataFrame(values, columns=["value"])

    # Repeat names for each 6 rows since there are 6 depths
    df_names = pd.DataFrame(names * 6, columns=["prop"])

    # Concatenate the DataFrames to form a single DataFrame
    sg_data = pd.concat([df_names, df_top_depth, df_bottom_depth, df_values], axis=1)

    # Pivot the data based on bottom depth and property name, setting values to be the 'value' column
    sg_data_w = sg_data.pivot(index="bottom_depth", columns="prop", values="value")

    # Add the top and bottom depth columns to the wide DataFrame
    sg_data_w["hzdept_r"] = df_top_depth["top_depth"].head(6).tolist()
    sg_data_w["hzdepb_r"] = df_bottom_depth["bottom_depth"].head(6).tolist()

    # Reset the index for the resulting DataFrame
    sg_data_w.reset_index(drop=True, inplace=True)

    # Check if all values in the specified columns are NaN
    if sg_data_w[["sand", "clay", "cfvo"]].isnull().all().all():
        if plot_id is not None:
            saveSoilGridsOutput(plot_id, 1, json.dumps({"status": "unavailable"}))
        return {"status": "unavailable"}
    else:
        # Apply the factor to the specific columns
        cols_to_multiply = ["sand", "clay", "cfvo", "phh2o", "cec"]
        sg_data_w[cols_to_multiply] = sg_data_w[cols_to_multiply].multiply(0.1)

        # Calculate silt and texture for the DataFrame
        sg_data_w["silt"] = sg_data_w.apply(silt_calc, axis=1)
        sg_data_w["texture"] = sg_data_w.apply(getTexture, axis=1)

        # # Pull component descriptions
        # ST_Comp_List = []
        # ST_Comp_List = TAXOUSDA_pd["TAXOUSDA"].drop_duplicates(keep='first').tolist()
        # ST_Comp_Desc = getST_descriptions(ST_Comp_List)
        # ST_Comp_Desc = ST_Comp_Desc.rename(index={0:'Rank1', 1:'Rank2', 2:'Rank3'})
        # TAXOUSDA_pd = pd.concat([TAXOUSDA_pd, ST_Comp_Desc], axis=1)

        # Avoid repetitive code by creating a function to get data and aggregate
        def get_and_agg(variable, sg_data_w, bottom, return_depth=False):
            pd_int = getProfile_SG(sg_data_w, variable, c_bot=False)
            if return_depth == True:
                pd_lpks, lpks_depths = agg_data_layer(
                    data=pd_int.var_pct_intpl, bottom=bottom, depth=True
                )
                return (pd_lpks.replace(np.nan, ""), lpks_depths)
            else:
                pd_lpks = agg_data_layer(data=pd_int.var_pct_intpl, bottom=bottom, depth=False)
                return pd_lpks.replace(np.nan, "")

        sand_pd_lpks, lpks_depths = get_and_agg("sand", sg_data_w, bottom, return_depth=True)

        variables = ["clay", "cfvo", "phh2o", "cec"]
        dataframes = {var: get_and_agg(var, sg_data_w, bottom) for var in variables}

        clay_pd_lpks = dataframes["clay"]
        rfv_pd_lpks = dataframes["cfvo"]
        pH_pd_lpks = dataframes["phh2o"]
        cec_pd_lpks = dataframes["cec"]

        # Additional texture calculation
        texture_pd = pd.DataFrame([sand_pd_lpks, clay_pd_lpks]).T
        texture_pd.columns = ["sand", "clay"]
        texture_pd["silt"] = texture_pd.apply(silt_calc, axis=1)
        texture_pd_lpks = texture_pd.apply(getTexture, axis=1).replace([None], "")

        # SoilGrids API call version 1
        model_version = 1

        # Define properties to be included in the SoilGrids dictionary
        properties = ["texture", "sand", "clay", "rock_fragments", "ph", "cec"]
        dataframes = {
            "texture": texture_pd_lpks,
            "sand": sand_pd_lpks,
            "clay": clay_pd_lpks,
            "rock_fragments": rfv_pd_lpks,
            "ph": pH_pd_lpks,
            "cec": cec_pd_lpks,
        }

        # Construct the nested dictionary using dictionary comprehension
        SoilGrids = {
            "bottom_depth": dict(lpks_depths.items()),
            "bedrock": bottom,
            **{prop: dict(dataframes[prop].items()) for prop in properties},
        }
        # Define the common metadata and SoilGrids data to be used in both saving and returning.
        data = {
            "metadata": {
                "location": "us",
                "model": "v1",
                "unit_measure": {
                    "depth": "cm",
                    "cec": "cmol(c)/kg",
                    "clay": "%",
                    "rock_fragments": "cm3/100cm3",
                    "sand": "%",
                },
            },
            "soilGrids": SoilGrids,
        }

        # Save the data if plot_id is provided
        if plot_id:
            saveSoilGridsOutput(plot_id, model_version, json.dumps(data))

        # Return the constructed data
        return data

    # -------------------------------------------------------------------------------------------
    # # This code was used for SoilGrids v1

    # def getST_descriptions(ST_Comp_List):
    #     try:
    #         conn = getDataStore_Connection()
    #         cur = conn.cursor()
    #         ST_Comp_List = [x.encode('UTF8') for x in ST_Comp_List]
    #         sql = 'SELECT Suborder, Description_en, Management_en, Description_es, Management_es, Description_ks, Management_ks, Description_fr, Management_fr FROM soil_taxonomy_desc WHERE Suborder IN (' + ''.join(str(ST_Comp_List)[1:-1]) + ')'
    #         cur.execute(sql)
    #         results = cur.fetchall()
    #         data = pd.DataFrame(list(results))
    #         data.columns = ['Suborder', 'Description_en', 'Management_en', 'Description_es', 'Management_es', 'Description_ks', 'Management_ks', 'Description_fr', 'Management_fr']
    #         return data
    #     except Exception, err:
    #         print err
    #         return None
    #     finally:
    #         conn.close()

    # # This function applies a cubic spline model to interpolate values at every 1cm for the SoilGrids data

    # def cspline_soil_lpks(data):
    #     xm=[0,5,15,30,60,100,199]
    #     ym=data.loc[['M.sl1','M.sl2','M.sl3','M.sl4','M.sl5','M.sl6','M.sl7'],0].tolist()
    #     x_int=np.arange(0, 200, 1)
    #     cs = CubicSpline(xm,ym,bc_type='natural')
    #     int_vals=cs(x_int)
    #     data=pd.Series(int_vals).apply(pd.to_numeric)
    #     return pd.DataFrame(data=data, columns=['intp_vals'])
    # -------------------------------------------------------------------------------------------
