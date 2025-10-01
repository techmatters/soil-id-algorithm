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

# Standard libraries
import collections
import io
import logging
import re
from dataclasses import dataclass

# Third-party libraries
import numpy as np
import pandas as pd
from pandas import json_normalize

# local libraries
import soil_id.config

from .color import getProfileLAB, lab2munsell, munsell2rgb
from .services import get_elev_data, get_soil_series_data, get_soilweb_data, sda_return
from .soil_sim import soil_sim
from .utils import (
    aggregate_data,
    create_new_layer,
    create_new_layer_osd,
    drop_cokey_horz,
    extract_mucompdata_STATSGO,
    extract_muhorzdata_STATSGO,
    fill_missing_comppct_r,
    getCF_fromClass,
    getClay,
    getOSDCF,
    getProfile,
    getSand,
    getTexture,
    gower_distances,
    max_comp_depth,
    process_distance_scores,
    process_horizon_data,
    process_site_data,
    update_esd_data,
    update_intpl_data,
)

# entry points
# getSoilLocationBasedGlobal
# list_soils
# rank_soils
# rankPredictionGlobal
# getSoilGridsGlobal

# when a site is created, call list_soils/getSoilLocationBasedGlobal.
# when a site is created, call getSoilGridsGlobal
# after user has collected data, call rank_soils/rankPredictionGlobal.

# set Pandas dataframe options
pd.set_option("future.no_silent_downcasting", True)


@dataclass
class SoilListOutputData:
    soil_list_json: dict
    rank_data_csv: str
    map_unit_component_data_csv: str


############################################################################################
#                                   list_soils                                 #
############################################################################################
def list_soils(lon, lat):
    # Load in LAB to Munsell conversion look-up table
    color_ref = pd.read_csv(soil_id.config.MUNSELL_RGB_LAB_PATH)
    LAB_ref = color_ref[["cielab_l", "cielab_a", "cielab_b"]]
    munsell_ref = color_ref[["hue", "value", "chroma"]]

    out = get_soilweb_data(lon, lat)

    OSD_compkind = ["Series", "Variant", "Family", "Taxadjunct"]
    # Check if point is in a NOTCOM area, and if so then infill with STATSGO from NRCS SDA
    # NOTCOM means not in SSURGO data
    if not out["spn"]:
        # Extract STATSGO data at point
        mucompdata_pd = extract_mucompdata_STATSGO(lon, lat)
        mucompdata_pd = process_site_data(mucompdata_pd)
        if mucompdata_pd is None or mucompdata_pd.empty:
            return "Soil ID not available in this area"
        else:
            data_source = "STATSGO"
    else:
        mucompdata_pd = pd.json_normalize(out["spn"])
        mucompdata_pd = process_site_data(mucompdata_pd)

        # For SSURGO, filter out data for distances over 1000m
        mucompdata_pd = mucompdata_pd[mucompdata_pd["distance"] <= 1000]

        if mucompdata_pd.empty:
            # Extract STATSGO data at point
            mucompdata_pd = extract_mucompdata_STATSGO(lon, lat)
            mucompdata_pd = process_site_data(mucompdata_pd)
            if mucompdata_pd.empty:
                return "Soil ID not available in this area"
            else:
                data_source = "STATSGO"
        else:
            data_source = "SSURGO"
    # Set the Exponential Decay Coefficient based on data source
    if data_source == "STATSGO":
        ExpCoeff = -0.0002772
    elif data_source == "SSURGO":
        ExpCoeff = -0.008

    if mucompdata_pd.empty:
        return "Soil ID not available in this area"
    if mucompdata_pd["compname"].iloc[0] == "Water" and mucompdata_pd["comppct_r"].iloc[0] == "100":
        return "Soil ID not available in this area"

    # Process and fill missing or zero values in the 'comppct_r' column.
    mucompdata_pd = fill_missing_comppct_r(mucompdata_pd)

    # --------------------------------------------------------------------
    # Location based calculation
    # --------------------------------------------------------------------
    # Process distance scores and perform group-wise aggregations.
    mucompdata_pd = process_distance_scores(mucompdata_pd, ExpCoeff, compkind_filter=True)

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
        muhorzdata_pd = pd.json_normalize(out["hz"])
        muhorzdata_pd = process_horizon_data(muhorzdata_pd)

        # Filter rows based on component keys
        muhorzdata_pd = muhorzdata_pd.loc[muhorzdata_pd["cokey"].isin(comp_key)]

        # If dataset is empty and none of the components are Series, switch to STATSGO
        # and rerun site data extraction
        if (
            muhorzdata_pd[["hzdept_r", "hzdepb_r"]].isnull().all().all()
            and not mucompdata_pd["compkind"].isin(OSD_compkind).any()
        ):
            # STATSGO Component Data Processing
            mucompdata_pd = extract_mucompdata_STATSGO(lon, lat)

            # Process the mucompdata results
            if mucompdata_pd is None:
                return "Soil ID not available in this area"
            else:
                data_source = "STATSGO"
                ExpCoeff = -0.0002772  # Expotential decay coefficient: 0.25 @ ~5km

                # Process and fill missing or zero values in the 'comppct_r' column.
                mucompdata_pd = fill_missing_comppct_r(mucompdata_pd)

                # --------------------------------------------------------------------
                # Location based calculation
                # --------------------------------------------------------------------
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
                muhorzdata_pd = process_horizon_data(muhorzdata_pd)
    elif data_source == "STATSGO":
        # STATSGO Horizon Data Query
        muhorzdata_pd = extract_muhorzdata_STATSGO(mucompdata_pd)
        muhorzdata_pd = process_horizon_data(muhorzdata_pd)

    # Merge muhorzdata_pd with selected columns from mucompdata_pd
    muhorzdata_pd = pd.merge(
        muhorzdata_pd,
        mucompdata_pd[["cokey", "comppct_r", "compname", "cond_prob", "slope_r"]],
        on="cokey",
        how="left",
    )

    # Filter out components with missing horizon depth data that aren't either
    # a Series, Variant, or Family
    filter_condition = muhorzdata_pd["cokey"].isin(cokey_series) | (
        pd.notnull(muhorzdata_pd["hzdept_r"]) & pd.notnull(muhorzdata_pd["hzdepb_r"])
    )
    muhorzdata_pd = muhorzdata_pd[filter_condition].drop_duplicates().reset_index(drop=True)

    # Add distance column from mucompdata_pd using cokey link
    muhorzdata_pd = pd.merge(
        muhorzdata_pd,
        mucompdata_pd[["cokey", "distance", "distance_score"]], 
        on="cokey",
        how="left",
    )

    # Check for duplicate component instances
    hz_drop = drop_cokey_horz(muhorzdata_pd)
    if hz_drop is not None:
        muhorzdata_pd = muhorzdata_pd[~muhorzdata_pd.cokey.isin(hz_drop)]

    muhorzdata_pd.reset_index(drop=True, inplace=True)

    # Extract unique cokeys and subset mucompdata_pd
    comp_key = muhorzdata_pd["cokey"].unique().tolist()
    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(comp_key)]

    # Sort mucompdata_pd based on 'cond_prob' and 'distance'
    mucompdata_pd.sort_values(["cond_prob", "distance", "compname"], ascending=[False, True, True], inplace=True)
    mucompdata_pd.reset_index(drop=True, inplace=True)

    # Duplicate the 'compname' column for grouping purposes
    mucompdata_pd["compname_grp"] = mucompdata_pd["compname"]

    # Extract unique cokeys and create a ranking dictionary
    comp_key = mucompdata_pd["cokey"].unique().tolist()
    cokey_index = {key: index for index, key in enumerate(comp_key)}

    # Apply the ranking to muhorzdata_pd for sorting
    muhorzdata_pd["Comp_Rank"] = muhorzdata_pd["cokey"].map(cokey_index)

    # Sort muhorzdata_pd by 'Comp_Rank' and 'hzdept_r', and clean up
    muhorzdata_pd.sort_values(["Comp_Rank", "hzdept_r"], ascending=[True, True], inplace=True)
    muhorzdata_pd.drop("Comp_Rank", axis=1, inplace=True)
    muhorzdata_pd.reset_index(drop=True, inplace=True)

    mucompdata_pd = mucompdata_pd.drop_duplicates().reset_index(drop=True)

    # Update component names in mucompdata_pd to handle duplicates
    component_names = mucompdata_pd["compname"].tolist()
    name_counts = collections.Counter(component_names)

    # Track which indices have been processed for each name
    processed_indices = {}
    
    for name, count in sorted(name_counts.items()):  # Sort for deterministic order
        if count > 1:  # If a component name is duplicated
            # Find all indices for this name
            indices = [i for i, comp_name in enumerate(component_names) if comp_name == name]
            # Sort indices for deterministic order
            indices.sort()
            
            # Add suffixes to all occurrences except the first
            for i, idx in enumerate(indices):
                if i > 0:  # Skip the first occurrence (keep original name)
                    component_names[idx] = name + str(i + 1)  # Append suffix starting from 2

    mucompdata_pd["compname"] = component_names
    muhorzdata_pd.rename(columns={"compname": "compname_grp"}, inplace=True)
    # Merge the modified component names from mucompdata_pd to muhorzdata_pd
    muhorzdata_pd = muhorzdata_pd.merge(
        mucompdata_pd[["cokey", "compname"]], on="cokey", how="left"
    )

    # Remove bedrock by filtering out 'R|r' in hzname
    muhorzdata_pd = muhorzdata_pd[~muhorzdata_pd["hzname"].str.contains("R", case=False, na=False)]

    # Group data by cokey (component key)
    muhorzdata_group_cokey = [group for _, group in muhorzdata_pd.groupby("cokey", sort=True)]

    getProfile_cokey = []
    comp_max_depths = []
    clay_texture = []
    snd_lyrs = []
    cly_lyrs = []
    txt_lyrs = []
    hzt_lyrs = []
    hzb_lyrs = []
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

        mucompdata_pd_group = mucompdata_pd[mucompdata_pd["cokey"].isin(group_sorted["cokey"])]
        if (
            group_sorted["sandtotal_r"].isnull().values.all()
            or group_sorted["claytotal_r"].isnull().values.all()
        ) and (mucompdata_pd_group["compkind"].isin(OSD_compkind).any()):
            OSD_text_int.append("Yes")
        else:
            OSD_text_int.append("No")

        if (group_sorted["total_frag_volume"].isnull().values.all()) and (
            mucompdata_pd_group["compkind"].isin(OSD_compkind).any()
        ):
            OSD_rfv_int.append("Yes")
        else:
            OSD_rfv_int.append("No")

        # extract horizon data
        hz_dept = group_sorted["hzdept_r"]
        hz_depb = group_sorted["hzdepb_r"]
        snd_d = group_sorted["sandtotal_r"]
        cly_d = group_sorted["claytotal_r"]
        txt_d = group_sorted["texture"]
        rf_d = group_sorted["total_frag_volume"]
        cec_d = group_sorted["CEC"]
        ec_d = group_sorted["EC"]
        ph_d = group_sorted["pH"]

        hz_dept = hz_dept.fillna("")
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
        hzt_lyrs.append(dict(zip(hz_dept.index, hz_dept)))
        hzb_lyrs.append(dict(zip(hz_depb.index, hz_depb)))

        cokey_group = group_sorted["cokey"].iloc[0]
        compname_group = group_sorted["compname"].iloc[0]
        comp_max_bottom = max_comp_depth(group_sorted)
        comp_max_depths_temp = pd.DataFrame(
            {
                "cokey": [cokey_group],
                "compname": [compname_group],
                "comp_max_bottom": [int(comp_max_bottom)],
            }
        )
        comp_max_depths.append(comp_max_depths_temp)

        # Handle texture information
        comp_texture_list = [x for x in group_sorted.texture.str.lower().tolist() if x is not None]
        clay_indicator = "Yes" if any("clay" in s for s in comp_texture_list) else "No"
        clay_texture_temp = pd.DataFrame({"compname": [compname_group], "clay": [clay_indicator]})
        clay_texture.append(clay_texture_temp)

        # extract information to be combined later with site soil measurements
        sand_pct_intpl = getProfile(group_sorted, "sandtotal_r")
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

        getProfile_cokey.append(getProfile_cokey_temp2)

    comp_max_depths = pd.concat(comp_max_depths, axis=0)
    clay_texture = pd.concat(clay_texture, axis=0)

    # Filter main dataframes based on cokeys in comp_max_depths
    valid_cokeys = comp_max_depths["cokey"]

    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(valid_cokeys)]
    muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(valid_cokeys)]

    # Add OSD infilling indicators to mucompdata_pd
    mucompdata_pd["OSD_text_int"] = OSD_text_int
    mucompdata_pd["OSD_rfv_int"] = OSD_rfv_int

    # Merge component bottom depth and clay texture information into mucompdata_pd
    mucompdata_pd = mucompdata_pd.merge(
        comp_max_depths[["compname", "comp_max_bottom"]], on="compname", how="left"
    )
    mucompdata_pd = mucompdata_pd.merge(clay_texture, on="compname", how="left")

    # Create a mapping from cokey to index
    comp_key = mucompdata_pd["cokey"].unique().tolist()
    cokey_Index = {key: index for index, key in enumerate(comp_key)}

    # ----------------------------------------------------------------------------
    # This extracts OSD color, texture, and CF data

    if data_source == "STATSGO":
        # If the condition is met, we perform the series of operations, otherwise,
        # we set OSDhorzdata_pd to None
        if mucompdata_pd["compkind"].isin(OSD_compkind).any():
            try:
                # Generate series names
                seriesDict = get_soil_series_data(mucompdata_pd, OSD_compkind)
                if seriesDict is None:
                    logging.error("STATSGO: series data is missing")
                    raise ValueError("STATSGO: series data is missing")

                if seriesDict.get("hz") is not None:
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
                        .str.replace(r"(fine|medium|coarse) ", "", regex=True)
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

                    munsell_RGB_df = pd.DataFrame(
                        munsell_RGB, columns=["srgb_r", "srgb_g", "srgb_b"]
                    )
                    OSDhorzdata_pd = pd.concat([OSDhorzdata_pd, munsell_RGB_df], axis=1)

                    # Merge with another dataframe
                    mucompdata_pd_merge = mucompdata_pd[["mukey", "cokey", "compname", "compkind"]]
                    mucompdata_pd_merge["series"] = mucompdata_pd_merge["compname"].str.replace(
                        r"\d+", "", regex=True
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
                else:
                    print("Data for 'hz' is not available.")
                    OSDhorzdata_pd = None
            except Exception as err:
                logging.error(f"Error occurred with STATSGO: {str(err)}")
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
            mucompdata_pd_merge = mucompdata_pd[["mukey", "cokey", "compname", "compkind"]].copy()
            mucompdata_pd_merge["series"] = mucompdata_pd_merge["compname"].str.replace(
                r"\d+", "", regex=True
            )

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
                    "srgb_r",
                    "srgb_g",
                    "srgb_b",
                    "cielab_l",
                    "cielab_a",
                    "cielab_b",
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
                .str.replace(r"(fine|medium|coarse) ", "", regex=True)
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
        mucompdata_pd_merge["series"] = mucompdata_pd_merge["compname"].str.replace(
            r"\d+", "", regex=True
        )
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
            OSDhorzdata_group_cokey = [group for _, group in OSDhorzdata_pd.groupby("cokey")]

            # Initialize empty lists
            lab_lyrs = []
            munsell_lyrs = []
            lab_intpl_lyrs = []

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
                        new_layer = create_new_layer_osd(
                            group_sorted.iloc[0], 0, group_sorted["top"].iloc[0]
                        )
                        group_sorted = pd.concat(
                            [pd.DataFrame([new_layer]), group_sorted], ignore_index=True
                        )

                    # Check for missing subsurface horizons
                    for j in range(len(group_sorted) - 1):
                        if group_sorted["top"].iloc[j + 1] > group_sorted["bottom"].iloc[j]:
                            new_layer = create_new_layer_osd(
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

                    # Initialize flags to indicate if OSD depth adjustment is needed
                    OSD_depth_add = False
                    OSD_depth_remove = False

                    # Extract OSD Color Data
                    lab_intpl = getProfileLAB(group_sorted, color_ref)
                    lab_intpl.columns = ["l", "a", "b"]
                    OSD_max_bottom = group_sorted["bottom"].iloc[-1]

                    # subset c_bottom_depth by cokey
                    comp_max_depths_group = comp_max_depths[
                        comp_max_depths["cokey"].isin(group_sorted["cokey"])
                    ]
                    muhorzdata_pd_group = muhorzdata_pd[
                        muhorzdata_pd["cokey"].isin(group_sorted["cokey"])
                    ]

                    # Check if OSD depth adjustment is needed
                    if OSD_max_bottom < comp_max_depths_group["comp_max_bottom"].iloc[0]:
                        OSD_depth_add = True
                        depth_difference = (
                            comp_max_depths_group["comp_max_bottom"].iloc[0] - OSD_max_bottom
                        )

                        lab_values = [
                            lab_intpl.loc[OSD_max_bottom - 1].values.tolist()
                        ] * depth_difference

                        pd_add = pd.DataFrame(lab_values, columns=["l", "a", "b"])

                        # Adjust LAB values for the OSD depth
                        lab_intpl = pd.concat(
                            [lab_intpl.loc[: OSD_max_bottom - 1], pd_add], axis=0
                        ).reset_index(drop=True)
                        OSD_max_bottom = comp_max_depths_group["comp_max_bottom"].iloc[0]

                    elif 0 < comp_max_depths_group["comp_max_bottom"].iloc[0] < OSD_max_bottom:
                        OSD_depth_remove = True

                        # Adjust LAB values for the component depth
                        lab_intpl = lab_intpl.loc[
                            : comp_max_depths_group["comp_max_bottom"].iloc[0]
                        ].reset_index(drop=True)
                        OSD_max_bottom = comp_max_depths_group["comp_max_bottom"].iloc[0]

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
                            (
                                lab2munsell(color_ref, LAB_ref, lab)
                                if lab[0] and lab[1] and lab[2]
                                else ""
                            )
                            for lab in lab_parse
                        ]
                        munsell_lyrs.append(dict(zip(l_d.index, munsell_values)))

                    # Extract OSD Texture and Rock Fragment Data
                    if OSD_text_int[index] == "Yes" or OSD_rfv_int[index] == "Yes":
                        group_sorted[["hzdept_r", "hzdepb_r", "texture"]] = group_sorted[
                            ["top", "bottom", "texture_class"]
                        ]
                        OSD_max_bottom_int = max_comp_depth(group_sorted)
                        OSD_clay_intpl = getProfile(group_sorted, "claytotal_r")
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

                        # Update data based on depth conditions
                        sand_values = OSD_sand_intpl.iloc[OSD_max_bottom_int - 1].tolist()
                        OSD_sand_intpl = update_intpl_data(
                            OSD_sand_intpl,
                            ["c_sandpct_intpl", "c_sandpct_intpl_grp"],
                            sand_values,
                            OSD_max_bottom,
                            OSD_depth_add,
                            OSD_depth_remove,
                            OSD_max_bottom_int,
                        )

                        clay_values = OSD_clay_intpl.iloc[OSD_max_bottom_int - 1].tolist()
                        OSD_clay_intpl = update_intpl_data(
                            OSD_clay_intpl,
                            ["c_claypct_intpl", "c_claypct_intpl_grp"],
                            clay_values,
                            OSD_max_bottom,
                            OSD_depth_add,
                            OSD_depth_remove,
                            OSD_max_bottom_int,
                        )

                        rfv_values = OSD_rfv_intpl.iloc[OSD_max_bottom_int - 1].tolist()
                        OSD_rfv_intpl = update_intpl_data(
                            OSD_rfv_intpl,
                            ["c_cfpct_intpl", "c_cfpct_intpl_grp"],
                            rfv_values,
                            OSD_max_bottom,
                            OSD_depth_add,
                            OSD_depth_remove,
                            OSD_max_bottom_int,
                        )

                        # If OSD bottom depth is greater than component depth
                        # and component depth is <200cm
                        if OSD_depth_remove:
                            # Remove data based on comp_max_depths
                            OSD_sand_intpl = OSD_sand_intpl.loc[: comp_max_depths.iloc[index, 2]]
                            OSD_clay_intpl = OSD_clay_intpl.loc[: comp_max_depths.iloc[index, 2]]
                            OSD_rfv_intpl = OSD_rfv_intpl.loc[: comp_max_depths.iloc[index, 2]]

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
                        snd_d_osd = aggregate_data(
                            data=OSD_sand_intpl.iloc[:, 0],
                            bottom_depths=muhorzdata_pd_group["hzdepb_r"].tolist(),
                        )

                        # Aggregate clay data
                        cly_d_osd = aggregate_data(
                            data=OSD_clay_intpl.iloc[:, 1],
                            bottom_depths=muhorzdata_pd_group["hzdepb_r"].tolist(),
                        )

                        # Calculate texture data based on sand and clay data
                        txt_d_osd = [
                            getTexture(row=None, sand=s, silt=(100 - (s + c)), clay=c)
                            for s, c in zip(snd_d_osd, cly_d_osd)
                        ]
                        txt_d_osd = pd.Series(txt_d_osd, index=snd_d_osd.index)

                        # Aggregate rock fragment data
                        rf_d_osd = aggregate_data(
                            data=OSD_rfv_intpl.c_cfpct_intpl,
                            bottom_depths=muhorzdata_pd_group["hzdepb_r"].tolist(),
                        )

                        # Fill NaN values
                        snd_d_osd.fillna(np.nan, inplace=True)
                        cly_d_osd.fillna(np.nan, inplace=True)
                        txt_d_osd.fillna(np.nan, inplace=True)
                        rf_d_osd.fillna(np.nan, inplace=True)

                        # Store aggregated data in dictionaries based on conditions
                        if OSD_text_int[index] == "Yes":
                            snd_lyrs[index] = snd_d_osd.to_dict()
                            cly_lyrs[index] = cly_d_osd.to_dict()
                            txt_lyrs[index] = txt_d_osd.to_dict()

                        if OSD_rfv_int[index] == "Yes":
                            rf_lyrs[index] = rf_d_osd.to_dict()

                        # Update cec, ph, and ec layers if they contain only a single
                        # empty string
                        for lyr in [cec_lyrs, ph_lyrs, ec_lyrs]:
                            if len(lyr[index]) == 1 and lyr[index][0] == "":
                                empty_values = [""] * len(hzb_lyrs[index])
                                lyr[index] = dict(zip(hzb_lyrs[index], empty_values))

                else:
                    OSDhorzdata_group_cokey[index] = group_sorted

                    # Create an empty dataframe with NaNs for lab_intpl
                    lab_intpl = pd.DataFrame(
                        np.nan,
                        index=np.arange(comp_max_depths.iloc[index, 2]),
                        columns=["l", "a", "b"],
                    )
                    lab_intpl_lyrs.append(lab_intpl)

                    # Create dummy data for lab_lyrs
                    lab_dummy = [["", "", ""] for _ in range(len(hzb_lyrs[index]))]
                    lab_lyrs.append(dict(zip(hzb_lyrs[index].keys(), lab_dummy)))

                    # Create dummy data for munsell_lyrs
                    munsell_dummy = [""] * len(hzb_lyrs[index])
                    munsell_lyrs.append(dict(zip(hzb_lyrs[index].keys(), munsell_dummy)))

            # Series URL Generation
            # Create a mapping of cokey to URLs for safe lookup
            cokey_to_urls = {}

            # Group data by 'cokey' - use sort=True for deterministic ordering
            OSDhorzdata_group_cokey = [g for _, g in OSDhorzdata_pd.groupby("cokey", sort=True)]

            for index, group in enumerate(OSDhorzdata_group_cokey):
                cokey = group["cokey"].iloc[0]  # Get the cokey for this group
                
                # Check if compkind is not in OSD_compkind or if series contains any null values
                if (
                    mucompdata_pd[mucompdata_pd["cokey"] == cokey]["compkind"].iloc[0] not in OSD_compkind
                    or group["series"].isnull().any()
                ):
                    cokey_to_urls[cokey] = {"sde": "", "see": ""}
                else:
                    # Extract compname, convert to lowercase, remove trailing numbers, and replace spaces with underscores
                    comp = group["compname"].iloc[0].lower()
                    comp = re.sub(r"\d+$", "", comp)
                    comp = comp.replace(" ", "_")

                    # Create URLs
                    cokey_to_urls[cokey] = {
                        "sde": f"https://casoilresource.lawr.ucdavis.edu/sde/?series={comp}",
                        "see": f"https://casoilresource.lawr.ucdavis.edu/see/#{comp}"
                    }

        else:
            # Initialize lists to store data layers and URLs
            lab_lyrs = []
            lab_intpl_lyrs = []
            munsell_lyrs = []
            cokey_to_urls = {}

            # Iterate over each entry in mucompdata_pd
            for i in range(len(mucompdata_pd)):
                # Initialize a DataFrame filled with NaNs
                lab_intpl = pd.DataFrame(
                    np.nan,
                    index=np.arange(comp_max_depths.iloc[i, 2]),
                    columns=["l", "a", "b"],
                )
                lab_intpl_lyrs.append(lab_intpl)

                # Create dummy data for lab and munsell layers
                keys = list(hzb_lyrs[i].keys())
                lab_dummy = [{"", "", ""} for _ in range(len(keys))]
                munsell_dummy = [""] * len(keys)

                # Append dummy data to lists
                lab_lyrs.append(dict(zip(keys, lab_dummy)))
                munsell_lyrs.append(dict(zip(keys, munsell_dummy)))

                # Create empty URLs for each component
                cokey = mucompdata_pd.iloc[i]["cokey"]
                cokey_to_urls[cokey] = {"sde": "", "see": ""}

    else:
        # Initialize lists to store data layers and URLs
        lab_lyrs = []
        lab_intpl_lyrs = []
        munsell_lyrs = []
        cokey_to_urls = {}

        # Iterate over each entry in mucompdata_pd
        for i in range(len(mucompdata_pd)):
            # Initialize a DataFrame filled with NaNs
            lab_intpl = pd.DataFrame(
                np.nan,
                index=np.arange(comp_max_depths.iloc[i, 2]),
                columns=["l", "a", "b"],
            )
            lab_intpl_lyrs.append(lab_intpl)

            # Create dummy data for lab and munsell layers
            keys = list(hzb_lyrs[i].keys())
            lab_dummy = [{"", "", ""} for _ in range(len(keys))]
            munsell_dummy = [""] * len(keys)

            # Append dummy data to lists
            lab_lyrs.append(dict(zip(keys, lab_dummy)))
            munsell_lyrs.append(dict(zip(keys, munsell_dummy)))

            # Create empty URLs for each component
            cokey = mucompdata_pd.iloc[i]["cokey"]
            cokey_to_urls[cokey] = {"sde": "", "see": ""}

    # Subset datasets to exclude pedons without any depth information
    cokeys_with_depth = mucompdata_pd[mucompdata_pd["comp_max_bottom"] > 0].cokey.unique()

    # If there are cokeys with depth
    if len(cokeys_with_depth) > 0:
        # Subset based on cokeys with depth data
        mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(cokeys_with_depth)].reset_index(
            drop=True
        )
        muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(cokeys_with_depth)].reset_index(
            drop=True
        )
        comp_max_depths = comp_max_depths[
            comp_max_depths["cokey"].isin(cokeys_with_depth)
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
            hzt_lyrs,
            hzb_lyrs,
            lab_lyrs,
            munsell_lyrs,
        ]

        for i, lst in enumerate(layer_lists):
            safe_indices = [index for index in indices_with_depth if index < len(lst)]
            layer_lists[i] = [lst[index] for index in safe_indices]

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
            hzt_lyrs,
            hzb_lyrs,
            lab_lyrs,
            munsell_lyrs,
        ) = layer_lists

    # Filter muhorzdata_pd to remove horizons with missing data == likely bedrock
    # --Remove rows where both 'sandtotal_r' and 'claytotal_r' are NaN
    muhorzdata_pd = muhorzdata_pd[
        ~(muhorzdata_pd["sandtotal_r"].isna() & muhorzdata_pd["claytotal_r"].isna())
    ]

    # Run soil simulations: functional similarity calculation and soil information value
    aws_PIW90, var_imp = soil_sim(muhorzdata_pd)

    # Create a new column 'soilID_rank' which will be True for the first row in each group sorted
    # by 'distance' and False for other rows
    mucompdata_pd = mucompdata_pd.sort_values(["compname_grp", "distance", "compname"])
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
            ESD[["cokey", "ecoclassid", "ecoclassname", "edit_url"]] = ESD[
                ["cokey", "ecoclassid", "ecoclassname", "edit_url"]
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
            "SELECT cokey,  ecoclassid, ecoclassname FROM coecoclass WHERE cokey IN ("
            + ",".join(map(str, comp_key))
            + ") ORDER BY cokey"
        )
        ESDcompdata_out = sda_return(propQry=ESDcompdataQry)

        if ESDcompdata_out is None:
            ESDcompdata_pd = None
        else:
            ESDcompdata_sda = pd.DataFrame(
                ESDcompdata_out["Table"].iloc[0][1:], columns=ESDcompdata_out["Table"].iloc[0][0]
            )

            ESDcompdata_sda[["cokey", "ecoclassid", "ecoclassname"]] = ESDcompdata_sda[
                ["cokey", "ecoclassid", "ecoclassname"]
            ].astype(str)

            # Create the "edit_url" column
            ESDcompdata_sda["edit_url"] = (
                "https://edit.jornada.nmsu.edu/catalogs/esd/"
                + ESDcompdata_sda["ecoclassid"].str[1:5]
                + "/"
                + ESDcompdata_sda["ecoclassid"]
            )

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

    # Initialize the list for storing ESD components data
    esd_comp_list = []

    # Main logic for handling ESD data based on its presence
    if ESDcompdata_pd is not None:
        # Process DataFrame: cleaning and updating
        ESDcompdata_pd.replace("NULL", np.nan, inplace=True)
        ESDcompdata_pd.drop_duplicates(subset=["cokey"], keep="first", inplace=True)
        ESDcompdata_pd = ESDcompdata_pd[ESDcompdata_pd["cokey"].isin(comp_key)]
        ESDcompdata_pd["Comp_Rank"] = ESDcompdata_pd["cokey"].map(cokey_Index)
        ESDcompdata_pd.sort_values("Comp_Rank", ascending=True, inplace=True)
        ESDcompdata_pd.drop(columns="Comp_Rank", inplace=True)
        ESDcompdata_pd["ecoclassname"] = ESDcompdata_pd["ecoclassname"].str.title()

        # Further processing and checks for missing ESD data
        ESDcompdata_pd = update_esd_data(ESDcompdata_pd)

        # Aggregate the ESD components for output
        for _, group in ESDcompdata_pd.groupby("cokey", sort=True):
            esd_data = {
                "ESD": {
                    "ecoclassid": group["ecoclassid"].tolist(),
                    "ecoclassname": group["ecoclassname"].tolist(),
                    "edit_url": group["edit_url"].tolist(),
                }
            }
            esd_comp_list.append(esd_data)
    else:
        # Fill the list with empty data if ESDcompdata_pd is not available
        esd_comp_list = [
            {"ESD": {"ecoclassid": "", "ecoclassname": "", "edit_url": ""}}
            for _ in range(len(mucompdata_pd))
        ]

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

            # Assign missing ESD for components that have other instances with an assigned ESD
            if ESDcompdata_pd is not None:
                if (
                    ESDcompdata_pd.ecoclassid.isnull().any()
                    or ESDcompdata_pd.ecoclassname.isnull().any()
                ):
                    ESDcompdata_pd["compname_grp"] = ESDcompdata_pd.compname.str.replace(
                        r"[0-9]+", "", regex=True
                    )
                    ESDcompdata_pd_comp_grps = [
                        g for _, g in ESDcompdata_pd.groupby(["compname_grp"], sort=True)
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
                            url = comp_grps_temp.edit_url.unique().tolist()
                            url = [x for x in url if x != ""]
                            if not url:
                                comp_grps_temp["edit_url"] = pd.Series(
                                    np.tile("", len(comp_grps_temp))
                                ).values
                            else:
                                comp_grps_temp["edit_url"] = pd.Series(
                                    np.tile(url, len(comp_grps_temp))
                                ).values
                            ecoList_out.append(comp_grps_temp)
                        else:
                            ecoList_out.append(comp_grps_temp)
                    ESDcompdata_pd = pd.concat(ecoList_out)

                ESDcompdata_group_cokey = [
                    g for _, g in ESDcompdata_pd.groupby(["cokey"], sort=True)
                ]
                for i in range(len(ESDcompdata_group_cokey)):
                    if ESDcompdata_group_cokey[i]["ecoclassname"].isnull().values.any():
                        esd_comp_list.append(
                            {"ESD": {"ecoclassid": "", "ecoclassname": "", "edit_url": ""}}
                        )
                    else:
                        esd_comp_list.append(
                            {
                                "ESD": {
                                    "ecoclassid": ESDcompdata_group_cokey[i]["ecoclassid"].tolist(),
                                    "ecoclassname": ESDcompdata_group_cokey[i][
                                        "ecoclassname"
                                    ].tolist(),
                                    "edit_url": ESDcompdata_group_cokey[i]["edit_url"].tolist(),
                                }
                            }
                        )
            else:
                for i in range(len(mucompdata_pd)):
                    esd_comp_list.append(
                        {"ESD": {"ecoclassid": "", "ecoclassname": "", "edit_url": ""}}
                    )
        else:
            for i in range(len(mucompdata_pd)):
                esd_comp_list.append(
                    {"ESD": {"ecoclassid": "", "ecoclassname": "", "edit_url": ""}}
                )

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

    # Create a new column that maps each 'cokey' value to its corresponding index value
    mucompdata_pd["cokey_order"] = mucompdata_pd["cokey"].map(cokey_Index)

    # Sort the DataFrame by the 'cokey_order' column
    mucompdata_cond_prob = mucompdata_pd.sort_values(by="cokey_order").reset_index(drop=True)
    mucompdata_cond_prob = mucompdata_cond_prob.drop(columns=["cokey_order"])

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

    # Sort mucompdata_cond_prob by soilID_rank, cond_prob, and compname for deterministic tie-breaking
    mucompdata_cond_prob = mucompdata_cond_prob.sort_values(
        ["soilID_rank", "cond_prob", "compname"], ascending=[False, False, True]
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
            mucompdata_cond_prob["cond_prob"].round(3),
            mucompdata_cond_prob["Rank_Loc"],
        )
    ]

    # Replace NaN values with an empty string
    mucompdata_cond_prob = mucompdata_cond_prob.fillna("")

    # Generate the Site list using cokey-based URL lookup
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
                "sdeURL": cokey_to_urls.get(row["cokey"], {"sde": ""})["sde"],
                "seeURL": cokey_to_urls.get(row["cokey"], {"see": ""})["see"],
            },
            "siteDescription": row["brief_narrative"],
        }
        for idx, row in mucompdata_cond_prob.iterrows()
    ]

    # Reordering lists using list comprehension and mucomp_index
    lists_to_reorder = [
        esd_comp_list,
        hzt_lyrs,
        hzb_lyrs,
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
        hzt_lyrs,
        hzb_lyrs,
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
                    "top_depth",
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
            hzt_lyrs,
            hzb_lyrs,
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

    soil_list_json = {
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
        "AWS_PIW90": aws_PIW90,
        "Soil Data Value": var_imp,
        "soilList": output_SoilList,
    }

    return SoilListOutputData(
        soil_list_json=soil_list_json,
        rank_data_csv=soilIDRank_output_pd.to_csv(index=None, header=True),
        map_unit_component_data_csv=mucompdata_cond_prob.to_csv(index=None, header=True),
    )


##############################################################################################
#                                   rank_soils                                         #
##############################################################################################
def rank_soils(
    lon,
    lat,
    list_output_data: SoilListOutputData,
    soilHorizon,
    topDepth,
    bottomDepth,
    rfvDepth,
    lab_Color,
    pSlope,
    pElev,
    bedrock,
    cracks,
):
    """
    TODO: Future testing to see if deltaE2000 values should be incorporated
    into site data use 'getColor_deltaE2000_OSD_pedon' and helper functions
    located in utils.py
    """
    # Check if list_output_data is a string (error message) instead of expected object
    if isinstance(list_output_data, str):
        return {"error": f"Cannot rank soils: {list_output_data}"}
    
    # ---------------------------------------------------------------------------------------
    # ------ Load in user data --------#
    # Initialize the DataFrame from the input data
    soil_df = pd.DataFrame(
        {
            "soilHorizon": soilHorizon,
            "top": topDepth,
            "bottom": bottomDepth,
            "rfvDepth": rfvDepth,
            "lab_Color": lab_Color,
        }
    )

    # Drop rows where all values are NaN
    soil_df.dropna(how="all", inplace=True)

    # Replace NaNs with None for consistency
    # soil_df.fillna(value=None, inplace=True)

    # Adjust the bottom depth based on bedrock depth
    if bedrock is not None:
        if bedrock is not soil_df.empty and soil_df["bottom"].iloc[-1] > bedrock:
            # Remove rows where top depth is already below bedrock
            soil_df = soil_df[soil_df["top"] < bedrock].copy()
            # If any remaining row has bottom depth exceeding bedrock, truncate it
            soil_df.loc[soil_df["bottom"] > bedrock, "bottom"] = bedrock

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
        pedon_slice_index = [x for x in pedon_slice_index if x < 200]

        if bedrock is not None:
            pedon_slice_index.extend(range(bedrock, 200))

        # Convert soil properties to lists
        soilHorizon = soil_df.soilHorizon.tolist()
        rfvDepth = soil_df.rfvDepth.tolist()
        horizonDepthB = [int(x) for x in soil_df.bottom.tolist()]
        horizonDepthT = [int(x) for x in soil_df.top.tolist()]
        lab_series = soil_df.lab_Color

        # Determine the maximum depth for user specified horizons
        if not horizonDepthB:
            max_user_depth = 0
        else:
            max_user_depth = max(horizonDepthB)
        if bedrock is not None:
            max_user_depth = min(bedrock, max_user_depth)

        # Generate user specified percent clay, sand, and rfv distributions
        spt = [getSand(sh) for sh in soilHorizon]
        cpt = [getClay(sh) for sh in soilHorizon]
        p_cfg = [getCF_fromClass(rf) for rf in rfvDepth]

        # Initialize full-length property arrays with NaNs
        sand_array = [np.nan] * max_user_depth
        clay_array = [np.nan] * max_user_depth
        cfg_array = [np.nan] * max_user_depth

        for i in range(len(soilHorizon)):
            t = horizonDepthT[i]
            b = horizonDepthB[i]
            if t >= max_user_depth:
                continue
            b = min(b, max_user_depth)
            for d in range(t, b):
                sand_array[d] = spt[i]
                clay_array[d] = cpt[i]
                cfg_array[d] = p_cfg[i]

        p_sandpct_intpl = pd.DataFrame(sand_array)
        p_claypct_intpl = pd.DataFrame(clay_array)
        p_cfg_intpl = pd.DataFrame(cfg_array)

        # Length of interpolated texture and RF depth
        p_bottom_depth = pd.DataFrame([-999, "sample_pedon", max_user_depth]).T
        p_bottom_depth.columns = ["cokey", "compname", "bottom_depth"]

        # Pedon color data
        lab_array = [[np.nan, np.nan, np.nan] for _ in range(max_user_depth)]

        # Force correct structure
        lab_cleaned = [
            row if (isinstance(row, (list, tuple)) and len(row) == 3) else [np.nan, np.nan, np.nan]
            for row in lab_series
        ]

        # now unpack into three columns
        lab_Color = pd.DataFrame(lab_cleaned, columns=["L", "A", "B"])

        # Interpolate colors across depth
        for i in range(len(lab_Color)):
            t = horizonDepthT[i]
            b = horizonDepthB[i]
            if t >= max_user_depth:
                continue
            b = min(b, max_user_depth)
            color_val = (
                lab_Color.iloc[i].tolist()
                if not pd.isnull(lab_Color.iloc[i]).all()
                else [np.nan, np.nan, np.nan]
            )
            for d in range(t, b):
                lab_array[d] = color_val

        p_lab_intpl = pd.DataFrame(lab_array, columns=["L", "A", "B"]).reset_index(drop=True)

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
        if bedrock is not None:
            p_bottom_depth = pd.DataFrame([-999, "sample_pedon", bedrock]).T
        else:
            p_bottom_depth = pd.DataFrame([-999, "sample_pedon", 0]).T
        p_bottom_depth.columns = ["cokey", "compname", "bottom_depth"]

    # -------------------------------------------------------------------------------------------
    # Load in component data from soilIDList
    soilIDRank_output_pd = pd.read_csv(io.StringIO(list_output_data.rank_data_csv))
    mucompdata_pd = pd.read_csv(io.StringIO(list_output_data.map_unit_component_data_csv))

    # Modify mucompdata_pd DataFrame
    # mucompdata_pd = process_site_data(mucompdata_pd)

    # Create soil depth DataFrame and subset component depths based on max user
    # depth if no bedrock specified

    comp_max_depths = mucompdata_pd[["cokey", "compname", "comp_max_bottom"]]
    comp_max_depths.columns = ["cokey", "compname", "bottom_depth"]
    slices_of_soil = pd.concat([p_bottom_depth, comp_max_depths], axis=0).reset_index(drop=True)
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
    if p_hz_data is None or p_hz_data.empty or p_bottom_depth.bottom_depth.le(0).any():
        soilIDRank_output_pd = None
    else:
        # Subset component soil properties to match user measured properties
        soilIDRank_output_pd = soilIDRank_output_pd[p_hz_data_names]

        # Subset soil_matrix to user measured slices
        soil_matrix = soil_matrix.loc[pedon_slice_index]

    # Horizon Data Similarity
    if soilIDRank_output_pd is not None:
        groups = [group for _, group in soilIDRank_output_pd.groupby(["compname"], sort=True)]

        Comp_Rank_Status = []
        Comp_Missing_Status = []
        Comp_name = []

        for group in groups:
            data_subgroup = group.drop(columns="compname")
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

            Comp_name.append(sorted(group["compname"].unique())[0])

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

        global_prop_bounds = {
            "sandpct_intpl": (10.0, 92.0),
            "claypct_intpl": (5.0, 70.0),
            "rfv_intpl": (0.0, 80.0),
            "l": (10.0, 95.0),
            "a": (-10.0, 35.0),
            "b": (-10.0, 60.0),
        }

        # numeric range (upper – lower):
        global_prop_ranges = {
            prop: upper - lower for prop, (lower, upper) in global_prop_bounds.items()
        }

        # Calculate similarity for each depth slice
        dis_mat_list = []
        for i in (
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
            if bedrock is None or (bedrock is not None and i < bedrock):
                sample_pedon_slice_vars = (
                    sliceT.dropna(axis="columns").drop(["compname"], axis=1).columns.tolist()
                )
                if len(sample_pedon_slice_vars) < 2:
                    sample_pedon_slice_vars = (
                        sliceT[sliceT["compname"] == "sample_pedon"]
                        .dropna(axis="columns")
                        .drop(["compname"], axis=1)
                        .columns.tolist()
                    )
            sliceT = sliceT[sample_pedon_slice_vars]

            theoretical_prop_ranges = [global_prop_ranges[c] for c in sample_pedon_slice_vars]
            D = gower_distances(
                sliceT, theoretical_ranges=theoretical_prop_ranges
            )  # Equal weighting given to all soil variables

            dis_mat_list.append(D)

        # Determine if any components have all NaNs at every slice
        dis_mat_nan_check = np.ma.MaskedArray(dis_mat_list, mask=np.isnan(dis_mat_list))
        D_check = np.ma.average(dis_mat_nan_check, axis=0)
        Rank_Filter["rank_status"] = [
            "Not Ranked" if np.ma.is_masked(x) else "Ranked" for x in D_check[0][1:]
        ]

        # Maximum dissimilarity
        dis_max = 1.0

        # Apply depth weight
        depth_weight = np.concatenate((np.repeat(0.2, 20), np.repeat(1.0, 180)), axis=0)
        depth_weight = depth_weight[pedon_slice_index]

        # Infill Nan data: soil vs non‑soil logic
        for i, dis_mat in enumerate(dis_mat_list):
            soil_slice = soil_matrix.iloc[i, :].values.astype(bool)
            pedon_idx = 0  # assuming sample_pedon is always row 0

            # 1) If pedon has data here, any NaN in pedon↔component → max dissimilarity
            if soil_slice[pedon_idx]:
                # components are those indices where soil_slice[j] is False
                nonsoil_j = np.where(~soil_slice)[0]
                for j in nonsoil_j:
                    if np.isnan(dis_mat[pedon_idx, j]):
                        dis_mat[pedon_idx, j] = dis_max
                        dis_mat[j, pedon_idx] = dis_max

            # 2) Every other NaN (component–component or missing–missing) → zero
            dis_mat[np.isnan(dis_mat)] = 0.0

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
    if pElev is None:
        pElev_dict = get_elev_data(lon, lat)
    try:
        pElev = float(pElev_dict["value"])
    except (KeyError, TypeError, ValueError):
        pElev = None  # or some default

    # 1) “Raw” guard on the three possible site inputs:
    provided = {
        "slope_r": pSlope,
        "elev_r": pElev,
        "bottom_depth": bedrock,  # this determines if bottom_depth is set
    }

    # 2) Figure out which of those are actually non-null
    features = [
        name
        for name, val in provided.items()
        if val is not None and not (isinstance(val, float) and np.isnan(val))
    ]

    # 3) If fewer than two features, skip entirely
    if len(features) < 2:
        D_site = None
    else:
        # 4) Build your one‐row pedon DataFrame (only slope/elev, depth comes from merge)
        pedon_dict = {"compname": "sample_pedon"}
        if "slope_r" in features:
            pedon_dict["slope_r"] = pSlope
        if "elev_r" in features:
            pedon_dict["elev_r"] = pElev
        pedon_df = pd.DataFrame([pedon_dict])

        # 5) Build your map‐unit library (only the columns you need)
        lib_cols = ["compname"] + [f for f in features if f in ("slope_r", "elev_r")]
        lib_df = mucompdata_pd[lib_cols].copy()

        # 6) Stack them together
        full_df = pd.concat([pedon_df, lib_df], ignore_index=True)

        # 7) If you need bottom_depth, merge it in for *all* rows
        if "bottom_depth" in features:
            full_df = full_df.merge(
                slices_of_soil[["compname", "bottom_depth"]], on="compname", how="left"
            )

        # 8) Build your weight vector
        DEFAULT_WEIGHTS = {"slope_r": 1.0, "elev_r": 0.5, "bottom_depth": 1.5}
        weights = np.array([DEFAULT_WEIGHTS[f] for f in features])

        # 9) Compute Gower distances on exactly those feature columns
        site_mat = full_df.set_index("compname")[features]
        D_raw = gower_distances(site_mat, feature_weight=weights)

        # 10 Replace any NaNs with the max distance, then (optionally) convert to similarity
        D_site = np.where(np.isnan(D_raw), np.nanmax(D_raw), D_raw)

        site_wt = 0.5
        D_site = (1 - D_site) * site_wt

    # Create the D_final dataframe based on the availability of D_horz and D_site data

    # When both D_horz and D_site are available (relative weights: 66% horz, 33% site)
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
    for _, group in D_final.groupby(["compname_grp"], sort=True):
        # Sort by score, and then by compname
        group = group.sort_values(by=["Score_Data", "compname"], ascending=[False, True])

        # The top component in each group gets a True rank, others get False
        group["soilID_rank_data"] = [True if i == 0 else False for i in range(len(group))]

        soilIDList_data.append(group)

    # Concatenate the sorted and ranked groups
    D_final = pd.concat(soilIDList_data).reset_index(drop=True)
    
    # Merge with the Rank_Filter data
    D_final = pd.merge(D_final, Rank_Filter, on="compname", how="left")

    # Sort dataframe to correctly assign Rank_Data
    D_final = D_final.sort_values(by=["soilID_rank_data", "Score_Data", "compname"], ascending=[False, False, True])

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
    'cond_prob', 'clay', 'taxorder', 'taxsubgrp', 'OSD_text_int',
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
                "cond_prob",
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
        # Calculate the combined score
        Score_Data_Loc = (D_final_loc["Score_Data"] + D_final_loc["cond_prob"]) / (
            D_final_loc["data_weight"] + location_weight
        )

    # Assign the final combined score to the dataframe
    D_final_loc["Score_Data_Loc"] = Score_Data_Loc

    # Rule-based code to identify vertisols or soils with vertic properties

    # Identify vertisols based on cracks, clay texture, and taxonomic presence of "ert"
    # Compute the condition for rows that meet the criteria
    if cracks is None or cracks is False:
        condition = pd.Series([False] * len(D_final_loc))
    else:
        condition = (
            (D_final_loc["clay"] == "Yes")
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

    for _, group in D_final_loc.groupby("compname_grp", sort=True):
        group = group.sort_values(["Score_Data_Loc", "compname"], ascending=[False, True]).reset_index(drop=True)
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

    # Sort dataframe based on soilID_rank_final, Score_Data_Loc, and compname for deterministic tie-breaking
    D_final_loc = D_final_loc.sort_values(
        ["soilID_rank_final", "Score_Data_Loc", "compname"], ascending=[False, False, True]
    ).reset_index(drop=True)

    # Replace NaN values in the specified columns with 0.0
    D_final_loc[
        [
            "Score_Data",
            "D_horz",
            "D_site",
            "Score_Data_Loc",
            "cond_prob",
        ]
    ] = D_final_loc[
        [
            "Score_Data",
            "D_horz",
            "D_site",
            "Score_Data_Loc",
            "cond_prob",
        ]
    ].fillna(0.0)

    # Construct the output format
    Rank = [
        {
            "name": row.compname.capitalize(),
            "component": row.compname_grp.capitalize(),
            "componentID": row.cokey,
            "score_data_loc": (
                "" if row.missing_status == "Location data only" else round(row.Score_Data_Loc, 3)
            ),
            "rank_data_loc": (
                "" if row.missing_status == "Location data only" else row.Rank_Data_Loc
            ),
            "score_data": (
                "" if row.missing_status == "Location data only" else round(row.Score_Data, 3)
            ),
            "rank_data": "" if row.missing_status == "Location data only" else row.Rank_Data,
            "score_loc": round(row.cond_prob, 3),
            "rank_loc": row.Rank_Loc,
            "componentData": row.missing_status,
        }
        for _, row in D_final_loc.iterrows()
    ]

    output_data = {
        "metadata": {
            "location": "us",
            "model": "v2",
        },
        "soilRank": Rank,
    }

    return output_data
