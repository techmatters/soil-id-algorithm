# Standard libraries
import collections
import io
import json
import logging
import re

# local libraries
import config

# Third-party libraries
import numpy as np
import pandas as pd
import soil_sim
from color import lab2munsell, munsell2rgb
from db import load_model_output, save_model_output, save_rank_output

# Flask
from pandas import json_normalize
from services import (
    get_elev_data,
    get_esd_data,
    get_soil_series_data,
    get_soilweb_data,
    sda_return,
)
from utils import (
    aggregate_data,
    compute_site_similarity,
    drop_cokey_horz,
    extract_mucompdata_STATSGO,
    extract_muhorzdata_STATSGO,
    fill_missing_comppct_r,
    getCF_fromClass,
    getClay,
    getOSDCF,
    getProfile,
    getProfileLAB,
    getSand,
    getTexture,
    gower_distances,
    max_comp_depth,
    process_distance_scores,
    process_horizon_data,
    process_site_data,
    update_esd_data,
)

# entry points
# getSoilLocationBasedGlobal
# getSoilLocationBasedUS
# rankPredictionUS
# rankPredictionGlobal
# getSoilGridsGlobal

# when a site is created, call getSoilLocationBasedUS/getSoilLocationBasedGlobal.
# when a site is created, call getSoilGridsGlobal
# after user has collected data, call rankPredictionUS/rankPredictionGlobal.

# set Pandas dataframe options
pd.set_option("future.no_silent_downcasting", True)


############################################################################################
#                                   getSoilLocationBasedUS                                 #
############################################################################################
def getSoilLocationBasedUS(lon, lat, plot_id, site_calc=False):
    # Load in LAB to Munsell conversion look-up table
    color_ref = pd.read_csv(config.MUNSELL_RGB_LAB_PATH)
    LAB_ref = color_ref[["L", "A", "B"]]
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
        mucompdata_pd[["cokey", "comppct_r", "compname", "distance_score", "slope_r"]],
        on="cokey",
        how="left",
    )

    # Filter out components with missing horizon depth data that aren't either
    # a Series, Variant, or Family
    filter_condition = muhorzdata_pd["cokey"].isin(cokey_series) | (
        pd.notnull(muhorzdata_pd["hzdept_r"]) & pd.notnull(muhorzdata_pd["hzdepb_r"])
    )
    muhorzdata_pd = muhorzdata_pd[filter_condition].drop_duplicates().reset_index(drop=True)

    # Check for duplicate component instances
    hz_drop = drop_cokey_horz(muhorzdata_pd)
    if hz_drop is not None:
        muhorzdata_pd = muhorzdata_pd[~muhorzdata_pd.cokey.isin(hz_drop)]

    muhorzdata_pd.reset_index(drop=True, inplace=True)

    # Extract unique cokeys and subset mucompdata_pd
    comp_key = muhorzdata_pd["cokey"].unique().tolist()
    mucompdata_pd = mucompdata_pd[mucompdata_pd["cokey"].isin(comp_key)]

    # Sort mucompdata_pd based on 'distance_score' and 'distance'
    mucompdata_pd.sort_values(["distance_score", "distance"], ascending=[False, True], inplace=True)
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

    # Group data by cokey (component key)
    muhorzdata_group_cokey = [group for _, group in muhorzdata_pd.groupby("cokey", sort=False)]

    getProfile_cokey = []
    comp_max_depths = []
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

        if site_calc:
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

    if site_calc:
        aws_PIW90, var_imp = soil_sim(muhorzdata_pd)
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
                                lab2munsell(color_ref, LAB_ref, LAB=lab)
                                if lab[0] and lab[1] and lab[2]
                                else ""
                            )
                            for lab in lab_parse
                        ]
                        munsell_lyrs.append(dict(zip(l_d.index, munsell_values)))

                    if site_calc:
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
                                OSD_sand_intpl = OSD_sand_intpl.loc[
                                    : comp_max_depths.iloc[index, 2]
                                ]
                                OSD_clay_intpl = OSD_clay_intpl.loc[
                                    : comp_max_depths.iloc[index, 2]
                                ]
                                OSD_rfv_intpl = OSD_rfv_intpl.loc[: comp_max_depths.iloc[index, 2]]

                            # Create the compname and cokey dataframes
                            compname_df = pd.DataFrame(
                                [group_sorted.compname.unique()] * len(OSD_sand_intpl)
                            )
                            cokey_df = pd.DataFrame(
                                [group_sorted.cokey.unique()] * len(OSD_sand_intpl)
                            )

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

                        # Update cec, ph, and ec layers if they contain only a single empty string
                        for lyr in [cec_lyrs, ph_lyrs, ec_lyrs]:
                            if len(lyr[index]) == 1 and lyr[index][0] == "":
                                lyr[index] = dict(zip(hz_lyrs[index], [""] * len(hz_lyrs[index])))

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
                    index=np.arange(comp_max_depths.iloc[i, 2]),
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
                index=np.arange(comp_max_depths.iloc[i, 2]),
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
            hz_lyrs,
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
            "SELECT cokey,  ecoclassid, ecoclassname FROM coecoclass WHERE cokey IN ("
            + ",".join(map(str, comp_key))
            + ")"
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

        # Further processing and checks for missing ESD data
        ESDcompdata_pd = update_esd_data(ESDcompdata_pd)

        # Aggregate the ESD components for output
        for _, group in ESDcompdata_pd.groupby("cokey"):
            esd_data = {
                "ESD": {
                    "ecoclassid": group["ecoclassid"].tolist(),
                    "ecoclassname": group["ecoclassname"].tolist(),
                    "esd_url": group["esd_url"].tolist(),
                }
            }
            esd_comp_list.append(esd_data)
    else:
        # Fill the list with empty data if ESDcompdata_pd is not available
        esd_comp_list = [
            {"ESD": {"ecoclassid": "", "ecoclassname": "", "esd_url": ""}}
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
            print(ecositeID, ESD_geo)
            ESDcompdata_pd = get_esd_data(ecositeID, ESD_geo, ESDcompdata_pd)
            # Assign missing ESD for components that have other instances with an assigned ESD
            if ESDcompdata_pd is not None:
                if (
                    ESDcompdata_pd.ecoclassid.isnull().any()
                    or ESDcompdata_pd.ecoclassname.isnull().any()
                ):
                    ESDcompdata_pd["compname_grp"] = ESDcompdata_pd.compname.str.replace(
                        r"[0-9]+", ""
                    )
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

                ESDcompdata_group_cokey = [
                    g for _, g in ESDcompdata_pd.groupby(["cokey"], sort=False)
                ]
                for i in range(len(ESDcompdata_group_cokey)):
                    if ESDcompdata_group_cokey[i]["ecoclassname"].isnull().values.any():
                        esd_comp_list.append(
                            {"ESD": {"ecoclassid": "", "ecoclassname": "", "esd_url": ""}}
                        )
                    else:
                        esd_comp_list.append(
                            {
                                "ESD": {
                                    "ecoclassid": ESDcompdata_group_cokey[i]["ecoclassid"].tolist(),
                                    "ecoclassname": ESDcompdata_group_cokey[i][
                                        "ecoclassname"
                                    ].tolist(),
                                    "esd_url": ESDcompdata_group_cokey[i]["esd_url"].tolist(),
                                }
                            }
                        )
            else:
                for i in range(len(mucompdata_pd)):
                    esd_comp_list.append(
                        {"ESD": {"ecoclassid": "", "ecoclassname": "", "esd_url": ""}}
                    )
        else:
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

    if site_calc:
        # Create the soilIDRank_output list by combining the dataframes of various data sources
        soilIDRank_output = [
            pd.concat(
                [
                    getProfile_cokey[i][
                        ["compname", "sandpct_intpl", "claypct_intpl", "rfv_intpl"]
                    ],
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

    if site_calc:
        # Writing out list of data needed for soilIDRank
        if plot_id is None:
            soilIDRank_output_pd.to_csv(
                config.SOIL_ID_RANK_PATH,
                index=None,
                header=True,
            )
            mucompdata_cond_prob.to_csv(
                config.SOIL_ID_PROB_PATH,
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
                    "AWS_PIW90": aws_PIW90,
                    "Soil Data Value": var_imp,
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
    if site_calc:
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
            "AWS_PIW90": aws_PIW90,
            "Soil Data Value": var_imp,
            "soilList": output_SoilList,
        }
    else:
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
    located in utils.py
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
    # soil_df.fillna(value=None, inplace=True)

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
        pedon_slice_index = [x for x in pedon_slice_index if x < 200]

        if bedrock is not None:
            pedon_slice_index.extend(range(bedrock, 200))

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
        if not isinstance(lab_Color, pd.DataFrame):
            lab_Color = pd.DataFrame(lab_Color)
        if not lab_Color.isnull().all().all():  # Use all().all() to check the entire DataFrame
            lab_Color = lab_Color.apply(
                lambda x: [np.nan, np.nan, np.nan] if x.isnull().all() else x, axis=1
            )
            p_lab_intpl = [
                lab_Color.iloc[i].tolist()
                for i in range(len(lab_Color))
                for _ in range(horizonDepthT[i], horizonDepthB[i])
            ]
            p_lab_intpl_list = [item[0] for item in p_lab_intpl]  # Access the inner list
            p_lab_intpl = pd.DataFrame(p_lab_intpl_list, columns=["L", "A", "B"]).reset_index(
                drop=True
            )
        else:
            lab_Color = lab_Color.dropna()  # Remove rows where all elements are None
            p_lab_intpl = pd.DataFrame(
                np.nan, index=np.arange(200), columns=["L", "A", "B"]
            ).reset_index(drop=True)

        # Adjust depth interval for each dataset
        p_sandpct_intpl = adjust_depth_interval(p_sandpct_intpl)
        p_claypct_intpl = adjust_depth_interval(p_claypct_intpl)
        p_cfg_intpl = adjust_depth_interval(p_cfg_intpl)
        p_lab_intpl = adjust_depth_interval(p_lab_intpl)

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

    if pElev is None:
        try:
            elevation_data = get_elev_data(lon, lat)
            if elevation_data is not None:
                pElev = round(float(elevation_data["value"]), 3)
        except Exception as err:
            logging.error(f"Error rounding elevation data: {err}")
            pElev = None

    # Compute text completeness
    p_sandpct_intpl = [x for x in p_sandpct_intpl if x is not None and x == x]
    text_len = len(p_sandpct_intpl)
    text_thresholds = [1, 10, 20, 50, 70, 100]
    text_scores = [3, 8, 15, 25, 30, 35, 40]
    text_comp = compute_soilid_data_completeness(text_len, text_thresholds, text_scores)

    # Compute rf completeness
    p_cfg_intpl = [x for x in p_cfg_intpl if x is not None and x == x]
    rf_len = len(p_cfg_intpl)
    rf_thresholds = [1, 10, 20, 50, 70, 100, 120]
    rf_scores = [3, 6, 10, 15, 20, 23, 25]
    rf_comp = compute_soilid_data_completeness(rf_len, rf_thresholds, rf_scores)

    # Compute lab completeness
    p_lab_intpl = [x for x in p_lab_intpl if x is not None and x == x]
    lab_len = len(p_lab_intpl)
    lab_thresholds = [1, 10, 20, 50, 70, 100, 120]
    lab_scores = [1, 3, 6, 9, 12, 14, 15]
    lab_comp = compute_soilid_data_completeness(lab_len, lab_thresholds, lab_scores)

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
        soilIDRank_output_pd = pd.read_csv(config.SOIL_ID_RANK_PATH)
        mucompdata_pd = pd.read_csv(config.SOIL_ID_PROB_PATH)
        record_id = None
    else:
        # Read from database
        modelRun = load_model_output(plot_id)
        if modelRun:
            record_id = modelRun[0]
            soilIDRank_output_pd = pd.read_csv(io.StringIO(modelRun[2]))
            mucompdata_pd = pd.read_csv(io.StringIO(modelRun[3]))
        else:
            logging.error(f"Cannot find a plot with ID: {plot_id}")

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
    if p_hz_data.empty or p_bottom_depth.bottom_depth.le(0).any():
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
        depth_weight = np.concatenate((np.repeat(0.2, 20), np.repeat(1.0, 180)), axis=0)
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

    # Initialize variables for site similarity
    p_slope = pd.DataFrame(["sample_pedon", pSlope, pElev]).T
    p_slope.columns = ["compname", "slope_r", "elev_r"]

    # Check conditions to determine the data columns and feature weights
    if (pSlope is not None) and (p_bottom_depth.bottom_depth.any() > 0):
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
        cracks
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


# Generate data completeness score
def compute_soilid_data_completeness(length, thresholds, scores):
    for thres, score in zip(thresholds, scores):
        if length <= thres:
            return score
    return scores[-1]


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
        "r",
        "g",
        "b",
        "total_frag_volume",
        "claytotal_r",
        "sandtotal_r",
    ]:
        new_row[col] = np.nan
    return new_row
