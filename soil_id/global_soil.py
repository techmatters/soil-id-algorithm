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

import collections
import io

# Standard libraries
import logging
import re
from dataclasses import dataclass

# Third-party libraries
import numpy as np
import pandas as pd

from .color import calculate_deltaE2000
from .db import extract_hwsd2_data, fetch_table_from_db, get_WRB_descriptions, getSG_descriptions
from .services import get_soilgrids_classification_data, get_soilgrids_property_data
from .utils import (
    adjust_depth_interval,
    agg_data_layer,
    process_distance_scores,
    # assign_max_distance_scores,
    # calculate_location_score,
    drop_cokey_horz,
    getCF_fromClass,
    getClay,
    getProfile,
    getSand,
    getTexture,
    gower_distances,
    max_comp_depth,
    pedon_color,
    silt_calc,
)

# local libraries


@dataclass
class SoilListOutputData:
    soil_list_json: dict
    rank_data_csv: str
    map_unit_component_data_csv: str


# entry points
# getSoilLocationBasedGlobal
# list_soils
# rank_soils
# rankPredictionGlobal
# getSoilGridsGlobal
# getSoilGridsUS

# when a site is created, call list_soils/getSoilLocationBasedGlobal.
# when a site is created, call getSoilGridsGlobal
# after user has collected data, call rank_soils/rankPredictionGlobal.


##################################################################################################
#                                 getSoilLocationBasedGlobal                                     #
##################################################################################################
def list_soils_global(connection, lon, lat, buffer_dist=30000):
    # Extract HWSD2 Data
    try:
        hwsd2_data = extract_hwsd2_data(
            connection,
            lon,
            lat,
            table_name="hwsdv2",
            buffer_dist=buffer_dist,
        )
    except KeyError:
        return "Data_unavailable"

    if hwsd2_data.empty:
        return "Data_unavailable"

    # Component Data
    mucompdata_pd = hwsd2_data[["hwsd2", "fao90_name", "distance", "share", "compid"]]
    mucompdata_pd.columns = ["mukey", "compname", "distance", "comppct_r", "cokey"]
    mucompdata_pd["distance"] = pd.to_numeric(mucompdata_pd["distance"])
    mucompdata_pd["comppct_r"] = pd.to_numeric(mucompdata_pd["comppct_r"])
    mucompdata_pd = mucompdata_pd.drop_duplicates().reset_index(drop=True)

    ##############################################################################################
    # Individual probability
    # Based on Fan et al 2018 EQ 1, the conditional probability for each component is calculated
    # by taking the sum of all occurances of a component in the home and adjacent mapunits and
    # dividing this by the sum of all map units and components. We have modified this approach
    # so that each instance of a component occurance is evaluated separately and assinged a
    # weight and the max distance score for each component group is assigned to all component
    # instances.
    ##############################################################################################
    ExpCoeff = -0.00036888  # Decays to 0.25 @ 10km
    mucompdata_pd = process_distance_scores(
        mucompdata_pd,
        ExpCoeff,
        compkind_filter=False,  # Disable compkind filtering for global data
    )

    comp_key = mucompdata_pd["cokey"].tolist()

    # -----------------------------------------------------------------------------------------------------------------
    # Create horizon data table
    columns_to_select = [
        "compid",
        "topdep",
        "botdep",
        "id",
        "layer",
        "sand",
        "silt",
        "clay",
        "coarse",
        "cec_soil",
        "ph_water",
        "elec_cond",
        "share",
        "fao90_name",
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
    ]

    muhorzdata_pd = hwsd2_data[columns_to_select]
    muhorzdata_pd.columns = new_column_names
    muhorzdata_pd = muhorzdata_pd[muhorzdata_pd["cokey"].isin(comp_key)]
    muhorzdata_pd[["hzdept_r", "hzdepb_r"]] = (
        muhorzdata_pd[["hzdept_r", "hzdepb_r"]].fillna(0).astype(int)
    )
    muhorzdata_pd["texture"] = muhorzdata_pd.apply(getTexture, axis=1)
    muhorzdata_pd["texture"] = muhorzdata_pd["texture"].apply(
        lambda x: str(x) if isinstance(x, np.ndarray) else x
    )

    # Add distance column from mucompdata_pd using cokey link
    muhorzdata_pd = pd.merge(
        muhorzdata_pd, 
        mucompdata_pd[["cokey", "distance"]], 
        on="cokey", 
        how="left"
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

    # Group data by cokey for texture
    muhorzdata_group_cokey = list(muhorzdata_pd.groupby("cokey", sort=False))
    pd.set_option("display.max_rows", None)
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", None)
    pd.set_option("display.max_colwidth", None)

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

    for group_key, group in muhorzdata_group_cokey:
        profile = (
            group.sort_values(by="hzdept_r").drop_duplicates(keep="first").reset_index(drop=True)
        )

        # profile depth
        c_very_bottom = max_comp_depth(profile)

        # extract information to be combined later with site soil measurements
        sand_pct_intpl = getProfile(profile, "sandtotal_r")
        sand_pct_intpl.columns = ["c_sandpct_intpl", "c_sandpct_intpl_grp"]
        clay_pct_intpl = getProfile(profile, "claytotal_r")
        clay_pct_intpl.columns = ["c_claypct_intpl", "c_claypct_intpl_grp"]
        cf_pct_intpl = getProfile(profile, "total_frag_volume")
        cf_pct_intpl.columns = ["c_cfpct_intpl", "c_cfpct_intpl_grp"]
        cec_intpl = getProfile(profile, "CEC")
        cec_intpl.columns = ["c_cec_intpl"]
        ph_intpl = getProfile(profile, "pH")
        ph_intpl.columns = ["c_ph_intpl"]
        ec_intpl = getProfile(profile, "EC")
        ec_intpl.columns = ["c_ec_intpl"]

        n_rows = sand_pct_intpl.shape[0]  # assuming all have the same length

        combined_data = pd.concat(
            [
                sand_pct_intpl[["c_sandpct_intpl_grp"]],
                clay_pct_intpl[["c_claypct_intpl_grp"]],
                cf_pct_intpl[["c_cfpct_intpl_grp"]],
                pd.DataFrame({"compname": [sorted(profile.compname.unique())[0]] * n_rows}),
                pd.DataFrame({"cokey": [sorted(profile.cokey.unique())[0]] * n_rows}),
                pd.DataFrame({"comppct": [sorted(profile.comppct_r.unique())[0]] * n_rows}),
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

        snd_d, hz_depb = agg_data_layer(
            sand_pct_intpl.c_sandpct_intpl,
            bottom=c_bottom_temp["c_very_bottom"].iloc[0],
            depth=True,
        )
        cly_d = agg_data_layer(
            clay_pct_intpl.c_claypct_intpl,
            bottom=c_bottom_temp["c_very_bottom"].iloc[0],
            depth=False,
        )
        txt_d = [
            getTexture(row=None, sand=s, silt=(100 - (s + c)), clay=c) for s, c in zip(snd_d, cly_d)
        ]
        txt_d = pd.Series(txt_d, index=snd_d.index)

        rf_d = agg_data_layer(
            cf_pct_intpl.c_cfpct_intpl, bottom=c_bottom_temp["c_very_bottom"].iloc[0], depth=False
        )
        cec_d = agg_data_layer(
            cec_intpl.c_cec_intpl, bottom=c_bottom_temp["c_very_bottom"].iloc[0], depth=False
        )
        ph_d = agg_data_layer(
            ph_intpl.c_ph_intpl, bottom=c_bottom_temp["c_very_bottom"].iloc[0], depth=False
        )
        ec_d = agg_data_layer(
            ec_intpl.c_ec_intpl, bottom=c_bottom_temp["c_very_bottom"].iloc[0], depth=False
        )

        # Fill NaN values and append to lists
        for data_list, data in zip(
            [snd_lyrs, cly_lyrs, txt_lyrs, rf_lyrs, cec_lyrs, ph_lyrs, ec_lyrs, hz_lyrs],
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
        mucompdata_pd, c_bottom_depths[["compname", "c_very_bottom"]], on="compname", how="left"
    )
    mucompdata_pd = pd.merge(mucompdata_pd, clay_texture, on="compname", how="left")

    # Create index for component instance display
    mucompdata_comp_grps_list = []
    mucompdata_comp_grps = [group for _, group in mucompdata_pd.groupby("compname_grp", sort=True)]

    for group in mucompdata_comp_grps:
        group = group.sort_values(["distance", "compname"]).reset_index(drop=True)
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

    mucompdata_cond_prob = mucompdata_pd.sort_values(["cond_prob", "compname"], ascending=[False, True]).reset_index(
        drop=True
    )

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

    # --------------------------------------------------------------------------------------------------------------------------
    # API output: Code outputs HWSD data without any alteration (i.e., texture class averaging)

    # Handle NaN values
    mucompdata_cond_prob.replace([np.nan, "nan", "None", [None]], "", inplace=True)

    # Sort mucompdata_cond_prob by soilID_rank, cond_prob, and compname for deterministic tie-breaking
    mucompdata_cond_prob = mucompdata_cond_prob.sort_values(
        ["soilID_rank", "cond_prob", "compname"], ascending=[False, False, True]
    )

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

    # Merge component descriptions
    WRB_Comp_Desc = get_WRB_descriptions(
        connection, mucompdata_cond_prob["compname_grp"].drop_duplicates().tolist()
    )

    mucompdata_cond_prob = pd.merge(
        mucompdata_cond_prob, WRB_Comp_Desc, left_on="compname_grp", right_on="WRB_tax", how="left"
    )

    mucompdata_cond_prob = mucompdata_cond_prob.drop_duplicates().reset_index(drop=True)
    mucomp_index = mucompdata_cond_prob.index

    # Extract site information
    Site = [
        {
            "siteData": {
                "mapunitID": row.mukey,
                "componentID": row.cokey,
                "share": row.comppct_r,
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

    # Reordering lists using list comprehension and mucomp_index
    lists_to_reorder = [
        hz_lyrs,
        snd_lyrs,
        cly_lyrs,
        txt_lyrs,
        rf_lyrs,
        cec_lyrs,
        ph_lyrs,
        ec_lyrs,
    ]
    for idx, lst in enumerate(lists_to_reorder):
        if len(lst) < max(mucomp_index) + 1:
            print(
                f"List at index {idx} is too short: len={len(lst)}, max index in mucomp_index={max(mucomp_index)}"
            )
    reordered_lists = [[lst[i] for i in mucomp_index] for lst in lists_to_reorder]

    # Destructuring reordered lists for clarity
    (
        hz_lyrs,
        snd_lyrs,
        cly_lyrs,
        txt_lyrs,
        rf_lyrs,
        cec_lyrs,
        ph_lyrs,
        ec_lyrs,
    ) = reordered_lists

    # Generating output_SoilList
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
                row,
            )
        )
        for row in zip(
            ID, Site, hz_lyrs, snd_lyrs, cly_lyrs, txt_lyrs, rf_lyrs, cec_lyrs, ph_lyrs, ec_lyrs
        )
    ]

    def convert_to_serializable(obj):
        if isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):  # Convert NumPy integers to Python int
            return int(obj)
        elif isinstance(obj, np.floating):  # Convert NumPy floats to Python float
            return float(obj)
        elif isinstance(obj, np.ndarray):  # Convert NumPy arrays to lists
            return obj.tolist()
        else:
            return obj

    output_SoilList_cleaned = convert_to_serializable(output_SoilList)

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
        # "AWS_PIW90": aws_PIW90,
        # "Soil Data Value": var_imp,
        "soilList": output_SoilList_cleaned,
    }

    return SoilListOutputData(
        soil_list_json=soil_list_json,
        rank_data_csv=soilIDRank_output_pd.to_csv(index=None, header=True),
        map_unit_component_data_csv=mucompdata_cond_prob.to_csv(index=None, header=True),
    )


##############################################################################################
#                                   rankPredictionGlobal                                     #
##############################################################################################
def rank_soils_global(
    connection,
    lon,
    lat,
    list_output_data: SoilListOutputData,
    soilHorizon,
    topDepth,
    bottomDepth,
    rfvDepth,
    lab_Color,
    bedrock,
    cracks,
):
    # ------------------------------------------------------------------------------------------------
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
            # Find the last valid index where bottom depth is less than or equal to bedrock
            last_valid_index = soil_df.loc[soil_df["bottom"] <= bedrock].index[-1]
            # Filter the DataFrame up to the last valid index
            soil_df = soil_df.loc[:last_valid_index].copy()
            # Set the bottom depth of the last row to the bedrock depth
            soil_df.at[last_valid_index, "bottom"] = bedrock

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
        bottom = [int(x) for x in soil_df.bottom.tolist()]
        top = [int(x) for x in soil_df.top.tolist()]
        lab_Color = soil_df.lab_Color

        # Generate user specified percent clay, sand, and rfv distributions
        spt = [getSand(sh) for sh in soilHorizon]
        cpt = [getClay(sh) for sh in soilHorizon]
        p_cfg = [getCF_fromClass(rf) for rf in rfvDepth]

        p_sandpct_intpl = [
            spt[i] for i in range(len(soilHorizon)) for _ in range(top[i], bottom[i])
        ]
        p_claypct_intpl = [
            cpt[i] for i in range(len(soilHorizon)) for _ in range(top[i], bottom[i])
        ]
        p_cfg_intpl = [p_cfg[i] for i in range(len(soilHorizon)) for _ in range(top[i], bottom[i])]

        # Length of interpolated texture and RF depth
        p_bottom_depth = pd.DataFrame([-999, "sample_pedon", soil_df_slice.bottom.iloc[-1]]).T
        p_bottom_depth.columns = ["cokey", "compname", "bottom_depth"]

        # Pedon color data
        lab_Color = pd.Series(lab_Color)
        if not lab_Color.isnull().all():
            lab_Color = [[np.nan, np.nan, np.nan] if x is None else x for x in lab_Color]
            lab_Color = pd.DataFrame(lab_Color)
            pedon_LAB = pedon_color(lab_Color, top, bottom)

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

        # Adjust depth interval for each dataset
        p_sandpct_intpl = adjust_depth_interval(p_sandpct_intpl)
        p_claypct_intpl = adjust_depth_interval(p_claypct_intpl)
        p_cfg_intpl = adjust_depth_interval(p_cfg_intpl)

        # Construct final dataframe with adjusted data
        p_compname = pd.Series("sample_pedon", index=np.arange(len(p_sandpct_intpl)))
        p_hz_data = pd.concat([p_compname, p_sandpct_intpl, p_claypct_intpl, p_cfg_intpl], axis=1)
        p_hz_data.columns = [
            "compname",
            "sandpct_intpl",
            "claypct_intpl",
            "rfv_intpl",
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
        cr_df = pd.Series([np.nan])

        # Set default bottom depth data
        if bedrock is not None:
            p_bottom_depth = pd.DataFrame([-999, "sample_pedon", bedrock]).T
        else:
            p_bottom_depth = pd.DataFrame([-999, "sample_pedon", 0]).T
        p_bottom_depth.columns = ["cokey", "compname", "bottom_depth"]

    # --------------------------------------------------------------------------------------------------------------------------------------
    # Load in component data from soilIDList
    soilIDRank_output_pd = pd.read_csv(io.StringIO(list_output_data.rank_data_csv))
    mucompdata_pd = pd.read_csv(io.StringIO(list_output_data.map_unit_component_data_csv))

    # Create soil depth DataFrame and subset component depths based on max user depth
    # if no bedrock specified
    c_bottom_depths = mucompdata_pd[["compname", "c_very_bottom"]].rename(
        columns={"c_very_bottom": "bottom_depth"}
    )
    slices_of_soil = pd.concat([p_bottom_depth, c_bottom_depths], axis=0).reset_index(drop=True)

    compnames = mucompdata_pd[["compname", "compname_grp"]]

    # Determine the maximum depth based on bedrock and user input
    if bedrock is None:
        max_depth = min(p_bottom_depth.bottom_depth.values[0], 200)
    else:
        max_depth = 200

    # Adjust slices of soil if they exceed the determined max_depth
    slices_of_soil.loc[slices_of_soil.bottom_depth > max_depth, "bottom_depth"] = max_depth

    # Generate a matrix describing soil (1) vs. non-soil (0) for each slice
    soil_matrix = pd.DataFrame(
        np.nan, index=np.arange(max_depth), columns=np.arange(len(slices_of_soil))
    )

    for i in range(len(slices_of_soil)):
        slice_end = slices_of_soil.bottom_depth.iloc[i]
        soil_matrix.iloc[:slice_end, i] = 1
        soil_matrix.iloc[slice_end:max_depth, i] = 0

    # Determine if user has entered horizon data and if so, subset component horizon
    # data based on user input data
    if p_hz_data is None or p_hz_data.empty or p_bottom_depth.bottom_depth.le(0).any():
        soilIDRank_output_pd = None
    else:
        # Subset component soil properties to match user measured properties
        soilIDRank_output_pd = soilIDRank_output_pd[p_hz_data_names]

        # Subset soil_matrix to user measured slices
        soil_matrix = soil_matrix.loc[pedon_slice_index]

    # Horizon Data Similarity
    if soilIDRank_output_pd is not None:
        cokey_groups = [group for _, group in soilIDRank_output_pd.groupby("compname", sort=True)]

        # Create lists to store component statuses
        Comp_Rank_Status, Comp_Missing_Status, Comp_name = [], [], []

        # Check component status
        for group in cokey_groups:
            subset_group = group[p_hz_data_names].drop(columns="compname")
            if subset_group.isnull().values.all():
                Comp_Rank_Status.append("Not Ranked")
                Comp_Missing_Status.append("No Data")
            else:
                Comp_Rank_Status.append("Ranked")
                Comp_Missing_Status.append(
                    "Missing Data" if subset_group.isnull().values.any() else "Data Complete"
                )
            Comp_name.append(sorted(group["compname"].unique())[0])

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
        for i in range(len(cokey_groups)):
            horz_vars_temp = cokey_groups[i]
            horz_vars_temp = horz_vars_temp.reset_index(drop=True)
            horz_vars_temp = horz_vars_temp[horz_vars_temp.index.isin(pedon_slice_index)]
            horz_vars.append(horz_vars_temp)

        dis_mat_list = []

        for depth in (
            soil_matrix.index
        ):  # depth represents a user-recorded depth slice (e.g. 100, 101, …, 120)
            # Gather the slice from each horizon variable
            slice_list = [horizon.loc[depth] for horizon in horz_vars]

            # Concatenate slices horizontally then transpose so that each row is one component's data
            slice_df = pd.concat(slice_list, axis=1).T

            # If bedrock is specified and the depth is less than bedrock, filter out columns with missing data
            if bedrock is not None and depth < bedrock:
                # Get columns that are non-null after dropping compname
                sample_vars = (
                    slice_df.dropna(axis="columns").drop("compname", axis=1).columns.tolist()
                )

                # If there are fewer than 2 variables available, use the "sample_pedon" row to decide
                if len(sample_vars) < 2:
                    sample_vars = (
                        slice_df.loc[slice_df["compname"] == "sample_pedon"]
                        .dropna(axis="columns")
                        .drop("compname", axis=1)
                        .columns.tolist()
                    )

                # Subset slice_df to only include the sample variables that were kept
                slice_mat = slice_df.loc[:, slice_df.columns.isin(sample_vars)]
            else:
                slice_mat = slice_df.drop("compname", axis=1)

            # Compute the Gower distance on the prepared slice matrix.
            D = gower_distances(slice_mat)

            dis_mat_list.append(D)

        # Check if any components have all NaNs at every slice
        dis_mat_nan_check = np.ma.MaskedArray(dis_mat_list, mask=np.isnan(dis_mat_list))
        D_check = np.ma.average(dis_mat_nan_check, axis=0)
        Rank_Filter["rank_status"] = [
            "Not Ranked" if np.ma.is_masked(x) else "Ranked" for x in D_check[0][1:]
        ]

        # Calculate max dissimilarity per depth slice
        dis_max = max(map(np.nanmax, dis_mat_list))

        # Apply depth weight
        depth_weight = np.concatenate([np.repeat(0.2, 20), np.repeat(1.0, 180)])
        depth_weight = depth_weight[soil_matrix.index]

        # Infill NaN data
        for idx, dis_mat in enumerate(dis_mat_list):
            soil_slice = soil_matrix.iloc[idx].to_numpy(dtype=bool)

            # Mask of NaNs in dis_mat
            nan_mask = np.isnan(dis_mat)

            # Broadcast soil slice to row and column vectors
            soil_row = soil_slice[:, np.newaxis]  # column vector
            soil_col = soil_slice[np.newaxis, :]  # row vector

            # Matrix where one is soil and the other isn't
            mismatch_mask = (soil_row & ~soil_col) | (~soil_row & soil_col)

            # Matrix where neither is soil
            nonsoil_mask = ~soil_row & ~soil_col

            # Set values for NaNs based on condition
            dis_mat[nan_mask & mismatch_mask] = dis_max
            dis_mat[nan_mask & nonsoil_mask] = 0

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
            mucompdata_pd[["compname", "mukey", "cokey", "cond_prob", "Rank_Loc"]],
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
            mucompdata_pd[["compname", "mukey", "cokey", "cond_prob", "Rank_Loc"]],
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
    wmf = []
    wsf = []
    rmf = []
    rsf = []
    ymf = []
    ysf = []

    # Load color distribution data from NormDist2 (FAO90) table
    rows = fetch_table_from_db(connection, "NormDist2")
    row_id = 0
    for row in rows:
        # row is a tuple; iterate over its values.
        for value in row:
            if row_id == 0:
                wmf.append(value)
            elif row_id == 1:
                wsf.append(value)
            elif row_id == 2:
                rmf.append(value)
            elif row_id == 3:
                rsf.append(value)
            elif row_id == 4:
                ymf.append(value)
            elif row_id == 5:
                ysf.append(value)
        row_id += 1

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
        w_df, r_df, y_df = cr_df.iloc[0], cr_df.iloc[1], cr_df.iloc[2]

        # Vectorized computation of color probabilities
        def norm_pdf_vec(x, mean_arr, std_arr):
            var = np.square(std_arr)
            denom = np.sqrt(2 * np.pi * var)
            num = np.exp(-np.square(x - mean_arr) / (2 * var))
            return num / denom

        # Convert to numpy arrays
        wmf, wsf = np.array(wmf, dtype=np.float64), np.array(wsf, dtype=np.float64)
        rmf, rsf = np.array(rmf, dtype=np.float64), np.array(rsf, dtype=np.float64)
        ymf, ysf = np.array(ymf, dtype=np.float64), np.array(ysf, dtype=np.float64)

        prob_w = norm_pdf_vec(float(w_df), wmf, wsf)
        prob_r = norm_pdf_vec(float(r_df), rmf, rsf)
        prob_y = norm_pdf_vec(float(y_df), ymf, ysf)

        # Normalize probabilities
        def normalize(arr):
            min_val, max_val = np.min(arr), np.max(arr)
            return (
                (arr - min_val) / (max_val - min_val) if max_val != min_val else np.ones_like(arr)
            )

        prob_w = normalize(prob_w)
        prob_r = normalize(prob_r)
        prob_y = normalize(prob_y)

        # Prepare FAO soil groups for lookup
        fao_list = [item.lower() for item in fao90]
        fao_index_map = {name: i for i, name in enumerate(fao_list)}

        # Vectorized scoring loop
        compnames = D_final_horz.compname.str.lower()
        color_sim = []

        for name in compnames:
            soilgroup = re.sub(r"\d+$", "", " ".join(name.split()[1:])).strip()
            idx = fao_index_map.get(soilgroup, -1)
            if idx != -1:
                score = (prob_w[idx] + prob_r[idx] + prob_y[idx]) / 3.0
            else:
                score = 1.0
            color_sim.append(score)

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
    D_final_comp_grps = [g for _, g in D_final_horz.groupby("compname_grp", sort=True)]

    for comp_grps_temp in D_final_comp_grps:
        comp_grps_temp = comp_grps_temp.sort_values(["Score_Data", "compname"], ascending=[False, True]).reset_index(
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
        mucompdata_pd[["compname", "clay"]],
        on="compname",
        how="left",
    )
    D_final_loc = pd.merge(D_final_loc, Rank_Filter, on="compname", how="left")

    # If Score_Data returns NA, assign 0 to weight
    D_final_loc.loc[D_final_loc.Score_Data.isnull(), "weight"] = 0

    # Calculate the final score incorporating the location score
    location_weight = 1

    # Calculate the combined score
    Score_Data_Loc = (D_final_loc[["Score_Data", "cond_prob"]].sum(axis=1)) / (
        D_final_loc.weight + location_weight
    )

    D_final_loc["Score_Data_Loc"] = Score_Data_Loc

    # Rule-based final score adjustment
    for i, row in D_final_loc.iterrows():
        if cracks and row["clay"] == "Yes" and "vert" in row["compname"].lower():
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (
            bedrock is not None
            and 0 <= bedrock <= 10
            and "lithic leptosols" in row["compname"].lower()
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif bedrock is not None and 10 < bedrock <= 30 and "leptosols" in row["compname"].lower():
            D_final_loc.at[i, "Score_Data_Loc"] = 1.001
        elif (bedrock is None or bedrock > 50) and any(
            term in row["compname"].lower()
            for term in ["lithosols", "leptosols", "rendzinas", "rankers"]
        ):
            D_final_loc.at[i, "Score_Data_Loc"] = 0.001

    D_final_loc["Score_Data_Loc"] = D_final_loc["Score_Data_Loc"] / np.nanmax(
        D_final_loc["Score_Data_Loc"]
    )
    D_final_loc = D_final_loc.sort_values(["Score_Data_Loc", "compname"], ascending=[False, True])

    # Sorting and reindexing of final dataframe
    soilIDList_out = []

    # Group by 'compname_grp' with deterministic sorting
    for _, comp_grps_temp in D_final_loc.groupby("compname_grp", sort=True):
        comp_grps_temp = comp_grps_temp.sort_values(["Score_Data_Loc", "compname"], ascending=[False, True]).reset_index(
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
            "horz_score",
            "Score_Data_Loc",
            "cond_prob",
        ]
    ] = D_final_loc[
        [
            "Score_Data",
            "horz_score",
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
            "location": "global",
            "model": "v2",
        },
        "soilRank": Rank,
    }

    return output_data


##################################################################################################
#                                          getSoilGridsGlobal                                    #
##################################################################################################


def sg_list(connection, lon, lat):
    """
    Query the SoilGrids API (via get_soilgrids_property_data) and post-process
    the returned JSON into a structured dictionary that includes:

    1. Soil horizons data (sand, clay, cfvo, pH, cec, silt, texture) at multiple depths.
    2. Classification probabilities and descriptions (WRB taxonomy).
    3. Summarized or aggregated soil variables, with depth tracking.

    Args:
        lon (float): Longitude in decimal degrees (WGS84).
        lat (float): Latitude in decimal degrees (WGS84).

    Returns:
        dict: A nested dictionary with:
            - "status": "unavailable" if needed columns are missing or data is invalid
            - "metadata": Info about location, model version, and units
            - "soilGrids": Detailed data on horizons, classification, etc.
    """
    # 1. Call the SoilGrids API for the specified lon/lat
    sg_out = get_soilgrids_property_data(lon, lat)

    if not sg_out or (isinstance(sg_out, dict) and sg_out.get("status") == "unavailable"):
        logging.warning("No data returned from SoilGrids API.")
        return {"status": "unavailable"}

    # 2. Extract arrays/lists of relevant data from the JSON.
    #    The new JSON structure nests data under properties -> layers -> depths.
    try:
        layers = sg_out.get("properties", {}).get("layers", [])
        top_depths = []
        bottom_depths = []
        values = []
        names = []
        for layer in layers:
            prop_name = layer.get("name")
            for depth in layer.get("depths", []):
                depth_range = depth.get("range", {})
                # Append the top and bottom depths
                top_depths.append(depth_range.get("top_depth"))
                bottom_depths.append(depth_range.get("bottom_depth"))
                # Append the mean value (from the "values" dict)
                mean_value = depth.get("values", {}).get("mean")
                values.append(mean_value)
                # Save the property name (e.g., "cec", "cfvo", "clay", etc.)
                names.append(prop_name)
    except Exception as e:
        logging.error(f"Error extracting values from SoilGrids JSON: {e}")
        return {"status": "unavailable"}

    # 3. Basic checks to ensure we have enough data
    if not top_depths or not bottom_depths or not names or not values:
        logging.warning("Missing required depth or property data in SoilGrids response.")
        return {"status": "unavailable"}

    # In case bottom_depths is too short to do iloc[-1], handle gracefully
    try:
        bottom = pd.Series(bottom_depths, name="bottom_depth").iloc[-1]
    except IndexError:
        logging.warning("No bottom_depth found in the data.")
        return {"status": "unavailable"}

    # 4. Convert extracted lists to DataFrames
    df_top_depth = pd.DataFrame(top_depths, columns=["top_depth"])
    df_bottom_depth = pd.DataFrame(bottom_depths, columns=["bottom_depth"])
    df_values = pd.DataFrame(values, columns=["value"])

    # The code assumes each property repeats over the same set of depths.
    n_depths = len(df_top_depth)
    if len(names) % n_depths != 0:
        logging.warning(
            "Number of property names isn't a multiple of the number of depths. Check data."
        )
    df_names = pd.DataFrame(names, columns=["prop"])

    # 5. Combine everything into a single DataFrame.
    #    If the lengths differ, take the minimum length.
    min_len = min(len(df_names), len(df_top_depth), len(df_bottom_depth), len(df_values))
    df_names = df_names.iloc[:min_len].reset_index(drop=True)
    df_top_depth = df_top_depth.iloc[:min_len].reset_index(drop=True)
    df_bottom_depth = df_bottom_depth.iloc[:min_len].reset_index(drop=True)
    df_values = df_values.iloc[:min_len].reset_index(drop=True)

    sg_data = pd.concat([df_names, df_top_depth, df_bottom_depth, df_values], axis=1)

    # 6. Pivot the data into a wide form with each property as a column.
    try:
        sg_data_w = sg_data.pivot_table(
            index="bottom_depth",
            columns="prop",
            values="value",
            aggfunc="first",  # Change the aggregator if needed
        )
    except Exception as e:
        logging.error(f"Error pivoting the data: {e}")
        return {"status": "unavailable"}

    # 7. Attach the 'hzdept_r' and 'hzdepb_r' columns (the top & bottom depths)
    unique_depths = sg_data.drop_duplicates(subset="bottom_depth")
    bottom_to_top_map = dict(zip(unique_depths["bottom_depth"], unique_depths["top_depth"]))
    sg_data_w["hzdept_r"] = sg_data_w.index.map(bottom_to_top_map)
    sg_data_w["hzdepb_r"] = sg_data_w.index
    sg_data_w.reset_index(drop=True, inplace=True)

    # 8. Check if we have valid data for key columns (e.g., sand, clay, cfvo).
    needed_cols = {"sand", "clay", "cfvo"}
    missing_cols = needed_cols - set(sg_data_w.columns)
    if missing_cols:
        logging.warning(f"Missing columns: {missing_cols}. Data might be incomplete.")
        if len(missing_cols) == len(needed_cols):
            return {"status": "unavailable"}
    existing_needed_cols = needed_cols & set(sg_data_w.columns)
    if sg_data_w[list(existing_needed_cols)].isnull().all().all():
        return {"status": "unavailable"}

    # 9. Scale certain columns by 0.1 (as in the original code)
    cols_to_multiply = ["sand", "clay", "cfvo", "phh2o", "cec"]
    for col in cols_to_multiply:
        if col in sg_data_w.columns:
            sg_data_w[col] = sg_data_w[col] * 0.1

    # 10. Calculate silt and texture if not already present.
    if "silt" not in sg_data_w.columns:
        sg_data_w["silt"] = sg_data_w.apply(silt_calc, axis=1)
    if "texture" not in sg_data_w.columns:
        sg_data_w["texture"] = sg_data_w.apply(getTexture, axis=1)

    # 11. Get WRB classification & probabilities
    sg_tax = get_soilgrids_classification_data(lon, lat)
    if sg_tax:
        try:
            sg_tax_prob = pd.DataFrame(sg_tax["wrb_class_probability"], columns=["WRB_tax", "Prob"])
            sg_tax_prob.sort_values(["Prob", "WRB_tax"], ascending=[False, True], inplace=True)

            # Merge with descriptive info
            WRB_Comp_Desc = getSG_descriptions(connection, sg_tax_prob["WRB_tax"].tolist())
            TAXNWRB_pd = pd.merge(sg_tax_prob, WRB_Comp_Desc, on="WRB_tax", how="left")

            # Only handle top 3 entries (or fewer if less are returned)
            ranks_needed = min(3, len(TAXNWRB_pd))
            TAXNWRB_pd = TAXNWRB_pd.head(ranks_needed)
            TAXNWRB_pd.index = [f"Rank{i + 1}" for i in range(ranks_needed)]
        except Exception as e:
            logging.error(f"Error processing WRB classification: {e}")
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
    else:
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

    # 12. Gather aggregated values for each property
    depths = sg_data_w["hzdepb_r"]
    sand_pd = sg_data_w["sand"]
    clay_pd = sg_data_w["clay"]
    rfv_pd = sg_data_w["cfvo"]
    pH_pd = sg_data_w["phh2o"]
    cec_pd = sg_data_w["cec"]
    # 13. Additional texture calculations from aggregated values
    texture_pd = pd.DataFrame({"sand": sand_pd, "clay": clay_pd})
    texture_pd["sand"] = pd.to_numeric(texture_pd["sand"], errors="coerce")
    texture_pd["clay"] = pd.to_numeric(texture_pd["clay"], errors="coerce")
    texture_pd["silt"] = texture_pd.apply(silt_calc, axis=1)
    texture_pd = texture_pd.apply(getTexture, axis=1).replace([None], "")

    # 14. Build the WRB classification dictionary (components)
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
        TAXNWRB_pd["WRB_tax"],
        TAXNWRB_pd["Prob"],
        TAXNWRB_pd.get("Description_en", ""),
        TAXNWRB_pd.get("Management_en", ""),
        TAXNWRB_pd.get("Description_es", ""),
        TAXNWRB_pd.get("Management_es", ""),
        TAXNWRB_pd.get("Description_ks", ""),
        TAXNWRB_pd.get("Management_ks", ""),
        TAXNWRB_pd.get("Description_fr", ""),
        TAXNWRB_pd.get("Management_fr", ""),
    ]
    components_dict = {}
    for k, v in zip(component_keys, component_values):
        components_dict[k] = dict(zip(v.index, v.values))

    # 15. Create the final SoilGrids dictionary.
    SoilGrids = {
        "components": components_dict,
        "bottom_depth": dict(zip(depths.index, depths)),
        "bedrock": bottom,
    }
    remaining_keys = ["texture", "sand", "clay", "rock_fragments", "ph", "cec"]
    remaining_values = [
        texture_pd,
        sand_pd,
        clay_pd,
        rfv_pd,
        pH_pd,
        cec_pd,
    ]
    for k, v in zip(remaining_keys, remaining_values):
        SoilGrids[k] = dict(zip(v.index, v.values))

    # 16. Define metadata and return the final dictionary.
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

    return {"metadata": metadata, "soilGrids": SoilGrids}
