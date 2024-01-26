# Standard libraries
import collections
import csv
import json
import re
import urllib

# Third-party libraries
import numpy as np
import pandas as pd
import requests

# Flask
from flask import current_app

# Import local fucntions
from model.local_functions_SoilID_v3 import (
    agg_data_layer,
    assign_max_distance_scores,
    calculate_deltaE2000,
    calculate_location_score,
    compute_data_completeness,
    drop_cokey_horz,
    extract_values,
    extract_WISE_data,
    get_WRB_descriptions,
    getCF_fromClass,
    getClay,
    getProfile,
    getProfile_SG,
    getSand,
    getSG_descriptions,
    getTexture,
    gower_distances,
    load_model_output,
    pedon_color,
    save_model_output,
    save_rank_output,
    save_soilgrids_output,
    silt_calc,
)
from scipy.stats import norm

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


##################################################################################################
#                                 getSoilLocationBasedGlobal                                     #
##################################################################################################
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
                data=data[column_name], bottom=c_bottom_temp["c_very_bottom"].iloc[0], depth=True
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
    WRB_Comp_Desc = get_WRB_descriptions(
        mucompdata_cond_prob["compname_grp"].drop_duplicates().tolist()
    )
    mucompdata_cond_prob = pd.merge(
        mucompdata_cond_prob, WRB_Comp_Desc, left_on="compname_grp", right_on="WRB_tax", how="left"
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
            ID, Site, hz_lyrs, snd_lyrs, cly_lyrs, txt_lyrs, rf_lyrs, cec_lyrs, ph_lyrs, ec_lyrs
        )
    ]

    # Save data
    if plot_id is None:
        soilIDRank_output_pd.to_csv(
            "%s/soilIDRank_ofile1.csv" % current_app.config["DATA_BACKEND"], index=None, header=True
        )
        mucompdata_cond_prob.to_csv(
            "%s/soilIDRank_ofile2.csv" % current_app.config["DATA_BACKEND"], index=None, header=True
        )
    else:
        save_model_output(
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


##############################################################################################
#                                   rankPredictionGlobal                                     #
##############################################################################################
def rankPredictionGlobal(
    lon, lat, soilHorizon, horizonDepth, rfvDepth, lab_Color, bedrock, cracks, plot_id=None
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
        # If bedrock has been recorded and the lowest soil depth associated with data is
        # greater than bedrock, then change lowest soil depth to bedrock depth
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
        soilHorizon, rfvDepth = soil_df["soilHorizon"].tolist(), soil_df["rfvDepth"].tolist()
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
        modelRun = load_model_output(plot_id)

        # Check if modelRun data was successfully fetched
        if modelRun:
            record_id = modelRun[0]
            soilIDRank_output = pd.read_csv(io.StringIO(modelRun[2]))
            mucompdata = pd.read_csv(io.StringIO(modelRun[3]))
        else:
            return "Cannot find a plot with this ID"

    # Group the soilIDRank_output dataframe by 'compname' and return
    grouped_soil_data = [group for _, group in soilIDRank_output.groupby("compname", sort=False)]

    # Create soil depth DataFrame and subset component depths based on max user depth
    # if no bedrock specified
    c_bottom_depths = mucompdata_pd[["compname", "c_very_bottom"]].rename(
        columns={"c_very_bottom": "bottom_depth"}
    )
    slices_of_soil = pd.concat([p_bottom_depth, c_bottom_depths], axis=0).reset_index(drop=True)

    compnames = mucompdata_pd[["compname", "compname_grp"]]

    # If bedrock is not specified, determine max_depth based on
    # user-recorded depth (limited to 120 cm)
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

    # Determine if user has entered horizon data and if so, subset component horizon
    # data based on user input data
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
                fao_list, wmf, wsf, rmf, rsf, ymf, ysf = fao74, wmf1, wsf1, rmf1, rsf1, ymf1, ysf1
            else:
                fao_list, wmf, wsf, rmf, rsf, ymf, ysf = fao90, wmf2, wsf2, rmf2, rsf2, ymf2, ysf2

            idx = fao_list.index(soilgroup) if soilgroup in fao_list else -1

            for mw, sw, mr, sr, my, sy in zip(wmf, wsf, rmf, rsf, ymf, ysf):
                prob_w.append(norm(float(mw), float(sw)).pdf(float(w_df)))
                prob_r.append(norm(float(mr), float(sr)).pdf(float(r_df)))
                prob_y.append(norm(float(my), float(sy)).pdf(float(y_df)))

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
        ["Score_Data", "horz_score", "Score_Data_Loc", "distance_score", "distance_score_norm"]
    ] = D_final_loc[
        ["Score_Data", "horz_score", "Score_Data_Loc", "distance_score", "distance_score_norm"]
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
        save_rank_output(record_id, model_version, json.dumps(result))

    return result


##################################################################################################
#                                          getSoilGridsGlobal                                    #
##################################################################################################
def getSoilGridsGlobal(lon, lat, plot_id=None):
    # -------------------------------------------------------------------------------------------

    # SoilGrids250
    # Construct the SoilGrids API v2 URL
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

    sg_api = (
        f"https://rest.isric.org/soilgrids/v2.0/properties/query?{urllib.parse.urlencode(params)}"
    )

    try:
        # Make the API request using the requests library
        response = requests.get(sg_api, timeout=160)
        response.raise_for_status()  # Check for unsuccessful status codes
        sg_out = response.json()

    except requests.RequestException:
        # Log the error and set the status to unavailable
        if plot_id is not None:
            save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
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

    # Pivot the data based on bottom depth and property name, setting values to
    # be the 'value' column
    sg_data_w = sg_data.pivot(index="bottom_depth", columns="prop", values="value")

    # Add the top and bottom depth columns to the wide DataFrame
    sg_data_w["hzdept_r"] = df_top_depth["top_depth"].head(6).tolist()
    sg_data_w["hzdepb_r"] = df_bottom_depth["bottom_depth"].head(6).tolist()

    # Reset the index for the resulting DataFrame
    sg_data_w.reset_index(drop=True, inplace=True)

    # Check if all values in the specified columns are NaN
    if sg_data_w[["sand", "clay", "cfvo"]].isnull().all().all():
        if plot_id is not None:
            save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
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
        params = urllib.parse.urlencode([("lon", lon), ("lat", lat), ("number_classes", 3)])
        api_url = f"https://rest.isric.org/soilgrids/v2.0/classification/query?{params}"

        # Fetch data from the API
        try:
            with request.urlopen(api_url, timeout=6) as response:
                sg_tax = json.load(response)
        except Exception:
            # Handle data fetch failure
            if plot_id is not None:
                # Assuming the function `save_soilgrids_output` exists elsewhere in the code
                save_soilgrids_output(plot_id, 1, json.dumps({"status": "unavailable"}))
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
            if return_depth:
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
        remaining_keys = ["bottom_depth", "texture", "sand", "clay", "rock_fragments", "ph", "cec"]
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
            save_soilgrids_output(
                plot_id, model_version, json.dumps({"metadata": metadata, "soilGrids": SoilGrids})
            )

        # Return the final result
        return {"metadata": metadata, "soilGrids": SoilGrids}
