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

# Third-party libraries
import re
from typing import List

import numpy as np
import pandas as pd
from composition_stats import ilr, ilr_inv
from scipy.stats import spearmanr

from .utils import (
    acomp,
    calculate_aws,
    calculate_vwc_awc,
    getCF_class,
    getTexture,
    information_gain,
    process_data_with_rosetta,
    regularize_matrix,
    remove_organic_layer,
    rename_simulated_soil_profile_columns,
    simulate_correlated_triangular,
    slice_and_aggregate_soil_data,
)


# ------------------------------------------------------------------------
def soil_sim(muhorzdata_pd):
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
        d. Simulate all properties and then perform inverse transform on ilr1 and ilr2
           to obtain sand, silt, and clay values.
        e. Append simulated values to dataframe

      Step 3. Run Rosetta and other Van Genuchten equations to calcuate AWS in top 50
              cm using simulated dataframe.
    """

    # Step 1. Calculate a local soil property correlation matrix

    # Subset data based on required soil inputs
    sim_columns = [
        "compname_grp",
        "hzname",
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
        "total_frag_volume",
    ]

    sim = muhorzdata_pd[sim_columns]

    if sim["hzname"].str.contains("O").any():
        sim = remove_organic_layer(sim)
    sim = sim.rename(columns={"total_frag_volume": "rfv_r"})
    sim["rfv_h"] = None
    sim["rfv_l"] = None

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
        "rfv_l",
        "rfv_r",
        "rfv_h",
    ]
    # infill missing data
    sim = infill_soil_data(sim)

    agg_data = []

    # Group data by compname_grp
    sim_group_compname = [group for _, group in sim.groupby("compname_grp", sort=False)]
    for index, group in enumerate(sim_group_compname):
        # aggregate data into 0-30 and 30-100 or bottom depth
        group_ag = slice_and_aggregate_soil_data(group[sim_data_columns])
        group_ag["compname_grp"] = group["compname_grp"].unique()[0]
        group_ag["distance_score"] = group["distance_score"].unique()[0]
        group_ag = group_ag.drop("Depth", axis=1)
        # group_ag = group_ag.where(pd.notna(group_ag), 0.01)
        group_ag = group_ag.reset_index(drop=True)
        agg_data.append(group_ag)

    # Concatenate the results for each column into a single dataframe
    agg_data_df = pd.concat(agg_data, axis=0, ignore_index=True).dropna().reset_index(drop=True)
    if agg_data_df["compname_grp"].nunique() < 2:
        aws_PIW90 = "Data not available"
        var_imp = "Data not available"
    else:
        # Extract columns with names ending in '_r'
        agg_data_r = agg_data_df[[col for col in agg_data_df.columns if col.endswith("_r")]]

        # Compute the local correlation matrix (Spearman correlation matrix)
        rep_columns = agg_data_r.drop(columns=["sandtotal_r", "silttotal_r", "claytotal_r"])
        # correlation_matrix, _ = spearmanr(selected_columns, axis=0)

        ilr_site_txt = ilr(agg_data_df[["sandtotal_r", "silttotal_r", "claytotal_r"]].values)
        cor_cols = [
            "ilr1",
            "ilr2",
            "dbovendry_r",
            "wthirdbar_r",
            "wfifteenbar_r",
        ]
        rep_columns["ilr1"] = pd.Series(ilr_site_txt[:, 0])
        rep_columns["ilr2"] = pd.Series(ilr_site_txt[:, 1])
        rep_columns[cor_cols] = rep_columns[cor_cols].replace(0, 0.01)
        is_constant = rep_columns["rfv_r"].nunique() == 1
        if is_constant:
            cor_cols = [
                "ilr1",
                "ilr2",
                "dbovendry_r",
                "wthirdbar_r",
                "wfifteenbar_r",
            ]
        else:
            cor_cols = [
                "ilr1",
                "ilr2",
                "dbovendry_r",
                "wthirdbar_r",
                "wfifteenbar_r",
                "rfv_r",
            ]

        # remove truncated profile layers from correlation matrix
        rep_columns = rep_columns[rep_columns["wthirdbar_r"] != 0.01]

        correlation_matrix_data = rep_columns[cor_cols]

        local_correlation_matrix, _ = spearmanr(correlation_matrix_data, axis=0)

        # Define global correlation matrix data (derived from KSSL SPCS)
        # Used if unable to calculate a local_correlation_matrix
        global_correlation_matrix = np.array(
            [
                [1.0000000, 0.6149869, -0.30000031, 0.5573609, 0.5240642, -0.33380115],  # "ilr1"
                [0.6149869, 1.0000000, -0.18742324, 0.5089337, 0.7580271, -0.32807023],  # "ilr2"
                [
                    -0.30000031,
                    -0.18742324,
                    1.00000000,
                    -0.7706528,
                    -0.5117794,
                    0.02803607,
                ],  # "dbovendry_r"
                [
                    0.5573609,
                    0.5089337,
                    -0.77065276,
                    1.0000000,
                    0.7826896,
                    -0.14029112,
                ],  # "wthirdbar_r"
                [
                    0.5240642,
                    0.7580271,
                    -0.51177936,
                    0.7826896,
                    1.0000000,
                    -0.17901686,
                ],  # "wfifteenbar_r"
                [
                    -0.33380115,
                    -0.32807023,
                    0.02803607,
                    -0.14029112,
                    -0.17901686,
                    1.00000000,
                ],  # "rfv_r"
            ]
        )

        # Check for NaNs or infs in the local correlation matrix and if so,
        # replace with global matrix
        if np.isnan(local_correlation_matrix).any() or np.isinf(local_correlation_matrix).any():
            local_correlation_matrix = global_correlation_matrix

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

        for index, row in agg_data_df.iterrows():
            """
            Step 2a. Simulate sand/silt/clay percentages
            """
            # Extract and format data params
            sand_params = [row["sandtotal_l"], row["sandtotal_r"], row["sandtotal_h"]]
            silt_params = [row["silttotal_l"], row["silttotal_r"], row["silttotal_h"]]
            clay_params = [row["claytotal_l"], row["claytotal_r"], row["claytotal_h"]]

            params_txt = [sand_params, silt_params, clay_params]

            """
            Step 2b. Perform processing steps on data
            - Convert simulated data using the acomp function and then compute the isometric
              log-ratio transformation.
            """

            simulated_txt = acomp(
                simulate_correlated_triangular(
                    (int(row["distance_score"] * 1000)), params_txt, texture_correlation_matrix
                )
            )
            simulated_txt_ilr = ilr(simulated_txt)

            """
            Step 2c. Extract l,r,h values (min, median, max for ilr1 and ilr2) and format into a
                     params object for simiulation
            """
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
            if is_constant:
                params = [
                    [ilr1_l, ilr1_r, ilr1_h],
                    [ilr2_l, ilr2_r, ilr2_h],
                    [row["dbovendry_l"], row["dbovendry_r"], row["dbovendry_h"]],
                    [row["wthirdbar_l"], row["wthirdbar_r"], row["wthirdbar_h"]],
                    [row["wfifteenbar_l"], row["wfifteenbar_r"], row["wfifteenbar_h"]],
                ]
            else:
                params = [
                    [ilr1_l, ilr1_r, ilr1_h],
                    [ilr2_l, ilr2_r, ilr2_h],
                    [row["dbovendry_l"], row["dbovendry_r"], row["dbovendry_h"]],
                    [row["wthirdbar_l"], row["wthirdbar_r"], row["wthirdbar_h"]],
                    [row["wfifteenbar_l"], row["wfifteenbar_r"], row["wfifteenbar_h"]],
                    [row["rfv_l"], row["rfv_r"], row["rfv_h"]],
                ]

            """
            Step 2d. Simulate all properties and then perform inverse transform on ilr1 and ilr2
                     to obtain sand, silt, and clay values.
            """
            # Initialize sim_data to an empty DataFrame with expected columns
            sim_data = pd.DataFrame(
                columns=[
                    "ilr1",
                    "ilr2",
                    "bulk_density_third_bar",
                    "water_retention_third_bar",
                    "water_retention_15_bar",
                    "rfv",
                ]
            )

            # Check diagonal elements and off-diagonal range
            if not np.all(np.diag(local_correlation_matrix) >= 0.99999999999999) or np.any(
                np.abs(local_correlation_matrix - np.eye(*local_correlation_matrix.shape)) > 1
            ):
                print(
                    f"""LinAlgError encountered in row index: {index}.
                           Correlation matrix diagonal/off-diagonal values are not valid."""
                )

            # Proceed with the simulation
            try:
                eigenvalues = np.linalg.eigvals(local_correlation_matrix)
                epsilon = 1e-8
                # Regularize the matrix if any eigenvalue is non-positive or very close to zero
                if np.any(eigenvalues <= epsilon):
                    print(
                        f"Regularizing matrix due to non-positive or small eigenvalues at row {index}."  # noqa: E501
                    )
                    local_correlation_matrix = regularize_matrix(local_correlation_matrix)

                n_sim = max(int(row["distance_score"] * 1000), 20)
                sim_data = simulate_correlated_triangular(
                    n=n_sim,
                    params=params,
                    correlation_matrix=local_correlation_matrix,
                )

            except np.linalg.LinAlgError:
                print("Adjusted matrix is still not positive definite.")
                continue  # Skip this iteration

            except ZeroDivisionError:
                # Handle the division by zero error
                print(f"Division by zero encountered in row index: {index}")
                continue  # Skip this iteration
            # Process sim_data only if it's valid
            if not sim_data.shape[0] == 0:
                if is_constant:
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
                    rfv_unique = rep_columns["rfv_r"].unique()[0]
                    sim_data["rfv"] = rfv_unique
                else:
                    sim_data = pd.DataFrame(
                        sim_data,
                        columns=[
                            "ilr1",
                            "ilr2",
                            "bulk_density_third_bar",
                            "water_retention_third_bar",
                            "water_retention_15_bar",
                            "rfv",
                        ],
                    )

                sim_data["water_retention_third_bar"] = sim_data["water_retention_third_bar"].div(
                    100
                )
                sim_data["water_retention_15_bar"] = sim_data["water_retention_15_bar"].div(100)
                sim_txt = ilr_inv(sim_data[["ilr1", "ilr2"]])
                sim_txt = pd.DataFrame(sim_txt, columns=["sand_total", "silt_total", "clay_total"])
                sim_txt = sim_txt.multiply(100)
                multi_sim = pd.concat([sim_data.drop(columns=["ilr1", "ilr2"]), sim_txt], axis=1)
                multi_sim["compname_grp"] = row["compname_grp"]
                multi_sim["hzdept_r"] = row["hzdept_r"]
                multi_sim["hzdepb_r"] = row["hzdepb_r"]
                sim_data_out.append(multi_sim)

        """
        Step 2e. Append simulated values to dataframe
        """
        sim_data_df = pd.concat(sim_data_out, axis=0, ignore_index=True)

        # Convert NaN values to None
        sim_data_df = sim_data_df.where(pd.notna(sim_data_df), None)

        # ------------------------------------------------------------------------------------
        # Step 3. Run Rosetta and other Van Genuchten equations
        # ------------------------------------------------------------------------------------

        # Step 3a: Run Rosetta

        # Convert the DataFrame to the desired format
        variables = [
            "sand_total",
            "silt_total",
            "clay_total",
            "bulk_density_third_bar",
            "water_retention_third_bar",
            "water_retention_15_bar",
        ]

        rosetta_data = process_data_with_rosetta(sim_data_df, vars=variables, v=3)

        # Create layerID
        sim_data_df["layerID"] = (
            sim_data_df["compname_grp"] + "_" + sim_data_df["hzdept_r"].astype(str)
        )
        rosetta_data["layerID"] = sim_data_df["layerID"]

        awc = calculate_vwc_awc(rosetta_data)
        awc = awc.map(lambda x: x.tolist() if isinstance(x, np.ndarray) else x)
        awc["top"] = sim_data_df["hzdept_r"]
        awc["bottom"] = sim_data_df["hzdepb_r"]
        awc["compname_grp"] = sim_data_df["compname_grp"]

        awc_grouped = awc.groupby(["top"])
        data_len_depth = awc_grouped.apply(
            lambda x: pd.DataFrame({"depth_len": [len(x)]}, index=[x.name]), include_groups=False
        )

        awc = awc.merge(data_len_depth, on="top", how="left")

        # Step 3b: Reshaping awc data

        # Apply the function to the entire ROI by depth
        awc_grouped_bottom = awc.groupby(["bottom"])
        awc_quant_list = awc_grouped_bottom.apply(
            lambda x: pd.DataFrame(
                {
                    "awc_quant": x["awc"].quantile([0.05, 0.50, 0.95]).values,
                    "prob": [0.05, 0.50, 0.95],
                    "n": len(x) / x["depth_len"].iloc[0],
                }
            ),
            include_groups=False,
        ).reset_index(level=[0, 1])

        # Pivoting the DataFrame
        awc_comp_quant = awc_quant_list.pivot(
            index=["bottom", "n"], columns="prob", values="awc_quant"
        ).reset_index()

        # Renaming columns for clarity
        awc_comp_quant.columns = ["bottom", "n", "0.05", "0.50", "0.95"]

        # Step 3c: Joining with mu_data and filtering distinct rows
        awc_comp_quant["top"] = np.where(
            awc_comp_quant["bottom"] <= 30, 0, np.where(awc_comp_quant["bottom"] > 30, 30, np.nan)
        )

        # Step 3d: Calculating depth
        awc_comp_quant["depth"] = awc_comp_quant["bottom"] - awc_comp_quant["top"]

        # Step 3e: Group by 'compname_grp' and merge
        aws05 = calculate_aws(awc_comp_quant, "0.05")
        aws95 = calculate_aws(awc_comp_quant, "0.95")

        """
        Width of the 90th prediction interval for available water
        storage in the top 100cm of soil (aws_PIW90). This value
        represents the weighted average of all soils mapped within
        the area of interest (AOI). Values greater than 3%
        indicate significant heterogenity in mapped soil properties.
        """
        aws_PIW90 = aws95["aws0.95_100"] - aws05["aws0.05_100"]
        aws_PIW90 = round(float(aws_PIW90.iloc[0]), 2)
        # ---------------------------------------------------------------------------
        # Calculate Information Gain, i.e., soil input variable importance

        sim_data_df["texture"] = sim_data_df.apply(getTexture, axis=1)
        sim_data_df["rfv_class"] = sim_data_df.apply(getCF_class, axis=1)

        # Remove the 'hzdepb_r' column
        sim_data_df = sim_data_df.drop(columns=["hzdepb_r"], errors="ignore")
        grp_list = sim_data_df["compname_grp"].unique().tolist()
        # Define the soil property columns
        soil_property_columns = ["texture", "rfv_class"]

        # Filter only the relevant columns
        filter_columns = ["compname_grp", "hzdept_r"] + soil_property_columns
        filtered_df = sim_data_df[filter_columns]
        final_columns = ["compname_grp"] + soil_property_columns

        # Create a new DataFrame for each soil depth
        df1 = filtered_df[filtered_df["hzdept_r"] == 0].copy().reset_index(drop=True)
        df1 = df1[final_columns]
        rename_simulated_soil_profile_columns(df1, soil_property_columns, 0)

        df2 = filtered_df[filtered_df["hzdept_r"] == 30].copy().reset_index(drop=True)
        df2 = df2[final_columns]
        rename_simulated_soil_profile_columns(df2, soil_property_columns, 30)
        # Assuming df1 and df2 are your dataframes
        groups1 = df1.groupby("compname_grp")
        groups2 = df2.groupby("compname_grp")

        # Concatenating corresponding groups

        concatenated_groups = []
        for group_label in grp_list:
            group_df1 = groups1.get_group(group_label).reset_index(drop=True)
            if group_label in groups2.groups:
                group_df2 = groups2.get_group(group_label).reset_index(drop=True)
            else:
                group_df2 = pd.DataFrame(index=range(len(group_df1)), columns=group_df1.columns)
                group_df2.columns = df2.columns
                group_df2["compname_grp"] = group_df1["compname_grp"]
                group_df2 = group_df2.fillna(0)
                group_df2 = group_df2.reset_index(drop=True)
            concatenated_group = pd.concat(
                [group_df1, group_df2.drop("compname_grp", axis=1)], axis=1
            )
            concatenated_groups.append(concatenated_group)

        # Combine all concatenated groups into a single dataframe
        result_df = pd.concat(concatenated_groups, axis=0, ignore_index=True)
        var_imp = information_gain(
            result_df, ["compname_grp"], ["texture_0", "texture_30", "rfv_class_0", "rfv_class_30"]
        )
    return aws_PIW90, var_imp


# ------------------------------------------------------------------------
# Soil simulation support functions


def infill_soil_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    More robust version that handles missing soil texture data at the horizon level
    rather than excluding entire groups. Works on entire DataFrame.

    Args:
        df: Input soil DataFrame

    Returns:
        DataFrame with infilled texture data
    """
    # Step 1: Group by 'compname_grp' and process each group
    grouped = df.groupby("compname_grp")

    def process_group(group: pd.DataFrame) -> pd.DataFrame:
        """Process each group individually with horizon-level filtering"""
        group = group.copy()

        # Identify horizons within top 50cm that have complete texture data
        top_50_mask = group["hzdepb_r"] <= 50
        texture_cols = ["sandtotal_r", "claytotal_r", "silttotal_r"]

        # Check which horizons have missing texture data
        missing_texture = group[texture_cols].isnull().any(axis=1)

        # For horizons in top 50cm with missing texture data, try to infill or mark for exclusion
        problematic_horizons = top_50_mask & missing_texture

        if problematic_horizons.any():
            # Try to infill missing texture data using group averages or nearby horizons
            group = infill_missing_texture_data(group, problematic_horizons)

            # If infilling fails, mark those specific horizons as problematic
            # but keep the rest of the group
            still_missing = group[texture_cols].isnull().any(axis=1)
            if (top_50_mask & still_missing).any():
                # Add a flag column to identify problematic horizons
                group["texture_data_complete"] = ~(top_50_mask & still_missing)
            else:
                group["texture_data_complete"] = True
        else:
            group["texture_data_complete"] = True

        return group

    # Apply processing to all groups and combine results
    processed_groups = []
    for name, group in grouped:
        processed_group = process_group(group)
        processed_groups.append(processed_group)

    # Combine all processed groups
    result_df = pd.concat(processed_groups, ignore_index=True)

    # Infill the _l and _h values for all valid horizons
    result_df = infill_range_values(result_df)

    # Apply RFV imputation if needed
    try:
        result_df = result_df.apply(impute_rfv_values, axis=1)
    except Exception as e:
        print(f"Warning: RFV imputation failed: {e}")

    return result_df


# Range Value Infilling Functions


def infill_range_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Robust infilling of _l and _h values using data-driven approaches

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with infilled range values
    """
    df = df.copy()

    # Define parameter groups with their characteristics
    param_configs = {
        "texture": {
            "params": ["sandtotal", "claytotal", "silttotal"],
            "units": "%",
            "sum_constraint": 100,  # Sand + Clay + Silt should sum to ~100%
            "fallback_range": 8,
        },
        "bulk_density": {
            "params": ["dbovendry"],
            "units": "g/cm³",
            "typical_range": (0.8, 2.2),
            "fallback_range": 0.01,
        },
        "water_retention": {
            "params": ["wthirdbar", "wfifteenbar"],
            "units": "%",
            "typical_range": (0, 60),
            "fallback_range": {"wthirdbar": 1, "wfifteenbar": 0.6},
        },
    }

    # Process each parameter group
    for group_name, config in param_configs.items():
        for param in config["params"]:
            df = infill_parameter_range(df, param, config, group_name)

    # Final validation and constraint enforcement
    df = enforce_range_constraints(df)

    return df


def infill_parameter_range(
    df: pd.DataFrame, param: str, config: dict, group_name: str
) -> pd.DataFrame:
    """
    Infill _l and _h values for a specific parameter using multiple strategies

    Args:
        df: Input DataFrame
        param: Parameter name
        config: Parameter configuration
        group_name: Parameter group name

    Returns:
        DataFrame with infilled range values
    """
    df = df.copy()
    r_col = f"{param}_r"
    l_col = f"{param}_l"
    h_col = f"{param}_h"

    if r_col not in df.columns:
        return df

    # Strategy 1: Learn from existing complete ranges in the dataset
    learned_ranges = learn_parameter_ranges(df, param, config)

    # Strategy 2: Apply horizon-specific and component-group-specific ranges
    context_ranges = get_contextual_ranges(df, param, config)

    # Strategy 3: Infill missing _l values
    if l_col not in df.columns:
        df[l_col] = np.nan

    missing_l_mask = df[l_col].isnull() & df[r_col].notna()
    if missing_l_mask.any():
        df.loc[missing_l_mask, l_col] = df.loc[missing_l_mask].apply(
            lambda row: calculate_lower_bound(row, param, learned_ranges, context_ranges, config),
            axis=1,
        )

    # Strategy 4: Infill missing _h values
    if h_col not in df.columns:
        df[h_col] = np.nan

    missing_h_mask = df[h_col].isnull() & df[r_col].notna()
    if missing_h_mask.any():
        df.loc[missing_h_mask, h_col] = df.loc[missing_h_mask].apply(
            lambda row: calculate_upper_bound(row, param, learned_ranges, context_ranges, config),
            axis=1,
        )

    # Apply bounds checking
    df[l_col] = df[l_col].apply(lambda x: max(x, 0) if pd.notna(x) else x)
    if "typical_range" in config:
        max_val = config["typical_range"][1]
        df[h_col] = df[h_col].apply(lambda x: min(x, max_val) if pd.notna(x) else x)

    return df


def learn_parameter_ranges(df: pd.DataFrame, param: str, config: dict) -> dict:
    """
    Learn typical ranges from existing complete data

    Args:
        df: Input DataFrame
        param: Parameter name
        config: Parameter configuration

    Returns:
        Dictionary of learned ranges by context
    """
    r_col = f"{param}_r"
    l_col = f"{param}_l"
    h_col = f"{param}_h"

    # Get complete records (have all three values)
    complete_mask = df[[r_col, l_col, h_col]].notna().all(axis=1)
    complete_data = df[complete_mask]

    if complete_data.empty:
        return {"default_spread": config.get("fallback_range", 5)}

    # Calculate actual spreads
    lower_spreads = complete_data[r_col] - complete_data[l_col]
    upper_spreads = complete_data[h_col] - complete_data[r_col]

    ranges_by_context = {}

    # Learn ranges by horizon type
    if "hzname" in complete_data.columns:
        for hzname in complete_data["hzname"].dropna().unique():
            hz_data = complete_data[complete_data["hzname"] == hzname]
            if len(hz_data) >= 3:  # Need minimum sample size
                hz_lower = hz_data[r_col] - hz_data[l_col]
                hz_upper = hz_data[h_col] - hz_data[r_col]

                ranges_by_context[f"hzname_{hzname}"] = {
                    "lower_spread_median": hz_lower.median(),
                    "upper_spread_median": hz_upper.median(),
                    "lower_spread_std": hz_lower.std(),
                    "upper_spread_std": hz_upper.std(),
                    "sample_size": len(hz_data),
                }

    # Learn ranges by component group
    if "compname_grp" in complete_data.columns:
        for grp in complete_data["compname_grp"].dropna().unique():
            grp_data = complete_data[complete_data["compname_grp"] == grp]
            if len(grp_data) >= 5:
                grp_lower = grp_data[r_col] - grp_data[l_col]
                grp_upper = grp_data[h_col] - grp_data[r_col]

                ranges_by_context[f"compgrp_{grp}"] = {
                    "lower_spread_median": grp_lower.median(),
                    "upper_spread_median": grp_upper.median(),
                    "lower_spread_std": grp_lower.std(),
                    "upper_spread_std": grp_upper.std(),
                    "sample_size": len(grp_data),
                }

    # Overall dataset statistics
    ranges_by_context["overall"] = {
        "lower_spread_median": lower_spreads.median(),
        "upper_spread_median": upper_spreads.median(),
        "lower_spread_std": lower_spreads.std(),
        "upper_spread_std": upper_spreads.std(),
        "sample_size": len(complete_data),
    }

    return ranges_by_context


def get_contextual_ranges(df: pd.DataFrame, param: str, config: dict) -> dict:
    """
    Get parameter-specific contextual ranges based on soil science knowledge

    Args:
        df: Input DataFrame
        param: Parameter name
        config: Parameter configuration

    Returns:
        Dictionary of contextual ranges
    """
    contextual_ranges = {}

    # Texture-specific ranges
    if param in ["sandtotal", "claytotal", "silttotal"]:
        contextual_ranges.update(
            {
                "sandtotal": {"typical_spread": 12, "min_spread": 5, "max_spread": 25},
                "claytotal": {"typical_spread": 8, "min_spread": 3, "max_spread": 20},
                "silttotal": {"typical_spread": 10, "min_spread": 4, "max_spread": 22},
            }
        )

    # Bulk density ranges by horizon type
    elif param == "dbovendry":
        contextual_ranges.update(
            {
                "A_horizons": {"typical_spread": 0.15, "range": (0.8, 1.6)},
                "B_horizons": {"typical_spread": 0.12, "range": (1.0, 1.8)},
                "C_horizons": {"typical_spread": 0.10, "range": (1.2, 2.0)},
            }
        )

    # Water retention ranges
    elif param in ["wthirdbar", "wfifteenbar"]:
        contextual_ranges.update(
            {
                "wthirdbar": {"typical_spread": 3, "clay_factor": 0.4},
                "wfifteenbar": {"typical_spread": 2, "clay_factor": 0.3},
            }
        )

    return contextual_ranges


def calculate_lower_bound(
    row: pd.Series, param: str, learned_ranges: dict, context_ranges: dict, config: dict
) -> float:
    """
    Calculate appropriate lower bound for a parameter

    Args:
        row: DataFrame row
        param: Parameter name
        learned_ranges: Learned ranges from data
        context_ranges: Contextual ranges
        config: Parameter configuration

    Returns:
        Calculated lower bound
    """
    r_value = row[f"{param}_r"]
    if pd.isna(r_value):
        return np.nan

    # Priority order for range estimation
    spread_estimates = []

    # 1. Try horizon-specific learned range
    if "hzname" in row and pd.notna(row["hzname"]):
        hz_key = f"hzname_{row['hzname']}"
        if hz_key in learned_ranges and learned_ranges[hz_key]["sample_size"] >= 3:
            spread = learned_ranges[hz_key]["lower_spread_median"]
            if pd.notna(spread) and spread > 0:
                spread_estimates.append(("horizon_learned", spread))

    # 2. Try component group learned range
    if "compname_grp" in row and pd.notna(row["compname_grp"]):
        grp_key = f"compgrp_{row['compname_grp']}"
        if grp_key in learned_ranges and learned_ranges[grp_key]["sample_size"] >= 5:
            spread = learned_ranges[grp_key]["lower_spread_median"]
            if pd.notna(spread) and spread > 0:
                spread_estimates.append(("group_learned", spread))

    # 3. Use overall dataset learned range
    if "overall" in learned_ranges:
        spread = learned_ranges["overall"]["lower_spread_median"]
        if pd.notna(spread) and spread > 0:
            spread_estimates.append(("overall_learned", spread))

    # 4. Use contextual/pedological knowledge
    if param in context_ranges:
        ctx_spread = context_ranges[param].get("typical_spread", config.get("fallback_range", 5))

        # Adjust spread based on soil properties
        if param in ["wthirdbar", "wfifteenbar"] and "claytotal_r" in row:
            clay_content = row["claytotal_r"]
            if pd.notna(clay_content):
                clay_factor = context_ranges[param].get("clay_factor", 0.3)
                ctx_spread = ctx_spread + (clay_content * clay_factor / 100)

        spread_estimates.append(("contextual", ctx_spread))

    # 5. Fallback to default
    if not spread_estimates:
        fallback = config.get("fallback_range", 5)
        spread_estimates.append(("fallback", fallback))

    # Use the first (highest priority) estimate
    spread_method, spread = spread_estimates[0]

    # Calculate lower bound
    lower_bound = r_value - spread

    # Apply parameter-specific constraints
    lower_bound = max(lower_bound, 0)  # Never below 0

    return lower_bound


def calculate_upper_bound(
    row: pd.Series, param: str, learned_ranges: dict, context_ranges: dict, config: dict
) -> float:
    """
    Calculate appropriate upper bound for a parameter

    Args:
        row: DataFrame row
        param: Parameter name
        learned_ranges: Learned ranges from data
        context_ranges: Contextual ranges
        config: Parameter configuration

    Returns:
        Calculated upper bound
    """
    r_value = row[f"{param}_r"]
    if pd.isna(r_value):
        return np.nan

    # Similar logic to lower bound but for upper spreads
    spread_estimates = []

    # 1. Horizon-specific learned range
    if "hzname" in row and pd.notna(row["hzname"]):
        hz_key = f"hzname_{row['hzname']}"
        if hz_key in learned_ranges and learned_ranges[hz_key]["sample_size"] >= 3:
            spread = learned_ranges[hz_key]["upper_spread_median"]
            if pd.notna(spread) and spread > 0:
                spread_estimates.append(("horizon_learned", spread))

    # 2. Component group learned range
    if "compname_grp" in row and pd.notna(row["compname_grp"]):
        grp_key = f"compgrp_{row['compname_grp']}"
        if grp_key in learned_ranges and learned_ranges[grp_key]["sample_size"] >= 5:
            spread = learned_ranges[grp_key]["upper_spread_median"]
            if pd.notna(spread) and spread > 0:
                spread_estimates.append(("group_learned", spread))

    # 3. Overall dataset learned range
    if "overall" in learned_ranges:
        spread = learned_ranges["overall"]["upper_spread_median"]
        if pd.notna(spread) and spread > 0:
            spread_estimates.append(("overall_learned", spread))

    # 4. Contextual knowledge
    if param in context_ranges:
        ctx_spread = context_ranges[param].get("typical_spread", config.get("fallback_range", 5))

        # Adjust for soil properties
        if param in ["wthirdbar", "wfifteenbar"] and "claytotal_r" in row:
            clay_content = row["claytotal_r"]
            if pd.notna(clay_content):
                clay_factor = context_ranges[param].get("clay_factor", 0.3)
                ctx_spread = ctx_spread + (clay_content * clay_factor / 100)

        spread_estimates.append(("contextual", ctx_spread))

    # 5. Fallback
    if not spread_estimates:
        fallback = config.get("fallback_range", 5)
        spread_estimates.append(("fallback", fallback))

    spread_method, spread = spread_estimates[0]
    upper_bound = r_value + spread

    # Apply parameter-specific upper limits
    if "typical_range" in config:
        upper_bound = min(upper_bound, config["typical_range"][1])

    return upper_bound


def enforce_range_constraints(df: pd.DataFrame) -> pd.DataFrame:
    """
    Enforce logical constraints on range values

    Args:
        df: Input DataFrame

    Returns:
        DataFrame with enforced constraints
    """
    df = df.copy()

    # Texture constraints: ensure _l ≤ _r ≤ _h and sum constraints
    texture_params = ["sandtotal", "claytotal", "silttotal"]

    for param in texture_params:
        r_col, l_col, h_col = f"{param}_r", f"{param}_l", f"{param}_h"

        if all(col in df.columns for col in [r_col, l_col, h_col]):
            # Ensure _l ≤ _r ≤ _h
            df[l_col] = np.minimum(df[l_col], df[r_col])
            df[h_col] = np.maximum(df[h_col], df[r_col])

    # Check texture sum constraints (sand + clay + silt ≈ 100%)
    has_all_texture = all(f"{p}_r" in df.columns for p in texture_params)
    if has_all_texture:
        texture_sum = df[["sandtotal_r", "claytotal_r", "silttotal_r"]].sum(axis=1)
        # Flag records where sum is far from 100%
        sum_issues = ((texture_sum < 95) | (texture_sum > 105)) & texture_sum.notna()
        if sum_issues.any():
            print(f"Warning: {sum_issues.sum()} records have texture sums outside 95-105% range")

    # Water retention constraints: wfifteenbar ≤ wthirdbar
    if "wfifteenbar_r" in df.columns and "wthirdbar_r" in df.columns:
        # Fifteen bar should be ≤ third bar (higher pressure = less water)
        invalid_water = df["wfifteenbar_r"] > df["wthirdbar_r"]
        if invalid_water.any():
            print(f"Warning: {invalid_water.sum()} records have wfifteenbar > wthirdbar")

    return df


# Texture Data Recovery Functions
def infill_missing_texture_data(group: pd.DataFrame, problematic_mask: pd.Series) -> pd.DataFrame:
    """
    Attempt to infill missing texture data using pedologically-informed strategies

    Args:
        group: DataFrame group to process
        problematic_mask: Boolean mask for problematic horizons

    Returns:
        DataFrame with infilled texture data
    """
    group = group.copy()
    texture_cols = ["sandtotal_r", "claytotal_r", "silttotal_r"]

    # Ensure we have depth information
    if "hzdept_r" not in group.columns:
        print("Warning: Missing hzdept_r (top depth), using 0 for surface horizons")
        group["hzdept_r"] = group.get("hzdept_r", 0)

    # Strategy 1: PRIORITY - Match by horizon name (pedologically similar horizons)
    if "hzname" in group.columns:
        group = horizon_name_texture_infill(group, texture_cols, problematic_mask)

    # Check if we still have missing data after horizon name matching
    still_missing = group[texture_cols].isnull().any(axis=1)
    if not still_missing.any():
        return group  # All data recovered, no need for further strategies

    # Strategy 2: Depth-weighted averaging from similar depth ranges
    group = depth_weighted_infill(group, texture_cols, problematic_mask)

    # Check progress after depth-weighted infill
    still_missing = group[texture_cols].isnull().any(axis=1)
    if not still_missing.any():
        return group

    # Strategy 3: Vertical interpolation within individual soil components
    group = within_component_interpolation(group, texture_cols)

    # Check progress after interpolation
    still_missing = group[texture_cols].isnull().any(axis=1)
    if not still_missing.any():
        return group

    # Strategy 4: Cross-component interpolation for similar depths
    group = cross_component_depth_interpolation(group, texture_cols)

    # Strategy 5: Fallback to group statistics for any remaining missing values
    for col in texture_cols:
        if group[col].isnull().any():
            # Use depth-weighted group mean as last resort
            depth_weighted_mean = calculate_depth_weighted_mean(group, col)
            if not pd.isna(depth_weighted_mean):
                group[col] = group[col].fillna(depth_weighted_mean)

    return group


def horizon_name_texture_infill(
    group: pd.DataFrame, texture_cols: List[str], problematic_mask: pd.Series
) -> pd.DataFrame:
    """
    Infill missing texture data using horizons with matching or similar names

    Args:
        group: DataFrame group to process
        texture_cols: List of texture column names
        problematic_mask: Boolean mask for problematic horizons

    Returns:
        DataFrame with infilled values
    """
    group = group.copy()

    for idx in group[problematic_mask].index:
        if idx not in group.index:
            continue

        target_hzname = group.loc[idx, "hzname"]
        if pd.isna(target_hzname):
            continue

        # Clean and standardize horizon name for matching
        target_hzname_clean = standardize_horizon_name(target_hzname)

        for col in texture_cols:
            if pd.isna(group.loc[idx, col]):
                # Find matching horizon names with valid data
                matching_values = []
                similarity_scores = []

                for other_idx in group.index:
                    if other_idx == idx or pd.isna(group.loc[other_idx, col]):
                        continue

                    other_hzname = group.loc[other_idx, "hzname"]
                    if pd.isna(other_hzname):
                        continue

                    other_hzname_clean = standardize_horizon_name(other_hzname)
                    similarity = calculate_horizon_similarity(
                        target_hzname_clean, other_hzname_clean
                    )

                    if similarity > 0.5:  # Only use reasonably similar horizons
                        matching_values.append(group.loc[other_idx, col])
                        similarity_scores.append(similarity)

                if matching_values and similarity_scores:
                    # Use similarity-weighted average
                    if len(matching_values) == 1:
                        infilled_value = matching_values[0]
                    else:
                        infilled_value = np.average(matching_values, weights=similarity_scores)

                    group.loc[idx, col] = infilled_value

                    # Track infilling method for reporting
                    if "infill_method" not in group.columns:
                        group["infill_method"] = ""
                    group.loc[idx, "infill_method"] += f"{col}:hzname({target_hzname_clean}); "

    return group


def standardize_horizon_name(hzname: str) -> str:
    """
    Standardize horizon names for better matching

    Args:
        hzname: Raw horizon name

    Returns:
        Cleaned and standardized horizon name
    """
    if pd.isna(hzname):
        return ""

    # Convert to string and clean
    hzname = str(hzname).strip().upper()

    # Remove common suffixes/prefixes that don't affect texture matching
    # Remove numbers at the end (layer designations)
    hzname = re.sub(r"\d+$", "", hzname)

    # Remove common punctuation but keep important ones like '/'
    hzname = re.sub(r"[^\w/]", "", hzname)

    return hzname


def calculate_horizon_similarity(hz1: str, hz2: str) -> float:
    """
    Calculate similarity between two horizon names

    Args:
        hz1, hz2: Horizon names to compare

    Returns:
        Similarity score from 0 (no similarity) to 1 (identical)
    """
    if hz1 == hz2:
        return 1.0

    if not hz1 or not hz2:
        return 0.0

    # Check for main horizon letter match
    main_hz1 = hz1[0] if hz1 else ""
    main_hz2 = hz2[0] if hz2 else ""

    if main_hz1 == main_hz2:
        base_score = 0.8  # Same main horizon type

        # Bonus for additional character matches
        common_chars = set(hz1) & set(hz2)
        bonus = len(common_chars) / max(len(hz1), len(hz2)) * 0.2

        return min(1.0, base_score + bonus)

    # Check for related horizons (pedologically similar)
    related_groups = [
        {"A", "AP", "AE"},  # Surface horizons
        {"E", "EB", "BE"},  # Eluvial horizons
        {"B", "BT", "BW", "BC", "BS"},  # Illuvial/subsurface horizons
        {"C", "CB", "CR"},  # Parent material
        {"O", "OA", "OE"},  # Organic horizons
    ]

    for group in related_groups:
        if main_hz1 in group and main_hz2 in group:
            return 0.6  # Related horizon types

    # Handle transitional horizons like 'B/C', 'A/E'
    if "/" in hz1 or "/" in hz2:
        parts1 = hz1.split("/") if "/" in hz1 else [hz1]
        parts2 = hz2.split("/") if "/" in hz2 else [hz2]

        max_sim = 0
        for p1 in parts1:
            for p2 in parts2:
                sim = calculate_horizon_similarity(p1.strip(), p2.strip())
                max_sim = max(max_sim, sim)

        return max_sim * 0.8  # Slight penalty for composite matching

    return 0.0  # No similarity


def depth_weighted_infill(
    group: pd.DataFrame, texture_cols: List[str], problematic_mask: pd.Series
) -> pd.DataFrame:
    """
    Fill missing values using depth-weighted averages from similar depth ranges

    Args:
        group: DataFrame group to process
        texture_cols: List of texture column names
        problematic_mask: Boolean mask for problematic horizons

    Returns:
        DataFrame with infilled values
    """
    group = group.copy()

    for idx in group[problematic_mask].index:
        if idx not in group.index:
            continue

        target_top = group.loc[idx, "hzdept_r"]
        target_bottom = group.loc[idx, "hzdepb_r"]
        target_mid = (target_top + target_bottom) / 2

        # Find horizons with overlapping or similar depth ranges
        for col in texture_cols:
            if pd.isna(group.loc[idx, col]):
                similar_depth_values = []
                weights = []

                for other_idx in group.index:
                    if other_idx == idx or pd.isna(group.loc[other_idx, col]):
                        continue

                    other_top = group.loc[other_idx, "hzdept_r"]
                    other_bottom = group.loc[other_idx, "hzdepb_r"]
                    other_mid = (other_top + other_bottom) / 2

                    # Calculate overlap and depth similarity
                    overlap = max(0, min(target_bottom, other_bottom) - max(target_top, other_top))
                    depth_similarity = 1 / (1 + abs(target_mid - other_mid))

                    if overlap > 0 or abs(target_mid - other_mid) < 20:  # 20cm tolerance
                        weight = overlap + depth_similarity
                        similar_depth_values.append(group.loc[other_idx, col])
                        weights.append(weight)

                if similar_depth_values and weights:
                    weighted_avg = np.average(similar_depth_values, weights=weights)
                    group.loc[idx, col] = weighted_avg

    return group


def within_component_interpolation(group: pd.DataFrame, texture_cols: List[str]) -> pd.DataFrame:
    """
    Interpolate missing values within individual soil components based on depth

    Args:
        group: DataFrame group to process
        texture_cols: List of texture column names

    Returns:
        DataFrame with interpolated values
    """
    group = group.copy()

    # Group by individual soil component if component ID exists
    if "cokey" in group.columns:
        component_groups = group.groupby("cokey")
    else:
        # Fallback: treat entire group as one component
        component_groups = [(None, group)]

    processed_components = []

    for comp_id, comp_data in component_groups:
        comp_data = comp_data.copy().sort_values("hzdepb_r")

        for col in texture_cols:
            if comp_data[col].isnull().any() and comp_data[col].notna().any():
                # Interpolate missing values based on depth
                comp_data[col] = comp_data[col].interpolate(method="index")

        processed_components.append(comp_data)

    if len(processed_components) > 1:
        return pd.concat(processed_components, ignore_index=False)
    else:
        return processed_components[0] if processed_components else group


def cross_component_depth_interpolation(
    group: pd.DataFrame, texture_cols: List[str]
) -> pd.DataFrame:
    """
    Use data from other components at similar depths to fill missing values

    Args:
        group: DataFrame group to process
        texture_cols: List of texture column names

    Returns:
        DataFrame with infilled values
    """
    group = group.copy()

    for col in texture_cols:
        missing_mask = group[col].isnull()
        if not missing_mask.any():
            continue

        for idx in group[missing_mask].index:
            target_depth_mid = (group.loc[idx, "hzdept_r"] + group.loc[idx, "hzdepb_r"]) / 2

            # Find values from similar depths in other components
            depth_matches = []
            for other_idx in group.index:
                if other_idx == idx or pd.isna(group.loc[other_idx, col]):
                    continue

                other_depth_mid = (
                    group.loc[other_idx, "hzdept_r"] + group.loc[other_idx, "hzdepb_r"]
                ) / 2
                depth_diff = abs(target_depth_mid - other_depth_mid)

                if depth_diff <= 15:  # 15cm tolerance for similar depths
                    weight = 1 / (1 + depth_diff)  # Closer depths get higher weight
                    depth_matches.append((group.loc[other_idx, col], weight))

            if depth_matches:
                values, weights = zip(*depth_matches)
                weighted_value = np.average(values, weights=weights)
                group.loc[idx, col] = weighted_value

    return group


def calculate_depth_weighted_mean(group: pd.DataFrame, col: str) -> float:
    """
    Calculate depth-weighted mean for a column, giving more weight to thicker horizons

    Args:
        group: DataFrame group
        col: Column name

    Returns:
        Depth-weighted mean value
    """
    valid_data = group[group[col].notna()]
    if valid_data.empty:
        return np.nan

    # Calculate horizon thickness as weight
    thicknesses = valid_data["hzdepb_r"] - valid_data["hzdept_r"]
    values = valid_data[col]

    if thicknesses.sum() == 0:
        return values.mean()

    return np.average(values, weights=thicknesses)


# Function to infill missing soil rock fragment data.
def impute_rfv_values(row):
    """
    Improved RFV (Rock Fragment Volume) imputation with better logic and safety checks

    RFV represents the percentage of rock fragments by volume in soil horizons.
    Typical ranges: 0-85% (rarely above 85% in productive soils)

    Special case: If rfv_r = 0 OR rfv_r is NA, set to 0.02% with range 0.01-0.03% for simulation compatibility.
    """

    # Handle missing RFV or zero RFV - both mean "no significant rock fragments"
    if pd.isna(row.get("rfv_r")) or row.get("rfv_r") == 0:
        row["rfv_r"] = 0.02
        row["rfv_l"] = 0.01
        row["rfv_h"] = 0.03
        return row

    rfv_r = row["rfv_r"]

    # Ensure rfv_r is within reasonable bounds (0.01-85%)
    rfv_r = max(0.01, min(rfv_r, 85))
    row["rfv_r"] = rfv_r

    # Determine appropriate range based on rfv_r value and context
    if 0.01 <= rfv_r <= 5:
        # Low RFV: careful with lower bound to avoid going to 0
        lower_spread = min(rfv_r - 0.01, 2)  # Don't go below 0.01
        upper_spread = 4

    elif 5 < rfv_r <= 15:
        # Moderate RFV: standard range
        lower_spread = 3
        upper_spread = 5

    elif 15 < rfv_r <= 35:
        # High RFV: wider range reflects more variability
        lower_spread = 5
        upper_spread = 8

    elif 35 < rfv_r <= 60:
        # Very high RFV: even wider range
        lower_spread = 8
        upper_spread = 12

    else:  # rfv_r > 60
        # Extremely high RFV: largest range but respect upper bound
        lower_spread = 10
        upper_spread = min(15, 85 - rfv_r)  # Don't exceed 85%

    # Adjust spreads based on horizon context if available
    if "hzname" in row and pd.notna(row["hzname"]):
        hzname = str(row["hzname"]).upper()

        # Surface horizons typically more variable due to management
        if hzname.startswith("A") and rfv_r > 0.01:
            upper_spread *= 1.3

        # Bt horizons often more uniform
        elif "BT" in hzname:
            lower_spread *= 0.8
            upper_spread *= 0.8

        # C horizons more variable (parent material heterogeneity)
        elif hzname.startswith("C"):
            lower_spread *= 1.2
            upper_spread *= 1.2

    # Calculate bounds with safety checks
    rfv_l = max(0.01, rfv_r - lower_spread)  # Never below 0.01% for simulation
    rfv_h = min(85, rfv_r + upper_spread)

    # Only update _l and _h if they're missing
    if pd.isna(row.get("rfv_l")):
        row["rfv_l"] = rfv_l

    if pd.isna(row.get("rfv_h")):
        row["rfv_h"] = rfv_h

    # Final consistency check
    if not pd.isna(row.get("rfv_l")) and not pd.isna(row.get("rfv_h")):
        # Ensure logical order: rfv_l ≤ rfv_r ≤ rfv_h
        row["rfv_l"] = min(row["rfv_l"], rfv_r)
        row["rfv_h"] = max(row["rfv_h"], rfv_r)

        # Ensure rfv_l never goes to 0 for simulation compatibility
        if row["rfv_l"] <= 0:
            row["rfv_l"] = 0.01

        # Ensure reasonable minimum range for simulation
        if row["rfv_h"] - row["rfv_l"] < 0.02:
            spread = max(0.02, rfv_r * 0.2)  # Minimum 0.02% or 20% of value
            row["rfv_l"] = max(0.01, rfv_r - spread / 2)
            row["rfv_h"] = min(85, rfv_r + spread / 2)

    return row
