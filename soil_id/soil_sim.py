# Third-party libraries
import numpy as np
import pandas as pd
from composition_stats import ilr, ilr_inv
from scipy.stats import spearmanr
from utils import (
    acomp,
    calculate_aws,
    calculate_vwc_awc,
    getCF_class,
    getTexture,
    infill_soil_data,
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
