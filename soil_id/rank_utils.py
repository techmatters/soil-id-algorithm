import pandas as pd


def finalize_rank_output(D_final_loc: pd.DataFrame, location: str):
    # Calculate minimum rank values per compname_grp for each rank field
    df_copy = D_final_loc.copy()

    # Find rows with minimum ranks and their corresponding scores for each group
    def get_min_values(group: pd.DataFrame):
        min_data_loc_idx = group["Rank_Data_Loc"].idxmin()
        min_data_idx = group["Rank_Data"].idxmin()
        min_loc_idx = group["Rank_Loc"].idxmin()

        return pd.Series(
            {
                "Rank_Data_Loc_grp": group["Rank_Data_Loc"].min(),
                "Rank_Data_grp": group["Rank_Data"].min(),
                "Rank_Loc_grp": group["Rank_Loc"].min(),
                "Score_Data_Loc_grp": group.loc[min_data_loc_idx, "Score_Data_Loc"],
                "Score_Data_grp": group.loc[min_data_idx, "Score_Data"],
                "Score_Loc_grp": group.loc[min_loc_idx, "cond_prob"],
            }
        )

    min_values = df_copy.groupby("compname_grp").apply(get_min_values).reset_index()

    # Merge minimum values back to original data
    df_copy = df_copy.merge(min_values, on="compname_grp")

    Rank = [
        {
            "name": row.compname.capitalize(),
            "component": row.compname_grp.capitalize(),
            "componentID": row.cokey,
            "score_data_loc": (
                None if row.missing_status == "Location data only" else round(row.Score_Data_Loc, 3)
            ),
            "score_data_loc_group": (
                None
                if row.missing_status == "Location data only"
                else round(row.Score_Data_Loc_grp, 3)
            ),
            "rank_data_loc": (
                None if row.missing_status == "Location data only" else row.Rank_Data_Loc
            ),
            "rank_data_loc_group": (
                None if row.missing_status == "Location data only" else row.Rank_Data_Loc_grp
            ),
            "score_data": (
                None if row.missing_status == "Location data only" else round(row.Score_Data, 3)
            ),
            "score_data_group": (
                None if row.missing_status == "Location data only" else round(row.Score_Data_grp, 3)
            ),
            "rank_data": None if row.missing_status == "Location data only" else row.Rank_Data,
            "rank_data_group": None
            if row.missing_status == "Location data only"
            else row.Rank_Data_grp,
            "score_loc": round(row.cond_prob, 3),
            "score_loc_group": round(row.Score_Loc_grp, 3),
            "rank_loc": row.Rank_Loc,
            "rank_loc_group": row.Rank_Loc_grp,
            "componentData": row.missing_status,
            "not_displayed": (
                row.Rank_Data_Loc == "Not Displayed"
                if row.missing_status != "Location data only"
                else row.Rank_Loc == "Not Displayed"
            ),
        }
        for _, row in df_copy.iterrows()
    ]

    output_data = {
        "metadata": {
            "location": location,
            "model": "v2",
        },
        "soilRank": Rank,
    }

    return output_data
