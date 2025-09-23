import pandas as pd


def finalize_rank_output(D_final_loc: pd.DataFrame, location: str):
    # Calculate minimum rank values per compname_grp for each rank field
    df_copy = D_final_loc.copy()

    # Group by compname_grp and find minimum rank values for each rank field
    min_ranks = (
        df_copy.groupby("compname_grp")
        .agg({"Rank_Data_Loc": "min", "Rank_Data": "min", "Rank_Loc": "min"})
        .reset_index()
    )

    # Merge minimum ranks back to original data
    df_copy = df_copy.merge(min_ranks, on="compname_grp", suffixes=("", "_grp"))
    Rank = [
        {
            "name": row.compname.capitalize(),
            "component": row.compname_grp.capitalize(),
            "componentID": row.cokey,
            "score_data_loc": (
                None if row.missing_status == "Location data only" else round(row.Score_Data_Loc, 3)
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
            "rank_data": None if row.missing_status == "Location data only" else row.Rank_Data,
            "rank_data_group": None
            if row.missing_status == "Location data only"
            else row.Rank_Data_grp,
            "score_loc": round(row.cond_prob, 3),
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
