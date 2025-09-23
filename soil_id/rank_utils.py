import pandas as pd


def finalize_rank_output(D_final_loc: pd.DataFrame, location: str):
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
            "location": location,
            "model": "v2",
        },
        "soilRank": Rank,
    }

    return output_data
