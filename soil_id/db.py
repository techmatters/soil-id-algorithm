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
import logging

import pandas as pd

# Third-party libraries
import psycopg
from dotenv import load_dotenv

# local libraries
import soil_id.config

# Load .env file
load_dotenv()


def get_datastore_connection():
    """
    Establish a connection to the datastore using app configurations.

    Returns:
        Connection object if successful, otherwise exits the program.
    """
    conn = None  # Initialize variable
    try:
        # conn = psycopg.connect(
        #     host=os.getenv("DB_HOST"),
        #     user=os.getenv("DB_USERNAME"),
        #     password=os.getenv("DB_PASSWORD"),
        #     dbname=os.getenv("DB_NAME"),
        # )
        conn = psycopg.connect(
            host=soil_id.config.DB_HOST,
            user=soil_id.config.DB_USERNAME,
            password=soil_id.config.DB_PASSWORD,
            dbname=soil_id.config.DB_NAME,
        )
        logging.info("Database connection successful.")
        return conn
    except Exception as err:
        logging.error(f"Database connection failed: {err}")
        raise


# us, global
def save_model_output(
    plot_id, model_version, result_blob, soilIDRank_output_pd, mucompdata_cond_prob
):
    """
    Save the output of the model to the 'landpks_soil_model' table.
    """
    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO landpks_soil_model
        (plot_id, model_version, result_blob, soilIDRank_output_pd, mucompdata_cond_prob)
        VALUES (%s, %s, %s, %s, %s)
        """
        cur.execute(
            sql,
            (
                plot_id,
                model_version,
                result_blob,
                soilIDRank_output_pd,
                mucompdata_cond_prob,
            ),
        )
        conn.commit()

    except Exception as err:
        logging.error(err)
        conn.rollback()
        return None
    finally:
        conn.close()


# us, global
def save_rank_output(record_id, model_version, rank_blob):
    """
    Update the rank of the soil model in the 'landpks_soil_model' table.
    """
    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        sql = """UPDATE landpks_soil_model
                 SET soilrank = %s
                 WHERE ID = %s AND model_version = %s"""
        cur.execute(sql, (rank_blob, record_id, model_version))
        conn.commit()

    except Exception as err:
        logging.error(err)
        conn.rollback()
        return None
    finally:
        conn.close()


# us, global
def load_model_output(plot_id):
    """
    Load model output based on plot ID and model version.
    """
    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()
        model_version = 2
        sql = """SELECT ID, result_blob, soilIDRank_output_pd, mucompdata_cond_prob
                  FROM  landpks_soil_model
                  WHERE plot_id = %s AND model_version = %s
                  ORDER BY ID DESC LIMIT 1"""
        cur.execute(sql, plot_id, model_version)
        results = cur.fetchall()
        for row in results:
            model_run = [row[0], row[1], row[2], row[3]]
        return model_run
    except Exception as err:
        logging.error(err)
        return None
    finally:
        conn.close()


# global only
def save_soilgrids_output(plot_id, model_version, soilgrids_blob):
    """
    Save the output of the soil grids to the 'landpks_soilgrids_model' table.
    """
    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        sql = """
        INSERT INTO landpks_soilgrids_model
        (plot_id, model_version, soilgrids_blob)
        VALUES (%s, %s, %s)
        """
        cur.execute(sql, (plot_id, model_version, soilgrids_blob))
        conn.commit()

    except Exception as err:
        logging.error(err)
        conn.rollback()
        return None
    finally:
        conn.close()


# global (via extract_WISE_data)
def get_WISE30sec_data(MUGLB_NEW_Select):
    """
    Retrieve WISE 30 second data based on selected MUGLB_NEW values.
    """
    if not MUGLB_NEW_Select:  # Handle empty list case
        logging.warning("MUGLB_NEW_Select is empty. Returning empty DataFrame.")
        return pd.DataFrame()  # Return an empty DataFrame

    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        # Create placeholders for the SQL IN clause
        placeholders = ", ".join(["%s"] * len(MUGLB_NEW_Select))
        sql_query = f"""SELECT MUGLB_NEW, COMPID, id, MU_GLOBAL, NEWSUID, SCID, PROP, CLAF,
                               PRID, Layer, TopDep, BotDep, CFRAG, SDTO, STPC, CLPC, CECS,
                               PHAQ, ELCO, SU_name, FAO_SYS
                        FROM wise_soil_data
                        WHERE MUGLB_NEW IN ({placeholders})"""

        # Execute the query only if the list is non-empty
        cur.execute(sql_query, tuple(MUGLB_NEW_Select))
        results = cur.fetchall()

        # Convert the results to a pandas DataFrame
        data = pd.DataFrame(
            results,
            columns=[
                "MUGLB_NEW",
                "COMPID",
                "id",
                "MU_GLOBAL",
                "NEWSUID",
                "SCID",
                "PROP",
                "CLAF",
                "PRID",
                "Layer",
                "TopDep",
                "BotDep",
                "CFRAG",
                "SDTO",
                "STPC",
                "CLPC",
                "CECS",
                "PHAQ",
                "ELCO",
                "SU_name",
                "FAO_SYS",
            ],
        )

        return data

    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return None

    finally:
        if conn:
            conn.close()


# global
def get_WRB_descriptions(WRB_Comp_List):
    """
    Retrieve WRB descriptions based on provided WRB component list.
    """
    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        # Create placeholders for the SQL IN clause
        placeholders = ", ".join(["%s"] * len(WRB_Comp_List))
        sql = f"""SELECT WRB_tax, Description_en, Management_en, Description_es, Management_es,
                         Description_ks, Management_ks, Description_fr, Management_fr
                  FROM wrb_fao90_desc
                  WHERE WRB_tax IN ({placeholders})"""

        # Execute the query with the parameters
        cur.execute(sql, tuple(WRB_Comp_List))
        results = cur.fetchall()

        # Convert the results to a pandas DataFrame
        data = pd.DataFrame(
            results,
            columns=[
                "WRB_tax",
                "Description_en",
                "Management_en",
                "Description_es",
                "Management_es",
                "Description_ks",
                "Management_ks",
                "Description_fr",
                "Management_fr",
            ],
        )

        return data

    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return None

    finally:
        if conn:
            conn.close()


# global only
def getSG_descriptions(WRB_Comp_List):
    """
    Fetch WRB descriptions from a PostgreSQL database using wrb2006_to_fao90
    and wrb_fao90_desc tables. Returns a pandas DataFrame with columns:
    [WRB_tax, Description_en, Management_en, Description_es, ...]

    Args:
        WRB_Comp_List (list[str]): List of WRB_2006_Full values (e.g. ["Chernozem","Gleysol"]).

    Returns:
        pandas.DataFrame or None if an error occurs.
    """

    conn = None
    try:
        # 1. Get a connection to your datastore (replace with your actual function):
        conn = get_datastore_connection()

        def execute_query(query, params):
            with conn.cursor() as cur:
                # Execute the query with the parameters
                cur.execute(query, params)
                return cur.fetchall()

        # 2. Map WRB_2006_Full -> WRB_1984_Full using wrb2006_to_fao90
        #    Make sure we pass (tuple(WRB_Comp_List),) so psycopg2 can fill IN ('A','B','C')
        #    Example: WHERE lu.WRB_2006_Full IN ('Chernozem','Gleysol',...)
        sql1 = """
            SELECT lu.WRB_1984_Full
            FROM wrb2006_to_fao90 AS lu
            WHERE lu.WRB_2006_Full = ANY(%s)
        """
        names = execute_query(sql1, ([WRB_Comp_List],))

        # Flatten from [(x,), (y,)] => [x, y]
        WRB_Comp_List_mapped = [item for (item,) in names]

        if not WRB_Comp_List_mapped:
            # If no mapping found, return an empty DataFrame or None
            logging.warning("No mapped WRB_1984_Full names found for given WRB_2006_Full values.")
            return pd.DataFrame(
                columns=[
                    "WRB_tax",
                    "Description_en",
                    "Management_en",
                    "Description_es",
                    "Management_es",
                    "Description_ks",
                    "Management_ks",
                    "Description_fr",
                    "Management_fr",
                ]
            )

        # 3. Get descriptions from wrb_fao90_desc where WRB_tax IN ...
        sql2 = """
            SELECT WRB_tax,
                   Description_en,
                   Management_en,
                   Description_es,
                   Management_es,
                   Description_ks,
                   Management_ks,
                   Description_fr,
                   Management_fr
            FROM wrb_fao90_desc
            WHERE WRB_tax = ANY(%s)
        """
        results = execute_query(sql2, ([WRB_Comp_List_mapped],))

        # 4. Convert the raw query results to a DataFrame
        data = pd.DataFrame(
            results,
            columns=[
                "WRB_tax",
                "Description_en",
                "Management_en",
                "Description_es",
                "Management_es",
                "Description_ks",
                "Management_ks",
                "Description_fr",
                "Management_fr",
            ],
        )
        return data

    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return None

    finally:
        if conn:
            conn.close()
