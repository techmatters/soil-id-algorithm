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
from shapely.geometry import Point
import geopandas as gpd

# local libraries
import soil_id.config
from .utils import (
    calculate_distances_and_intersections,
    create_circular_buffer,
)


def get_datastore_connection():
    """
    Establish a connection to the datastore using app configurations.

    Returns:
        Connection object if successful, otherwise exits the program.
    """
    conn = None
    try:
        conn = psycopg.connect(
            host=soil_id.config.DB_HOST,
            user=soil_id.config.DB_USERNAME,
            password=soil_id.config.DB_PASSWORD,
            dbname=soil_id.config.DB_NAME,
        )
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


def extract_WISE_data(lon, lat, table_name='hwsdv2', buffer_dist=10000):
    """
    Extracts WISE data by querying a PostGIS table using psycopg2.

    Args:
        lon (float): Longitude of the point.
        lat (float): Latitude of the point.
        table_name (str): Name of the PostGIS table.
        buffer_dist (int, optional): Buffer distance in meters. Default is 10,000m.

    Returns:
        DataFrame: Filtered WISE data from PostGIS.
    """
    
    # Create a buffer around point
    buffer = create_circular_buffer(lon, lat, buffer_dist=10000)

    # Convert buffer to WKT for SQL query
    buffer_wkt = buffer.wkt  # ✅ Use buffer.wkt directly

    # Construct SQL query to retrieve features **inside** the buffer
    query = f"""
        SELECT MUGLB_NEW, ST_AsEWKB(geom) AS geom
        FROM {table_name}
        WHERE ST_Intersects(geom, ST_GeomFromText('{buffer_wkt}', 4326));
    """
    try:
        # Establish connection
        conn = get_datastore_connection()
        cur = conn.cursor()

        # Execute query
        cur.execute(query)

        # Fetch results
        rows = cur.fetchall()

        # Convert results into a GeoDataFrame
        hwsd = gpd.GeoDataFrame(rows, columns=["hwsd2", "geom"], crs="EPSG:4326")

        # Ensure only unique map units are considered
        mu_geo = hwsd.drop_duplicates(subset="hwsd2")

        # Calculate distances and intersections
        mu_id_dist = calculate_distances_and_intersections(mu_geo, point_geo)
        mu_id_dist.loc[mu_id_dist["pt_intersect"], "dist_meters"] = 0
        mu_id_dist["distance"] = mu_id_dist.groupby("hwsd2")["dist_meters"].transform(min)
        mu_id_dist = mu_id_dist.nsmallest(2, "distance")

        # Merge results and remove duplicates
        hwsd = hwsd.drop(columns=["geom"])
        hwsd = pd.merge(mu_id_dist, hwsd, on="hwsd2", how="left").drop_duplicates()

        # Query WISE data based on MUGLB_NEW values
        MUGLB_NEW_Select = hwsd["hwsd2"].tolist()
        wise_data = get_WISE30sec_data(MUGLB_NEW_Select)
        wise_data = pd.merge(wise_data, mu_id_dist, on="MUGLB_NEW", how="left")

        return wise_data

    except Exception as e:
        print(f"Error querying PostGIS: {e}")
        return None

    finally:
        # Ensure the database connection is closed
        cur.close()
        conn.close()


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

# Function to fetch data from a PostgreSQL table
def fetch_table_from_db(table_name):
    conn = None
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()

        query = f"SELECT * FROM {table_name} ORDER BY id ASC;"
        cur.execute(query)
        rows = cur.fetchall()

        # Transpose rows to access them in the same format as the CSV
        return list(map(list, zip(*rows[1:])))  # Skip 'id' column, Transpose to get lists

    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return None

    finally:
        if conn:
            conn.close()


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
