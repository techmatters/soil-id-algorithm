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

import geopandas as gpd
import pandas as pd

# Third-party libraries
import psycopg
from shapely import wkb
from shapely.geometry import Point

# local libraries
import soil_id.config

from .utils import get_target_utm_srid


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


def get_hwsd2_profile_data(conn, hwsd2_mu_select):
    """
    Retrieve HWSD v2 data based on selected hwsd2 (map unit) values.
    This version reuses an existing connection.
    
    Parameters:
        conn: A live database connection.
        hwsd2_mu_select (list): List of selected hwsd2 values.
    
    Returns:
        DataFrame: Data from hwsd2_data.
    """
    if not hwsd2_mu_select:
        logging.warning("HWSD2 map unit selection is empty. Returning empty DataFrame.")
        return pd.DataFrame()
    
    try:
        with conn.cursor() as cur:
            # Create placeholders for the SQL IN clause
            placeholders = ", ".join(["%s"] * len(hwsd2_mu_select))
            sql_query = f"""
                SELECT hwsd2_smu_id, compid, id, wise30s_smu_id, sequence, share, fao90,
                       layer, topdep, botdep, coarse, sand, silt, clay, cec_soil,
                       ph_water, elec_cond, fao90_name
                FROM hwsd2_data
                WHERE hwsd2_smu_id IN ({placeholders})
            """
            cur.execute(sql_query, tuple(hwsd2_mu_select))
            results = cur.fetchall()
            
            # Convert the results to a pandas DataFrame.
            data = pd.DataFrame(
                results,
                columns=[
                    "hwsd2", "compid", "id", "wise30s_smu_id", "sequence", "share", "fao90",
                    "layer", "topdep", "botdep", "coarse", "sand", "silt", "clay", "cec_soil",
                    "ph_water", "elec_cond", "fao90_name"
                ],
            )
            return data
    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return pd.DataFrame()

def extract_hwsd2_data(lon, lat, buffer_dist, table_name):
    """
    Fetches HWSD soil data from a PostGIS table within a given buffer around a point.
    
    Parameters:
        lon (float): Longitude of the problem point.
        lat (float): Latitude of the problem point.
        buffer_dist (int): Buffer distance in meters.
        table_name (str): Name of the PostGIS table (e.g., "hwsdv2").
    
    Returns:
        DataFrame: Merged data from hwsdv2 and hwsdv2_data.
    """
    # Determine target UTM SRID as an integer (e.g., 32642)
    target_srid = get_target_utm_srid(lat, lon)
    
    # Use a single connection for both queries.
    with get_datastore_connection() as conn:
        # Compute the buffer polygon (in WKT) around the problem point.
        buffer_query = """
            WITH buffer AS (
              SELECT ST_AsText(
                       ST_Transform(
                         ST_Buffer(
                           ST_Transform(
                             ST_SetSRID(ST_Point(%s, %s), 4326),
                             %s
                           ),
                           %s
                         ),
                         4326
                       )
                     ) AS wkt
            )
            SELECT wkt FROM buffer;
        """
        with conn.cursor() as cur:
            cur.execute(buffer_query, (lon, lat, target_srid, buffer_dist))
            buffer_wkt = cur.fetchone()[0]
            print("Buffer WKT:", buffer_wkt)
        
        # Build the main query that uses the computed buffer.
        main_query = f"""
            WITH valid_geom AS (
              SELECT
                hwsd2,
                ST_MakeValid(geom) AS geom
              FROM {table_name}
              WHERE geom && ST_GeomFromText('{buffer_wkt}', 4326)
                AND ST_Intersects(geom, ST_GeomFromText('{buffer_wkt}', 4326))
            )
            SELECT
              hwsd2,
              ST_AsEWKB(ST_Transform(geom, {target_srid})) AS geom,
              ST_Distance(
                ST_Transform(geom, {target_srid}),
                ST_Transform(ST_SetSRID(ST_Point({lon}, {lat}), 4326), {target_srid})
              ) AS distance,
              ST_Intersects(
                ST_Transform(geom, {target_srid}),
                ST_Transform(ST_SetSRID(ST_Point({lon}, {lat}), 4326), {target_srid})
              ) AS pt_intersect
            FROM valid_geom
            WHERE ST_Intersects(
                ST_Transform(geom, {target_srid}),
                ST_Transform(ST_SetSRID(ST_Point({lon}, {lat}), 4326), {target_srid})
            );
        """
        
        # Use GeoPandas to execute the main query and load results into a GeoDataFrame.
        hwsd = gpd.read_postgis(main_query, conn, geom_col='geom')
        print("Main query returned", len(hwsd), "rows.")
        
        # Remove the geometry column (if not needed) from this dataset.
        hwsd = hwsd.drop(columns=["geom"])
        
        # Get the list of hwsd2 identifiers.
        hwsd2_mu_select = hwsd["hwsd2"].tolist()
        
        # Call get_hwsd2_profile_data using the same connection.
        hwsd_data = get_hwsd2_profile_data(conn, hwsd2_mu_select)
        
        # Merge the two datasets.
        merged = pd.merge(hwsd_data, hwsd, on="hwsd2", how="left").drop_duplicates()
        return merged


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

        return rows

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
