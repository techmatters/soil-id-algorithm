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

# local libraries
import soil_id.config


def get_datastore_connection():
    """
    Establish a connection to the datastore using app configurations.

    Returns:
        Connection object if successful, otherwise exits the program.
    """
    try:
        return psycopg.connect(
            host=soil_id.config.DB_HOST,
            user=soil_id.config.DB_USERNAME,
            password=soil_id.config.DB_PASSWORD,
            dbname=soil_id.config.DB_NAME,
        )
    except Exception as err:
        logging.error(f"Database connection failed: {err}")
        raise


def get_hwsd2_profile_data(connection, hwsd2_mu_select):
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
        with connection.cursor() as cur:
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
                    "hwsd2",
                    "compid",
                    "id",
                    "wise30s_smu_id",
                    "sequence",
                    "share",
                    "fao90",
                    "layer",
                    "topdep",
                    "botdep",
                    "coarse",
                    "sand",
                    "silt",
                    "clay",
                    "cec_soil",
                    "ph_water",
                    "elec_cond",
                    "fao90_name",
                ],
            )
            return data
    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return pd.DataFrame()


def extract_hwsd2_data(connection, lon, lat, buffer_dist):
    """
    Fetches HWSD soil data from a PostGIS table within a given buffer around a point,
    performing distance and intersection calculations directly on geographic coordinates.

    Parameters:
        lon (float): Longitude of the problem point.
        lat (float): Latitude of the problem point.
        buffer_dist (int): Buffer distance in meters.

    Returns:
        DataFrame: Merged data from hwsd2_segment and hwsd2_data.
    """
    point = f"ST_SetSRID(ST_Point({lon}, {lat}), 4326)::geography"
    main_query = f"""
        SELECT
            hwsd2_id as hwsd2,
            MIN(ST_Distance(
                shape,
                {point}
            )) AS distance,
            BOOL_OR(ST_Intersects(shape, {point})) AS pt_intersect
        FROM hwsd2_segment
        WHERE ST_DWithin(shape, {point}, {buffer_dist})
        GROUP BY hwsd2_id;
    """

    # Use GeoPandas to execute the main query and load results into a GeoDataFrame.
    hwsd = pd.read_sql_query(main_query, connection)

    # Get the list of hwsd2 identifiers.
    hwsd2_mu_select = hwsd["hwsd2"].tolist()

    # Call get_hwsd2_profile_data using the same connection.
    hwsd_data = get_hwsd2_profile_data(connection, hwsd2_mu_select)

    # Merge the two datasets.
    merged = pd.merge(hwsd_data, hwsd, on="hwsd2", how="left").drop_duplicates()
    return merged


# Function to fetch data from a PostgreSQL table
def query_db(connection, query):
    try:
        with connection.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()

            return rows

    except Exception as err:
        logging.error(f"Error querying PostgreSQL: {err}")
        return None


def fetch_normdist(connection):
    return query_db(connection, "SELECT * FROM normdist2 ORDER BY id ASC;")


# global only
def getSG_descriptions(connection, WRB_Comp_List):
    """
    Fetch WRB descriptions from a PostgreSQL database using wrb2006_to_fao90
    and wrb_fao90_desc tables. Returns a pandas DataFrame with columns:
    [WRB_tax, Description_en, Management_en, Description_es, ...]

    Args:
        WRB_Comp_List (list[str]): List of WRB_2006_Full values (e.g. ["Chernozem","Gleysol"]).

    Returns:
        pandas.DataFrame or None if an error occurs.
    """

    try:

        def execute_query(query, params):
            with connection.cursor() as cur:
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


def fetch_munsell_rgb_lab(connection):
    return pd.read_sql_query("SELECT * FROM landpks_munsell_rgb_lab;", connection)
