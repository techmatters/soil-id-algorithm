# Standard libraries
import logging
import sys

# local libraries
import config

# Third-party libraries
import MySQLdb
import pandas as pd


def get_datastore_connection():
    """
    Establish a connection to the datastore using app configurations.

    Returns:
        Connection object if successful, otherwise exits the program.
    """
    try:
        conn = MySQLdb.connect(
            host=config.DB_HOST,
            user=config.DB_USERNAME,
            passwd=config.DB_PASSWORD,
            db=config.DB_NAME,
        )
        return conn
    except Exception as err:
        logging.error(err)
        sys.exit(str(err))


# us, global
def save_model_output(
    plot_id, model_version, result_blob, soilIDRank_output_pd, mucompdata_cond_prob
):
    """
    Save the output of the model to the 'landpks_soil_model' table.
    """
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
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()
        placeholders = ", ".join(["%s"] * len(MUGLB_NEW_Select))
        sql = """SELECT MUGLB_NEW, COMPID, id, MU_GLOBAL, NEWSUID, SCID, PROP, CLAF,
                       PRID, Layer, TopDep, BotDep,  CFRAG,  SDTO,  STPC,  CLPC, CECS,
                       PHAQ, ELCO, SU_name, FAO_SYS
                  FROM  wise_soil_data
                  WHERE MUGLB_NEW IN (%s)"""
        cur.execute(sql, placeholders)
        results = cur.fetchall()
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
        logging.error(err)
        return None
    finally:
        conn.close()


# global
def get_WRB_descriptions(WRB_Comp_List):
    """
    Retrieve WRB descriptions based on provided WRB component list.
    """
    try:
        conn = get_datastore_connection()
        cur = conn.cursor()
        placeholders = ", ".join(["%s"] * len(WRB_Comp_List))
        sql = """SELECT WRB_tax, Description_en, Management_en, Description_es, Management_es,
                       Description_ks, Management_ks, Description_fr, Management_fr
                FROM wrb_fao90_desc
                WHERE WRB_tax IN %s"""
        cur.execute(sql, placeholders)
        results = cur.fetchall()
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
        logging.error(err)
        return None
    finally:
        conn.close()


# global only
def getSG_descriptions(WRB_Comp_List):
    try:
        conn = get_datastore_connection()

        # Execute a SQL query and return the results
        def execute_query(query, params):
            with conn.cursor() as cur:
                cur.execute(query, params)
                return cur.fetchall()

        # First SQL query
        sql1 = """SELECT lu.WRB_1984_Full
                  FROM wrb2006_to_fao90 AS lu
                  WHERE lu.WRB_2006_Full IN %s"""
        names = execute_query(sql1, (tuple(WRB_Comp_List),))
        WRB_Comp_List = [item for t in list(names) for item in t]

        # Second SQL query
        sql2 = """SELECT WRB_tax, Description_en, Management_en, Description_es, Management_es,
                  Description_ks, Management_ks, Description_fr, Management_fr
                  FROM wrb_fao90_desc
                  WHERE WRB_tax IN %s"""
        results = execute_query(sql2, (tuple(WRB_Comp_List),))

        # Convert results to DataFrame
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
        logging.error(err)
        return None

    finally:
        conn.close()
