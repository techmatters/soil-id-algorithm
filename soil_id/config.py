import os
import tempfile

from dotenv import load_dotenv

load_dotenv()

DATA_PATH = os.environ.get("DATA_PATH", "Data")

# Output
TEMP_DIR = tempfile.TemporaryDirectory(delete=False)
OUTPUT_PATH = TEMP_DIR.name
SOIL_ID_RANK_PATH = f"{OUTPUT_PATH}/soil_id_rank.csv"
SOIL_ID_PROB_PATH = f"{OUTPUT_PATH}/soil_id_cond_prob.csv"

# Shared data
HWSD_PATH = f"{DATA_PATH}/HWSD_global_noWater_no_country.shp"
US_AREA_PATH = f"{DATA_PATH}/SoilID_US_Areas.shp"

# US Soil ID
STATSGO_PATH = f"{DATA_PATH}/gsmsoilmu_a_us.shp"
MUNSELL_RGB_LAB_PATH = f"{DATA_PATH}/LandPKS_munsell_rgb_lab.csv"

# Global Soil ID
WISE_PATH = f"{DATA_PATH}/wise30sec_poly_simp_soil.shp"
NORM_DIST_1_PATH = f"{DATA_PATH}/NormDist1.csv"
NORM_DIST_2_PATH = f"{DATA_PATH}/NormDist2.csv"
