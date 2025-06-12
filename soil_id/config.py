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
import os
import tempfile

from platformdirs import user_cache_dir

DATA_PATH = os.environ.get("DATA_PATH", "Data")

# Numpy seeding
RANDOM_SEED = os.environ.get("RANDOM_SEED", 19)

# Output
APP_NAME = os.environ.get("APP_NAME", "org.terraso.soilid")
TEMP_DIR = tempfile.TemporaryDirectory()
CACHE_DIR = user_cache_dir(APP_NAME)
OUTPUT_PATH = TEMP_DIR.name
SOIL_ID_RANK_PATH = f"{OUTPUT_PATH}/soil_id_rank.csv"
SOIL_ID_PROB_PATH = f"{OUTPUT_PATH}/soil_id_cond_prob.csv"
REQUESTS_CACHE_PATH = f"{CACHE_DIR}/requests_cache"

# Determines if in/out of US
US_AREA_PATH = f"{DATA_PATH}/SoilID_US_Areas.shp"

# US Soil ID
STATSGO_PATH = f"{DATA_PATH}/gsmsoilmu_a_us.shp"
MUNSELL_RGB_LAB_PATH = f"{DATA_PATH}/LandPKS_munsell_rgb_lab.csv"

# Global Soil ID
HWSD_PATH = f"{DATA_PATH}/HWSD_global_noWater_no_country.shp"
WISE_PATH = f"{DATA_PATH}/wise30sec_poly_simp_soil.shp"

# Database
DB_NAME = os.environ.get("DB_NAME", "terraso_backend")
DB_HOST = os.environ.get("DB_HOST")
DB_USERNAME = os.environ.get("DB_USERNAME")
DB_PASSWORD = os.environ.get("DB_PASSWORD")
