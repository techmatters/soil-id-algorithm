# Copyright 2015 Google Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
This file contains all of the configuration values for the application.
Update this file with the values for your specific Google Cloud project.
You can create and manage projects at https://console.developers.google.com
"""

import os


class Config(object):
    ENV = "production"
    # The secret key is used by Flask to encrypt session cookies.
    SECRET_KEY = "secret"
    # Location of the data file
    DATA_BACKEND = "/mnt/c/LandPKS_API3_SoilID-master/global/"
    # TNC_DATA = '/opt/app/soilid/data/Habitat.gdb'
    # TNC_DATA_V2 = '/opt/app/soilid/data/Habitat_V2.gdb'
    # CLIMATE_DATA_DIR = '/opt/app/soilid/data/'
    # Google Cloud Project ID. This can be found on the 'Overview' page at
    # https://console.developers.google.com
    PROJECT_ID = "landpks-api-soilid"
    URL_PREFIX = "/api"
    PORT = 5000
    HOST = "127.0.0.1"
    # CloudSQL & SQLAlchemy configuration
    # Replace the following values the respective values of your Cloud SQL
    # instance.
    CLOUDSQL_IP = "127.0.0.1"
    CLOUDSQL_USER = "root"
    CLOUDSQL_PASSWORD = "root"
    CLOUDSQL_DATABASE = "apex"

    IMAGE_BUCKET = "images.landpotential.org"


class Development(Config):
    ENV = "development"
    PORT = 5000
    HOST = "127.0.0.1"
    DATA_BACKEND = "/mnt/c/LandPKS_API3_SoilID-master/global/"
    # TNC_DATA = '/Users/ciarankenny/projects/Data/Habitat.gdb'
    # TNC_DATA_V2 = '/Users/ciarankenny/projects/Data/Habitat_V2.gdb'
    # CLIMATE_DATA_DIR = '/Users/ciarankenny/projects/Data/CLIMATE/'
    # DATA_BACKEND = '/Users/ckenny-a/Documents/LandPKS/FAN/model-run'

    CLOUDSQL_CONNECTION_NAME = ""
    CLOUDSQL_IP = "127.0.0.1"
    CLOUDSQL_USER = "root"
    CLOUDSQL_PASSWORD = "root"
    CLOUDSQL_DATABASE = "apex"


class Production(Config):
    pass
