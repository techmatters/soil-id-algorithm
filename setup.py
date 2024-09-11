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
from setuptools import setup
from pathlib import Path

this_directory = Path(__file__).parent

version = {}
with open(os.path.join(this_directory, "soil_id", "__version__.py")) as f:
    exec(f.read(), version)

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

short_description = "Soil identification listing and ranking"

setup(
    name="soil_id",
    version=version["__version__"],
    description=short_description,
    long_description=long_description,
    url="https://github.com/techmatters/soil-id-algorithm",
    packages=["soil_id"],
    python_requires=">=3.12",
    license="License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
    author="Paul Schreiber",
    author_email="paul@techmatters.org",
    classifiers=[
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Programming Language :: Python :: 3.12",
    ],
)
