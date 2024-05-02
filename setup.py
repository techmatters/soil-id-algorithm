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
