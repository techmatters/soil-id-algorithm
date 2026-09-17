# Copyright © 2025 Technology Matters
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

import re

import soil_id
from soil_id.__version__ import __version__
from soil_id.rank_utils import MODEL_VERSION


def test_version_is_semver():
    # MAJOR.MINOR.PATCH — the client keys its cache-flush on MAJOR.MINOR.
    assert re.fullmatch(r"\d+\.\d+\.\d+", __version__), __version__


def test_version_is_reexported_from_package():
    # The backend reads soil_id.__version__; keep the re-export working.
    assert soil_id.__version__ == __version__


def test_model_metadata_tracks_major():
    # The legacy `model` metadata must equal "v{MAJOR}" so the two never diverge.
    major = __version__.split(".", 1)[0]
    assert MODEL_VERSION == f"v{major}"
