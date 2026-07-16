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

# Runner image for regenerating soil-id test snapshots off the machine's native
# environment (GDAL does not build under uv/pip on macOS). This mirrors the CI
# "test" job (.github/workflows/build.yml) as closely as practical so the
# snapshots it produces match what CI validates:
#   - Ubuntu 24.04 + ubuntugis-unstable  -> same libgdal as CI
#   - Python 3.13                         -> same interpreter series as CI (3.13.7)
#   - `make install` / `make install_dev` -> the exact pinned requirements
#
# Only requirements are baked in (layer-cached on the requirements files); the
# soil-id source and Data/ are bind-mounted at run time by regen_snapshots.sh.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH="/root/.local/bin:${PATH}"
# uv --system installs must target the 3.13 we register as default python3 below.
ENV UV_SYSTEM_PYTHON=1
ENV UV_BREAK_SYSTEM_PACKAGES=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      software-properties-common curl ca-certificates git make gcc g++ \
 && add-apt-repository -y ppa:ubuntugis/ubuntugis-unstable \
 && add-apt-repository -y ppa:deadsnakes/ppa \
 && apt-get update && apt-get install -y --no-install-recommends \
      libgdal-dev gdal-bin \
      python3.13 python3.13-dev python3.13-venv \
 && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.13 1 \
 && update-alternatives --install /usr/bin/python  python  /usr/bin/python3.13 1 \
 && rm -rf /var/lib/apt/lists/*

# uv (used by the Makefile install targets)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /src

# Install the pinned deps. Keyed on the requirements files so this layer is
# reused until they change. `DC_ENV=ci` makes the Makefile pass uv `--system`.
COPY requirements.txt requirements-dev.txt Makefile ./
COPY requirements/ requirements/
RUN make install DC_ENV=ci && make install_dev DC_ENV=ci

CMD ["bash"]
