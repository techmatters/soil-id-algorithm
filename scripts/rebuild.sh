#!/bin/bash
set -e

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

build_hash()
{
    if [ -z $(which md5sum) ]; then
        # macOS support
        md5 -r requirements/base.in requirements/dev.in requirements.txt requirements-dev.txt > .rebuild
    else
        # Linux support
        md5sum requirements/base.in requirements/dev.in requirements.txt requirements-dev.txt > .rebuild
    fi
}

check_hash()
{
    if [ -z $(which md5sum) ]; then
        # macOS support
        md5 -r requirements/base.in requirements/dev.in requirements.txt requirements-dev.txt | diff .rebuild - > /dev/null 2>&1
        echo $?
    else
        # Linux support
        md5sum --quiet -c .rebuild > /dev/null 2>&1
        echo $?
    fi
}

echo "Checking if we need a docker images rebuild before running this command"
if [ $(check_hash) != 0 ]; then
    make build
    build_hash
fi
