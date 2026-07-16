#!/usr/bin/env bash
set -euo pipefail

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

# ---------------------------------------------------------------------------
# Run one or more Makefile targets inside a GDAL-capable container, against the
# pinned soil-id-db image, on machines where the native env can't run the tests
# (GDAL doesn't build under uv/pip on macOS).
#
# The Makefile's test targets call this automatically when the local interpreter
# doesn't have pytest + the *pinned* GDAL version (see the NATIVE_OK guard in the
# Makefile), so `make test_unit`, `make test_update_unit_snapshots`, etc. work
# identically on macOS and CI. You normally don't invoke it directly.
#
# It:
#   1. Builds the CI-faithful runner image (scripts/snapshot-runner.Dockerfile),
#      layer-cached on the requirements files.
#   2. Starts the pinned soil-id-db (docker-compose.yml `db.image` — the one CI
#      uses, NOT your local :latest) on a private throwaway network with no host
#      port published (avoids colliding with a running terraso-backend on 5432).
#   3. Runs `make <targets> NATIVE=1` in the runner with the repo bind-mounted
#      read-write (so snapshots land in your working tree). NATIVE=1 makes the
#      in-container make run pytest directly instead of recursing back here.
#   4. Tears the DB + network down (including the anonymous data volume).
#
# Usage:
#   scripts/run_in_container.sh test_unit
#   scripts/run_in_container.sh test_update_unit_snapshots test_unit
#   REBUILD=1 scripts/run_in_container.sh test_unit   # force runner-image rebuild
# ---------------------------------------------------------------------------

[ "$#" -ge 1 ] || { echo "usage: $0 <make-target> [make-target...]" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

RUNNER_IMAGE="${RUNNER_IMAGE:-soil-id-snapshot-runner:latest}"
RUNNER_DOCKERFILE="scripts/snapshot-runner.Dockerfile"
PLATFORM="${PLATFORM:-linux/amd64}"

# The pinned DB and runner run on a private, throwaway network and talk by
# container name. We deliberately avoid `docker compose up` / the compose host
# port (5432), which collides with a running terraso-backend Postgres.
NETWORK="${NETWORK:-soil-id-snapshot-net}"
DB_CONTAINER="${DB_CONTAINER:-soil-id-snapshot-db}"
# Pinned image read straight from docker-compose.yml so it stays in lockstep
# with the pin CI uses (do not point at your local :latest).
DB_IMAGE="$(awk '/^[[:space:]]*image:/{print $2; exit}' docker-compose.yml)"

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31merror: %s\033[0m\n' "$*" >&2; exit 1; }

command -v docker >/dev/null || die "docker not found on PATH"

# Data/ must be present (US path reads shapefiles + Munsell CSV).
if [ ! -f Data/gsmsoilmu_a_us.shp ] || [ ! -f Data/LandPKS_munsell_rgb_lab.csv ]; then
  die "Data/ is missing required files. Run 'make download_soil_data' first."
fi

# --- Build the runner image (cheap when the deps layer is cached) --------------
if [ -n "${REBUILD:-}" ] || ! docker image inspect "$RUNNER_IMAGE" >/dev/null 2>&1; then
  log "Building runner image $RUNNER_IMAGE (mirrors CI: Ubuntu + ubuntugis + Python 3.13)"
  docker build --platform "$PLATFORM" -f "$RUNNER_DOCKERFILE" -t "$RUNNER_IMAGE" .
else
  log "Reusing runner image $RUNNER_IMAGE (set REBUILD=1 to force a rebuild)"
fi

# --- Start the pinned soil-id-db (private network, no host port) ----------------
cleanup() {
  # -v removes the anonymous data volume too; without it each run leaks ~3.7GB.
  docker rm -fv "$DB_CONTAINER" >/dev/null 2>&1 || true
  docker network rm "$NETWORK" >/dev/null 2>&1 || true
}
trap cleanup EXIT

# Fresh network + DB each run so a stale container can't serve wrong data.
docker rm -fv "$DB_CONTAINER" >/dev/null 2>&1 || true
docker network inspect "$NETWORK" >/dev/null 2>&1 || docker network create "$NETWORK" >/dev/null

log "Starting pinned soil-id-db ($DB_IMAGE)"
docker run -d --name "$DB_CONTAINER" --platform "$PLATFORM" \
  --network "$NETWORK" \
  -e POSTGRES_USER=postgres -e POSTGRES_PASSWORD=postgres -e POSTGRES_DB=soil_id \
  "$DB_IMAGE" >/dev/null

log "Waiting for the DB to accept connections"
for i in $(seq 1 60); do
  if docker exec "$DB_CONTAINER" pg_isready -U postgres -d soil_id >/dev/null 2>&1; then
    echo "  db ready"
    break
  fi
  [ "$i" -eq 60 ] && die "DB did not become ready in time"
  sleep 2
done

# --- Run the requested make target(s) in the runner -----------------------------
# NATIVE=1 so the in-container make runs pytest directly (no recursion). DB_* is
# what soil_id/config.py reads; the runner reaches the DB by name on NETWORK.
log "Running in container: make $* NATIVE=1"
docker run --rm \
  --platform "$PLATFORM" \
  --network "$NETWORK" \
  -v "$REPO_ROOT:/src" -w /src \
  -e DB_HOST="$DB_CONTAINER" -e DB_PORT=5432 -e DB_NAME=soil_id \
  -e DB_USERNAME=postgres -e DB_PASSWORD=postgres \
  "$RUNNER_IMAGE" make "$@" NATIVE=1
