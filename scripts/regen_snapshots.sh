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
# Regenerate the unit-test *output* snapshots (soil_id/tests/{us,global}/
# __snapshots__/test_soil_location[...].json) reproducibly, then verify them.
#
# The snapshot command is just `make test_update_unit_snapshots`; the fiddly
# part is the environment (GDAL, which won't build on macOS, + the exact pinned
# soil-id-db image CI uses). scripts/run_in_container.sh supplies that. Here we:
#   1. Regenerate + verify in one container/DB session:
#      `make test_update_unit_snapshots test_unit` — update, then re-run without
#      --snapshot-update. The verify pass MUST pass clean, or the snapshots are
#      non-deterministic and untrustworthy.
#   2. Show `git status` for the snapshot dirs.
#
# Usage:
#   scripts/regen_snapshots.sh            # regenerate + verify (default)
#   scripts/regen_snapshots.sh --no-verify
#   REBUILD=1 scripts/regen_snapshots.sh  # force rebuild of the runner image
# ---------------------------------------------------------------------------

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

VERIFY=1
for arg in "$@"; do
  case "$arg" in
    --no-verify) VERIFY=0 ;;
    -h|--help) sed -n '19,40p' "$0"; exit 0 ;;
    *) echo "unknown arg: $arg" >&2; exit 2 ;;
  esac
done

log() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }

# Update, then (unless --no-verify) re-run the same tests without
# --snapshot-update to prove determinism — both in one DB session.
if [ "$VERIFY" -eq 1 ]; then
  log "Regenerating unit snapshots + verifying determinism"
  if ./scripts/run_in_container.sh test_update_unit_snapshots test_unit; then
    echo "  verification passed — snapshots are deterministic"
  else
    printf '\033[1;31m%s\033[0m\n' \
      "verification FAILED: snapshots differ on a second run (non-determinism). Do not commit." >&2
    exit 1
  fi
else
  log "Regenerating unit snapshots (no verify)"
  ./scripts/run_in_container.sh test_update_unit_snapshots
fi

log "Snapshot changes:"
git -c color.status=always status --short \
  soil_id/tests/us/__snapshots__ soil_id/tests/global/__snapshots__ || true
echo
echo "Review the diff, then commit. Snapshots were generated against the pinned"
echo "image in docker-compose.yml, so they should match CI."
