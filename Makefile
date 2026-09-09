ifeq ($(DC_ENV),ci)
	UV_FLAGS = "--system"
endif

PYTHON ?= python3

# The test targets below need pytest + GDAL + the pinned soil-id-db. They run
# natively only when THIS interpreter ($(PYTHON)) has pytest AND the *pinned*
# GDAL version (so results match CI and the pinned image); otherwise they
# transparently re-run themselves in a GDAL-capable container against the pinned
# soil-id-db image (see scripts/run_in_container.sh), so the same `make test*`
# command works on macOS and on CI alike.
#
# The version gate matters: snapshots are GDAL-version-sensitive, so a native
# Homebrew GDAL of a *different* version must NOT take the native path — it would
# produce snapshots that disagree with CI. It also naturally routes macOS to the
# container, since the interpreter with GDAL (Homebrew) lacks pytest and the
# venv with pytest lacks GDAL. Set NATIVE=1 to force native (also set inside the
# runner container, where the versions do match, to prevent infinite recursion).
GDAL_PIN := $(shell sed -n 's/^gdal==\([0-9.]*\).*/\1/p' requirements.txt)
ifeq ($(NATIVE),1)
	NATIVE_OK := 1
else
	NATIVE_OK := $(shell $(PYTHON) -c 'import pytest, osgeo.gdal as g; raise SystemExit(g.__version__ != "$(GDAL_PIN)")' >/dev/null 2>&1 && echo 1)
endif

install:
	uv pip install -r requirements.txt $(UV_FLAGS)

install_dev:
	uv pip install -r requirements-dev.txt $(UV_FLAGS)

setup_git_hooks:
	@pre-commit install

lint:
	ruff check soil_id
	ruff format soil_id --diff

format:
	ruff check soil_id --fix
	ruff format soil_id

lock:
	CUSTOM_COMPILE_COMMAND="make lock" uv pip compile --upgrade --generate-hashes requirements/base.in -o requirements.txt

lock_package:
	CUSTOM_COMPILE_COMMAND="make lock" uv pip compile --upgrade-package $(PACKAGE) --generate-hashes --emit-build-options requirements/base.in requirements/deploy.in -o requirements.txt

lock_dev:
	CUSTOM_COMPILE_COMMAND="make lock-dev" uv pip compile --upgrade --generate-hashes requirements/dev.in -o requirements-dev.txt

lock_dev_package:
	CUSTOM_COMPILE_COMMAND="make lock-dev" uv pip compile --upgrade-package $(PACKAGE) --generate-hashes requirements/dev.in -o requirements-dev.txt

clean:
	@find . -name *.pyc -delete
	@find . -name __pycache__ -delete

# run the standard test suite (unit + integration, no api_snapshots)
test:
	@if [ "$(NATIVE_OK)" != "1" ]; then exec ./scripts/run_in_container.sh test; fi; \
	if [ -z "$(PATTERN)" ]; then \
		$(PYTHON) -m pytest soil_id -m "not api_snapshot"; \
	else \
		$(PYTHON) -m pytest soil_id -m "not api_snapshot" -k "$(PATTERN)"; \
	fi

# All tests except api_snapshot and integration (no live external APIs)
test_unit:
	@if [ "$(NATIVE_OK)" != "1" ]; then exec ./scripts/run_in_container.sh test_unit; fi; \
	$(PYTHON) -m pytest soil_id -m "not api_snapshot and not integration"

# update the unit test snapshots (but not the API snapshots)
test_update_unit_snapshots:
	@if [ "$(NATIVE_OK)" != "1" ]; then exec ./scripts/run_in_container.sh test_update_unit_snapshots; fi; \
	$(PYTHON) -m pytest soil_id -m "not api_snapshot and not integration" --snapshot-update

# Regenerate the unit-test output snapshots reproducibly in a GDAL-capable
# container, against the pinned soil-id-db image CI uses. Wraps
# test_update_unit_snapshots for machines where GDAL won't build natively
# (e.g. macOS). See scripts/regen_snapshots.sh.
regen_snapshots:
	./scripts/regen_snapshots.sh

# Integration smoke tests only (full live API run, no output validation)
test_integration:
	pytest soil_id -m integration

# API response snapshot tests only (compares live API responses to stored snapshots)
test_api_snapshot:
	pytest soil_id -m api_snapshot

# Refresh stored API response snapshots from live APIs
test_update_api_snapshots:
	pytest soil_id -m api_snapshot --snapshot-update

test_verbose:
	pytest soil_id --capture=no

test_profile:
	pytest soil_id --profile

test_graphs: test-profile graphs

graphs:
	# gprof2dot -f pstats  prof/combined.prof | dot -Tsvg -o prof/combined.svg
	# flameprof prof/combined.prof > prof/combined_flame.svg
	gprof2dot -f pstats  prof/test_soil_location.prof | dot -Tsvg -o prof/test_soil_location.svg
	flameprof prof/test_soil_location.prof > prof/test_soil_location_flame.svg

generate_bulk_test_results_us:
	python -m soil_id.tests.us.generate_bulk_test_results

process_bulk_test_results_us:
	python -m soil_id.tests.us.process_bulk_test_results $(RESULTS_FILE)

generate_bulk_test_results_global:
	python -m soil_id.tests.global.generate_bulk_test_results

process_bulk_test_results_global:
	python -m soil_id.tests.global.process_bulk_test_results $(RESULTS_FILE)

# Paired before/after accuracy comparison of two bulk-result files (the reliable
# way to validate an algorithm change against ground truth). Usage:
#   make compare_bulk_test_results_us BEFORE=before.jsonl AFTER=after.jsonl
compare_bulk_test_results_us:
	python -m soil_id.tests.compare_bulk_test_results $(BEFORE) $(AFTER) --dataset us

compare_bulk_test_results_global:
	python -m soil_id.tests.compare_bulk_test_results $(BEFORE) $(AFTER) --dataset global

generate_bulk_test_results_legacy:
	python -m soil_id.tests.legacy.generate_bulk_test_results

process_bulk_test_results_legacy:
	python -m soil_id.tests.legacy.process_bulk_test_results $(RESULTS_FILE)

# Donwload Munsell CSV, SHX, SHP, SBX, SBN, PRJ, DBF
# List of files:
#   1HpJK681LAbCkJE_Wqyb1ayP4Gb46zULD: LandPKS_munsell_rgb_lab.csv
#   1HWg2-PEvdeltf72XMeM_pc9TJWGJvyAV: gsmsoilmu_a_us.shx
#   1MoR5E3InvVNcMERvyMBK47iN3Imgx3TJ: gsmsoilmu_a_us.shp
#   1qBC624vGRIV5yht1itAPpifI9bU4ix_q: gsmsoilmu_a_us.sbx
#   1krQl5g5-UjmAuiBtv6aE1k49GHbvt8hQ: gsmsoilmu_a_us.sbn
#   1Vm0_0uw_QSszj6OEinmF5QyGNAJZp2zt: gsmsoilmu_a_us.prj
#   1v4I9edYf0ybls-vnfvmmLSPdDipmNSBS: gsmsoilmu_a_us.dbf
#   1PYk-aWaWFyABzP0-stgTaFI0Vp8rV5IM: SoilID_US_Areas.shz
download_soil_data:
	mkdir -p Data
	cd Data; \
	gdown 1HpJK681LAbCkJE_Wqyb1ayP4Gb46zULD; \
	gdown 1HWg2-PEvdeltf72XMeM_pc9TJWGJvyAV; \
	gdown 1MoR5E3InvVNcMERvyMBK47iN3Imgx3TJ; \
	gdown 1qBC624vGRIV5yht1itAPpifI9bU4ix_q; \
	gdown 1krQl5g5-UjmAuiBtv6aE1k49GHbvt8hQ; \
	gdown 1Vm0_0uw_QSszj6OEinmF5QyGNAJZp2zt; \
	gdown 1v4I9edYf0ybls-vnfvmmLSPdDipmNSBS; \
	gdown 1PYk-aWaWFyABzP0-stgTaFI0Vp8rV5IM \

DATABASE_DUMP_FILE ?= Data/soil_id_db.dump
DOCKER_IMAGE_TAG ?= ghcr.io/techmatters/soil-id-db:latest
# Build linux/amd64 explicitly: the postgis:16-3.5 base has no arm64 manifest,
# and CI/prod run the image on amd64 regardless.
build_docker_image:
	@echo "Building to tag $(DOCKER_IMAGE_TAG)"
	docker build \
	  --platform linux/amd64 \
	  --build-arg DATABASE_DUMP_FILE=$(DATABASE_DUMP_FILE) \
	  -t $(DOCKER_IMAGE_TAG) \
	  .

push_docker_image:
	@echo "Pushing tag $(DOCKER_IMAGE_TAG). Make sure to provide a versioned tag in addition to updating latest!"
	docker push $(DOCKER_IMAGE_TAG)

start_db:
	docker compose up -d

stop_db:
	docker compose down

connect_db:
	docker compose exec db psql -U postgres -d soil_id

# Container running the local soil-id-db (PostgreSQL 16). Override if your
# compose project names it differently.
SOIL_ID_DB_CONTAINER ?= terraso-backend-soil-id-db-1

# Dump from *inside* the PG16 container so the archive is always restorable by
# the PG16 image. A host pg_dump newer than 16 (e.g. 17/18) writes an archive
# the image's pg_restore 16 cannot read ("unsupported version in file header"),
# which silently breaks build_docker_image.
dump_soil_id_db:
	docker exec $(SOIL_ID_DB_CONTAINER) pg_dump --format=custom -U postgres -d soil_id \
	  -t hwsd2_segment -t hwsd2_data -t landpks_munsell_rgb_lab -t normdist1 -t normdist2 \
	  -t wise_soil_data -t wrb2006_to_fao90 -t wrb_fao90_desc -f /tmp/soil_id_db.dump
	docker cp $(SOIL_ID_DB_CONTAINER):/tmp/soil_id_db.dump $(DATABASE_DUMP_FILE)

restore_soil_id_db:
	pg_restore --dbname=$(DATABASE_URL) --single-transaction --clean --if-exists --no-owner $(DATABASE_DUMP_FILE)
	psql $(DATABASE_URL) -c "CLUSTER hwsd2_segment USING hwsd2_segment_shape_idx;"
	psql $(DATABASE_URL) -c "ANALYZE;"
