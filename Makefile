ifeq ($(DC_ENV),ci)
	UV_FLAGS = "--system"
endif

install:
	uv pip install -r requirements.txt $(UV_FLAGS)

install-dev:
	uv pip install -r requirements-dev.txt $(UV_FLAGS)

setup-git-hooks:
	@pre-commit install

lint:
	ruff check soil_id

format:
	ruff format soil_id

lock:
	CUSTOM_COMPILE_COMMAND="make lock" uv pip compile --upgrade --generate-hashes requirements/base.in -o requirements.txt

lock-package:
	CUSTOM_COMPILE_COMMAND="make lock" uv pip compile --upgrade-package $(PACKAGE) --generate-hashes --emit-build-options requirements/base.in requirements/deploy.in -o requirements.txt

lock-dev:
	CUSTOM_COMPILE_COMMAND="make lock-dev" uv pip compile --upgrade --generate-hashes requirements/dev.in -o requirements-dev.txt

lock-dev-package:
	CUSTOM_COMPILE_COMMAND="make lock-dev" uv pip compile --upgrade-package $(PACKAGE) --generate-hashes requirements/dev.in -o requirements-dev.txt

build:
	echo "Building TK..."

check_rebuild:
	./scripts/rebuild.sh

clean:
	@find . -name *.pyc -delete
	@find . -name __pycache__ -delete

test: clean check_rebuild
	if [ -z "$(PATTERN)" ]; then \
		$(DC_RUN_CMD) pytest soil_id -vv; \
	else \
		$(DC_RUN_CMD) pytest soil_id -vv -k $(PATTERN); \
	fi

test_update_snapshots: clean check_rebuild
	if [ -z "$(PATTERN)" ]; then \
		$(DC_RUN_CMD) pytest soil_id --snapshot-update; \
	else \
		$(DC_RUN_CMD) pytest soil_id --snapshot-update -k $(PATTERN); \
	fi

test-verbose:
	pytest soil_id --capture=no

test-profile:
	pytest soil_id --profile

test-graphs: test-profile graphs

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

generate_bulk_test_results_legacy:
	python -m soil_id.tests.legacy.generate_bulk_test_results

process_bulk_test_results_legacy:
	python -m soil_id.tests.legacy.process_bulk_test_results $(RESULTS_FILE)

# Donwload Munsell CSV, SHX, SHP, SBX, SBN, PRJ, DBF
# 1tN23iVe6X1fcomcfveVp4w3Pwd0HJuTe: LandPKS_munsell_rgb_lab.csv
# 1WUa9e3vTWPi6G8h4OI3CBUZP5y7tf1Li: gsmsoilmu_a_us.shx
# 1l9MxC0xENGmI_NmGlBY74EtlD6SZid_a: gsmsoilmu_a_us.shp
# 1asGnnqe0zI2v8xuOszlsNmZkOSl7cJ2n: gsmsoilmu_a_us.sbx
# 185Qjb9pJJn4AzOissiTz283tINrDqgI0: gsmsoilmu_a_us.sbn
# 1P3xl1YRlfcMjfO_4PM39tkrrlL3hoLzv: gsmsoilmu_a_us.prj
# 1K0GkqxhZiVUND6yfFmaI7tYanLktekyp: gsmsoilmu_a_us.dbf
# 1z7foFFHv_mTsuxMYnfOQRvXT5LKYlYFN: SoilID_US_Areas.shz
download-soil-data:
	mkdir -p Data
	cd Data; \
	gdown 1tN23iVe6X1fcomcfveVp4w3Pwd0HJuTe; \
	gdown 1WUa9e3vTWPi6G8h4OI3CBUZP5y7tf1Li; \
	gdown 1l9MxC0xENGmI_NmGlBY74EtlD6SZid_a; \
	gdown 1asGnnqe0zI2v8xuOszlsNmZkOSl7cJ2n; \
	gdown 185Qjb9pJJn4AzOissiTz283tINrDqgI0; \
	gdown 1P3xl1YRlfcMjfO_4PM39tkrrlL3hoLzv; \
	gdown 1K0GkqxhZiVUND6yfFmaI7tYanLktekyp; \
	gdown 1z7foFFHv_mTsuxMYnfOQRvXT5LKYlYFN \

DATABASE_DUMP_FILE ?= Data/soil_id_db.dump
DOCKER_IMAGE_TAG ?= ghcr.io/techmatters/soil-id-db:latest
build_docker_image:
	@echo "Building to tag $(DOCKER_IMAGE_TAG)"
	docker build \
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

dump_soil_id_db:
	pg_dump --format=custom $(DATABASE_URL)  -t hwsd2_segment -t hwsd2_data -t landpks_munsell_rgb_lab -t normdist1 -t normdist2 -t wise_soil_data -t wrb2006_to_fao90 -t wrb_fao90_desc -f $(DATABASE_DUMP_FILE)

restore_soil_id_db:
	pg_restore --dbname=$(DATABASE_URL) --single-transaction --clean --if-exists --no-owner $(DATABASE_DUMP_FILE)
	psql $(DATABASE_URL) -c "CLUSTER hwsd2_segment USING hwsd2_segment_shape_idx;"
	psql $(DATABASE_URL) -c "ANALYZE;"
