format:
	ruff format soil_id

install:
	uv pip install -r requirements.txt

install-dev:
	uv pip install -r requirements-dev.txt

lint:
	ruff check soil_id

lock:
	CUSTOM_COMPILE_COMMAND="make lock" uv pip compile --upgrade --generate-hashes requirements/base.in -o requirements.txt

lock-dev:
	CUSTOM_COMPILE_COMMAND="make lock-dev" uv pip compile --upgrade --generate-hashes requirements/dev.in -o requirements-dev.txt

build:
	echo "Building TK..."

check_rebuild:
	./scripts/rebuild.sh

clean:
	@find . -name *.pyc -delete
	@find . -name __pycache__ -delete

test: clean check_rebuild
	if [ -z "$(PATTERN)" ]; then \
		$(DC_RUN_CMD) pytest soil_id; \
	else \
		$(DC_RUN_CMD) pytest soil_id -k $(PATTERN); \
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

generate_bulk_test_results:
	python -m soil_id.tests.us.generate_bulk_test_results

process_bulk_test_results:
	python -m soil_id.tests.us.process_bulk_test_results $(RESULTS_FILE)

# Donwload Munsell CSV, SHX, SHP, SBX, SBN, PRJ, DBF
download-soil-data:
	mkdir -p Data
	cd Data; \
	gdown 1tN23iVe6X1fcomcfveVp4w3Pwd0HJuTe; \
	gdown 1WUa9e3vTWPi6G8h4OI3CBUZP5y7tf1Li; \
	gdown 1l9MxC0xENGmI_NmGlBY74EtlD6SZid_a; \
	gdown 1asGnnqe0zI2v8xuOszlsNmZkOSl7cJ2n; \
	gdown 185Qjb9pJJn4AzOissiTz283tINrDqgI0; \
	gdown 1P3xl1YRlfcMjfO_4PM39tkrrlL3hoLzv; \
	gdown 1K0GkqxhZiVUND6yfFmaI7tYanLktekyp \
