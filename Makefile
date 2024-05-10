pip-tools: ${VIRTUAL_ENV}/scripts/pip-sync

format:
	isort --atomic soil_id API
	black soil_id API

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	flake8 soil_id API && isort -c soil_id API && black --check soil_id API

lock: pip-tools
	CUSTOM_COMPILE_COMMAND="make lock" pip-compile --upgrade --generate-hashes --resolver=backtracking --output-file requirements.txt requirements/base.in

lock-dev: pip-tools
	CUSTOM_COMPILE_COMMAND="make lock-dev" pip-compile --upgrade --generate-hashes --resolver=backtracking --output-file requirements-dev.txt requirements/dev.in

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

${VIRTUAL_ENV}/scripts/pip-sync:
	pip install pip-tools
