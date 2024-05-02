pip-tools: ${VIRTUAL_ENV}/scripts/pip-sync

format:
	isort --atomic soil_id API
	black soil_id API

install:
	pip install -r requirements.txt

install-no-deps:
	pip install -r requirements.txt --no-deps
	pip install GDAL==`gdal-config --version`

install-dev:
	pip install -r requirements-dev.txt

lint:
	flake8 soil_id API && isort -c soil_id API && black --check soil_id API

lock: pip-tools
	CUSTOM_COMPILE_COMMAND="make lock" pip-compile --upgrade --generate-hashes --resolver=backtracking --output-file requirements.txt requirements/base.in requirements/deploy.in

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

${VIRTUAL_ENV}/scripts/pip-sync:
	pip install pip-tools
