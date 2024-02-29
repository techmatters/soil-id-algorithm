pip-tools: ${VIRTUAL_ENV}/scripts/pip-sync

format:
	isort --atomic python_code API
	black python_code API

install:
	pip install -r requirements.txt

install-dev:
	pip install -r requirements-dev.txt

lint:
	flake8 python_code API && isort -c python_code API && black --check python_code API

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
		$(DC_RUN_CMD) pytest python_code; \
	else \
		$(DC_RUN_CMD) pytest python_code -k $(PATTERN); \
	fi

${VIRTUAL_ENV}/scripts/pip-sync:
	pip install pip-tools
