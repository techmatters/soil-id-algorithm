format:
	isort --atomic python_code API
	black python_code API

lint:
	flake8 python_code API && isort -c python_code API && black --check python_code API
