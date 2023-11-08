format:
	isort --atomic python_code
	black python_code

lint:
	flake8 python_code && isort -c python_code && black --check python_code
