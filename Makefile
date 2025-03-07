.PHONY: setup test run clean

# Default target to run the application
all: setup run

# Setup virtual environment and install dependencies
setup:
	python3 -m venv venv
	. venv/bin/activate && pip install -r requirements.txt
	. venv/bin/activate && pip install -e .
	touch .requirements_installed

# Run the application
run:
	./run.sh

# Run tests
test:
	./test.sh

# Clean the environment (remove venv and cached files)
clean:
	rm -rf venv
	rm -rf data/vector_store
	rm -rf data/models
	rm -rf logs
	rm -f .requirements_installed
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete