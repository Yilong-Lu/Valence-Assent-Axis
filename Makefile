.PHONY: install test verify figures notebooks audit qa

install:
	python -m pip install -e '.[analysis,test]'

test:
	pytest -q

verify:
	python analysis/python/verify_reported_results.py --output results/reproduction_check.json

figures:
	python -m analysis.figures.build_all

notebooks:
	python -m analysis.figures.sync_notebooks

audit:
	python analysis/python/release_audit.py

qa: audit verify test
