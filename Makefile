.PHONY: test marts
test:
	python -m pytest -q
marts:
	python -m src.spendsense.data.build_marts