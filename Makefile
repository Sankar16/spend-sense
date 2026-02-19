.PHONY: test marts dashboard
test:
	python -m pytest -q
marts:
	python -m src.spendsense.data.build_marts
dashboard: 
	python -m src.spendsense.eda.dashboard_preview