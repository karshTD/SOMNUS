.PHONY: help install test bench bench-full run sleep migrate health mcp dashboard clean

help:
	@echo "make install    install dependencies"
	@echo "make test       run the test suite (offline, no credentials)"
	@echo "make bench      run the forgetting benchmark, 1 seed"
	@echo "make bench-full run the benchmark across 12 seeds with variance"
	@echo "make run        run the agent (offline unless COCKROACH_DB_URL is set)"
	@echo "make migrate    apply the CockroachDB schema"
	@echo "make health     check CockroachDB + AWS connectivity"
	@echo "make dashboard  launch the Streamlit dashboard"
	@echo "make mcp        run the MCP server on stdio"

install:
	pip install -r requirements.txt

test:
	SOMNUS_OFFLINE=1 python -m pytest tests/ -q

bench:
	SOMNUS_OFFLINE=1 python -m eval.benchmark

bench-full:
	SOMNUS_OFFLINE=1 python -m eval.benchmark --seed 1 --seeds 12

run:
	python main.py

sleep:
	python main.py --sleep-once

migrate:
	python main.py --migrate

health:
	python main.py --health

mcp:
	python -m mcp_server

dashboard:
	streamlit run dashboard.py

clean:
	rm -rf data/*.json skills/*.md __pycache__ .pytest_cache
	find . -name '__pycache__' -type d -exec rm -rf {} + 2>/dev/null || true
