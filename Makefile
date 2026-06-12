.PHONY: install test test-gpu fmt lint bench-attn profile-attn smoke

install:
	python -m pip install -e ".[dev,profile]"

test:
	pytest

test-gpu:
	pytest -m gpu

smoke:
	python scripts/run_all_smoke_tests.py

lint:
	ruff check .

fmt:
	ruff format .

bench-attn:
	python benchmarks/bench_attention.py

profile-attn:
	python benchmarks/profile_attention.py --trace-dir traces
