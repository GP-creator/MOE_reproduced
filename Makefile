# MoE reproduction. Scripts add the repo root to sys.path themselves, so they run as `$(PY) scripts/xxx.py`.
PY ?= .venv/bin/python

.DEFAULT_GOAL := help
.PHONY: help setup test data smoke small main sweep ablate profile trace placement report all dashboard clean-results

help: ## show targets
	@grep -E '^[a-z-]+:.*## ' $(MAKEFILE_LIST) | awk -F':.*## ' '{printf "  %-14s %s\n", $$1, $$2}'

setup: ## create venv and install torch (cu128) then requirements
	python3 -m venv .venv
	.venv/bin/pip install torch==2.11.0 --index-url https://download.pytorch.org/whl/cu128
	.venv/bin/pip install -r requirements.txt

test: ## run unit tests
	$(PY) -m pytest -q tests

data: ## download + tokenize + cache data
	$(PY) -m moe.data --config configs/main.yaml

# End-to-end CPU check of the whole pipeline (tiny model, minutes).
smoke: ## end-to-end pipeline on CPU with configs/smoke.yaml
	$(PY) scripts/run_main.py --config configs/smoke.yaml
	$(PY) scripts/sweep_capacity.py --config configs/smoke.yaml
	$(PY) scripts/ablate_experts.py --config configs/smoke.yaml
	$(PY) scripts/profile_layer.py --config configs/smoke.yaml
	$(PY) scripts/route_trace.py --config configs/smoke.yaml
	$(PY) scripts/placement_sim.py --config configs/smoke.yaml
	$(PY) scripts/make_report.py --config configs/smoke.yaml

small: ## main experiment at small scale
	$(PY) scripts/run_main.py --config configs/small.yaml

# E1 exactly as it was run: all 5 variants with seed 1337, then a 2nd seed (1338) for dense/switch/deepseek.
# --skip-existing skips runs that already completed, so a rerun only does what is missing.
main: ## E1: 5 variants (seed 1337) + dense/switch/deepseek (seed 1338)
	$(PY) scripts/run_main.py --config configs/main.yaml --include-optional --seeds 1337 --skip-existing
	$(PY) scripts/run_main.py --config configs/main.yaml --variants dense switch deepseek --seeds 1337 1338 --skip-existing

sweep: ## capacity-factor sweep
	$(PY) scripts/sweep_capacity.py --config configs/main.yaml

ablate: ## expert ablation
	$(PY) scripts/ablate_experts.py --config configs/main.yaml

profile: ## single-layer MoE profiling
	$(PY) scripts/profile_layer.py --config configs/main.yaml

trace: ## routing trace
	$(PY) scripts/route_trace.py --config configs/main.yaml

placement: ## expert placement simulation
	$(PY) scripts/placement_sim.py --config configs/main.yaml

report: ## build the report
	$(PY) scripts/make_report.py --config configs/main.yaml

# Full chain in dependency order: placement reads the routing traces, so trace runs before it.
all: ## whole main pipeline: main sweep ablate trace placement profile report
	$(MAKE) main sweep ablate trace placement profile report

# headless + no usage stats: otherwise Streamlit blocks on a first-run e-mail prompt. Open the URL from the Windows browser.
dashboard: ## launch Streamlit dashboard on http://localhost:8501
	$(PY) -m streamlit run dashboard/app.py --server.headless true --browser.gatherUsageStats false

clean-results: ## delete everything under results/ except results/sample
	find results -mindepth 1 -maxdepth 1 ! -name sample -exec rm -rf {} +
