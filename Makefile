.PHONY: help setup run run-all resume patch-all dashboard clean

# ─── Default target ────────────────────────────────────────────────────────
help:
	@echo ""
	@echo "Job Evaluation Pipeline — available commands:"
	@echo ""
	@echo "  make setup        Install dependencies and copy config template"
	@echo "  make run          Run pipeline interactively (pick a job from menu)"
	@echo "  make run-all      Run all 114 jobs (saves to results/)"
	@echo "  make resume       Run all jobs, skip already-completed ones"
	@echo "  make patch-all    Re-grade and fill gaps in existing results"
	@echo "  make dashboard    Launch the Streamlit results dashboard"
	@echo "  make clean        Remove __pycache__ and .pyc files"
	@echo ""
	@echo "Override server/model inline (highest priority):"
	@echo "  make run JOB='Accountants' SERVER_IP=10.0.0.1 PORT=8000 MODEL=llama3"
	@echo ""

# ─── Setup ─────────────────────────────────────────────────────────────────
setup:
	pip install -r requirements.txt
	@if [ ! -f pipeline_config.json ]; then \
		cp pipeline_config.example.json pipeline_config.json; \
		echo ""; \
		echo "  Created pipeline_config.json from template."; \
		echo "  Edit it and set your server IP and model name before running."; \
		echo ""; \
	else \
		echo "  pipeline_config.json already exists — skipping copy."; \
	fi

# ─── Run pipeline ──────────────────────────────────────────────────────────
# Usage:  make run JOB="Accountants"
#         make run JOB="Accountants" SERVER_IP=10.0.0.1 PORT=8000 MODEL=llama3
JOB        ?=
SERVER_IP  ?=
PORT       ?=
MODEL      ?=

_server_flags = \
	$(if $(SERVER_IP), --server-ip $(SERVER_IP)) \
	$(if $(PORT),      --port $(PORT)) \
	$(if $(MODEL),     --teacher-model $(MODEL) --student-model $(MODEL))

run:
	python run_pipeline.py $(if $(JOB),--job "$(JOB)") $(_server_flags)

run-all:
	python run_pipeline.py --batch-all $(_server_flags)

resume:
	python run_pipeline.py --batch-all --resume $(_server_flags)

patch-all:
	python run_pipeline.py --patch-all $(_server_flags)

# ─── Dashboard ─────────────────────────────────────────────────────────────
dashboard:
	streamlit run dashboard.py

# ─── Cleanup ───────────────────────────────────────────────────────────────
clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	@echo "  Cleaned up compiled Python files."
