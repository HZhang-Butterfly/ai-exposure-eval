# Contributing to Job Evaluation Pipeline

Thanks for your interest in contributing. This document covers prerequisites, development setup, code structure, and the PR process.

---

## Prerequisites

- **Python 3.10+**
- An **OpenAI-compatible LLM server** (vLLM, Ollama, or similar) with a model loaded
- The **O\*NET Task Statements Excel file** (`Task Statements.xlsx`) — not committed to git due to size; download from [O\*NET Resource Center](https://www.onetcenter.org/database.html)

---

## Development Setup

```bash
git clone https://github.com/ButterflyLabs-org/job_eval.git
cd job_eval

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Install all dependencies
pip install -r requirements.txt

# Set up your local config
cp pipeline_config.example.json pipeline_config.json
# Edit pipeline_config.json — set server.ip and models.teacher / models.student

# Verify your setup with a single job
python run_pipeline.py --job "Accountants" --skip-eval   # generates config only, no API calls for eval
python run_pipeline.py --job "Accountants"               # full run (needs LLM server)
```

---

## Code Structure

| File | Responsibility |
|---|---|
| `run_pipeline.py` | CLI entrypoint; orchestrates the 3-step evaluation loop |
| `utils.py` | All shared functions: `call_llm`, `extract_json`, `load_config`, prompt helpers |
| `dashboard.py` | Streamlit dashboard; reads from `results/` |
| `prompt_templates/*.txt` | Prompt templates for each pipeline step |
| `pipeline_config.example.json` | Config schema and documentation |

### Key functions in `utils.py`

- **`load_config(path)`** — Loads `pipeline_config.json`, applies env var overrides, sets module-level globals (`TEACHER_API_URL`, `TEACHER_MODEL_NAME`, etc.)
- **`call_llm(...)`** — Sends a request to the configured API endpoint with retry logic and exponential backoff
- **`extract_json(text)`** — Robustly parses JSON from LLM output (handles `<think>` tags, markdown fences, truncated arrays)
- **`strip_thinking(text)`** — Removes chain-of-thought content before returning the final answer

### How prompt templates work

Templates live in `prompt_templates/` as plain text files with `{placeholder}` substitution:

```
# prompt_templates/step3_grading.txt
You are a {judge_role}. Grade the following submission...
--- STUDENT SUBMISSION ---
{student_answer}
```

`render_prompt_template(template_str, **kwargs)` replaces all `{key}` placeholders. Unknown placeholders are silently removed (no `KeyError`).

To override a template for a specific job, create `step3_grading_{job_id}.txt` — it takes precedence over the generic template.

---

## Configuration Priority

When adding features that read configuration values, always respect this priority order:

```
CLI argument  >  environment variable  >  pipeline_config.json  >  hardcoded default
```

- CLI args are applied in `run_pipeline.py::main()` after `utils.load_config()`
- Env vars are applied inside `utils.load_config()`
- Hardcoded defaults in `utils.py` should be generic (`"localhost"`, `"default"`) not environment-specific

---

## Making Changes

### Adding a new evaluation dimension

1. Update `FIXED_RUBRIC` and `FIXED_DIMENSION_LABELS` in `run_pipeline.py`
2. Update `prompt_templates/step3_grading.txt` to match
3. Re-run a job to verify the new dimension appears in results

### Changing grading behavior

Edit `prompt_templates/step3_grading.txt`. The key levers are:
- Score anchor descriptions (what 1/2/3/4/5 mean)
- The CRITIQUE → SCORE instruction flow
- The `{additional_checks}` placeholder (filled from `job_config.grading_notes`)

### Adding support for a new model API

If the API isn't OpenAI-compatible, add a new `call_llm_*` variant in `utils.py` and make it selectable via a `--api-style` CLI flag.

---

## Pull Request Process

1. Fork the repo and create a feature branch: `git checkout -b feat/my-change`
2. Make your changes with clear, focused commits
3. Test with at least one job: `python run_pipeline.py --job "Accountants"`
4. Open a PR with a description of **what** changed and **why**
5. PRs that modify prompts should include before/after score comparisons on at least one job

---

## Common Issues

**`ConnectionError` when running the pipeline**
→ Your server IP or port is wrong. Check `pipeline_config.json` or run with `--server-ip`.

**`ModuleNotFoundError: No module named 'openpyxl'`**
→ Run `pip install -r requirements.txt` — openpyxl is needed to read the Excel file.

**JSON parse errors in Step 1 or Step 3**
→ Usually means `max_tokens` is too low and the output was truncated. Increase `step1_generate` or `step3_grading` in `pipeline_config.json`.

**All tasks score 5/5 (self-evaluation bias)**
→ Known limitation when teacher and judge are the same model. The `critique-then-score` prompt in `step3_grading.txt` mitigates this. A different judge model would eliminate it.
