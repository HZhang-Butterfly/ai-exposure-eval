# Job Evaluation Pipeline

Quantify **AI exposure risk** across 114 real-world occupations using a three-step LLM evaluation pipeline grounded in [O\*NET](https://www.onetonline.org/) task data.

Works with **any OpenAI-compatible model server** — vLLM, Ollama, OpenAI API, etc.

---

## Architecture

```
Task Statements.xlsx  (O*NET task data)
         │
         ▼
┌─────────────────────────────────────────────────────┐
│  For each occupation (114 total)                    │
│                                                     │
│  ① Config    LLM generates job evaluation config    │
│      │        (cached in configs/ after first run)  │
│      ▼                                              │
│  ② Step 1   Teacher LLM → evaluation tasks         │
│      ↓        (5 tasks per batch from O*NET data)   │
│  ③ Step 2   Student LLM → answers each task        │
│      ↓                                              │
│  ④ Step 3   Judge LLM  → scores 5 dimensions       │
│                           (1–5 per task)            │
└─────────────────────────────────────────────────────┘
         │
         ▼
results/{job_id}_results_auto.json
         │
         ▼
streamlit run dashboard.py   →   interactive analytics
```

**Five scoring dimensions** (applied uniformly across all occupations):

| # | Dimension | What it measures |
|---|---|---|
| 1 | Correctness | Does the answer solve the task? |
| 2 | Completeness | Are all parts fully addressed? |
| 3 | Best Practices | Does it follow professional standards? |
| 4 | Domain Accuracy | Are domain-specific facts and terms correct? |
| 5 | Clarity | Is the output well-structured and readable? |

The **Exposure Score** (1–5) is the mean across all dimensions and all tasks for a given occupation. Higher = more of that occupation's work can be done well by the model.

---

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/ButterflyLabs-org/job_eval.git
cd job_eval
pip install -r requirements.txt

# 2. Configure your model server
cp pipeline_config.example.json pipeline_config.json
# Edit pipeline_config.json — set "server.ip" and "models.teacher" / "models.student"

# 3. Run a single job to verify the setup
python run_pipeline.py --job "Accountants"

# 4. View results in the dashboard
streamlit run dashboard.py
```

Or use the Makefile:

```bash
make setup       # installs deps + copies config template
make run JOB="Accountants"
make dashboard
```

---

## Configuration

Configuration is resolved in priority order (highest first):

```
CLI arguments  >  environment variables (.env)  >  pipeline_config.json  >  built-in defaults
```

### pipeline_config.json

Copy `pipeline_config.example.json` → `pipeline_config.json` (never committed to git):

```json
{
  "server": {
    "ip": "10.0.0.1",
    "teacher_port": 8000,
    "student_port": 8000
  },
  "models": {
    "teacher": "meta-llama/Llama-3-8B-Instruct",
    "student": "meta-llama/Llama-3-8B-Instruct"
  }
}
```

### Environment variables

Copy `.env.example` → `.env` (also never committed):

| Variable | Description |
|---|---|
| `EVAL_SERVER_IP` | Server IP or hostname |
| `EVAL_SERVER_PORT` | Port (teacher and student share) |
| `EVAL_TEACHER_MODEL` | Teacher / Judge model name |
| `EVAL_STUDENT_MODEL` | Student model name |
| `EVAL_TEACHER_PORT` | Teacher-only port (optional) |
| `EVAL_STUDENT_PORT` | Student-only port (optional) |

### CLI arguments

| Argument | Description |
|---|---|
| `--server-ip` | Override server IP |
| `--port` | Override port |
| `--teacher-model` | Override teacher model |
| `--student-model` | Override student model |
| `--pipeline-config` | Use a different config file |

Example — run without editing any file:
```bash
EVAL_SERVER_IP=10.0.0.1 EVAL_TEACHER_MODEL=YOUR_MODEL_NAME python run_pipeline.py --job "Budget"
# or
python run_pipeline.py --server-ip 10.0.0.1 --teacher-model YOUR_MODEL_NAME --job "Budget"
```

---

## Using Your Own Model

The pipeline calls a standard OpenAI-compatible `/v1/chat/completions` endpoint. Any server that implements this API works.

### vLLM (recommended for local models)

```bash
pip install vllm
vllm serve meta-llama/Llama-3-8B-Instruct --port 8000
```

Then set `"ip": "localhost"` and `"teacher_port": 8000` in `pipeline_config.json`.

### Ollama

```bash
ollama pull llama3
ollama serve   # runs on localhost:11434
```

Set port to `11434`. Note: Ollama's model names use short names like `llama3`, not HuggingFace paths.

### OpenAI API

Set `"ip": "api.openai.com"`, port `443`, and use HTTPS — you'll need to modify `utils.py` to use `https://` and add your API key as an `Authorization` header. A future update will support this natively.

---

## Running the Pipeline

```bash
# Single job (interactive picker if no --job)
python run_pipeline.py
python run_pipeline.py --job "Software Developers"

# All 114 jobs
python run_pipeline.py --batch-all

# Resume interrupted run (skip completed jobs)
python run_pipeline.py --batch-all --resume

# Fill gaps in existing results (re-grade missing tasks)
python run_pipeline.py --patch-all
python run_pipeline.py --patch --job "Accountants"

# Generate job config only, skip evaluation
python run_pipeline.py --job "Accountants" --skip-eval

# Use a pre-saved job config
python run_pipeline.py --config job_config_interactive.json
```

---

## Output Format

Each job produces `results/{job_id}_results_auto.json`:

```json
{
  "meta": {
    "job_title": "Accountants and Auditors",
    "onet_code": "13-2011.00",
    "total_onet_tasks": 30,
    "evaluated_tasks": 30,
    "grading_dimensions": ["Correctness", "Completeness", "Best Practices", "Domain Accuracy", "Clarity"],
    "exposure_score": {
      "overall": 4.04,
      "dimensions": {
        "Correctness": 3.60,
        "Completeness": 4.23,
        "Best Practices": 3.73,
        "Domain Accuracy": 3.77,
        "Clarity": 4.87
      },
      "evaluated_tasks": 30,
      "scale_max": 5
    }
  },
  "tasks": [
    {
      "task_id": "gen_task_01",
      "onet_source": "Original O*NET task statement",
      "task_type": "fraud_detection",
      "user_prompt": "Full task description given to the student model",
      "reference_context": "Data / context provided alongside the task",
      "student_answer": "Model's complete answer",
      "dimension_scores": [4, 5, 4, 5, 5],
      "score": 4.6,
      "grade_reason": "Judge's critique-based explanation"
    }
  ]
}
```

---

## Project Structure

```
job_eval/
├── run_pipeline.py              # Main pipeline entry point
├── utils.py                     # Shared helpers: LLM calls, config, prompt loading
├── dashboard.py                 # Streamlit analytics dashboard
│
├── pipeline_config.example.json # Config template (copy → pipeline_config.json)
├── .env.example                 # Env var template (copy → .env)
├── requirements.txt             # All Python dependencies
├── Makefile                     # Convenience commands
│
├── prompt_templates/            # Prompt templates (edit to change behavior)
│   ├── generate_configuration.txt
│   ├── step1_teacher_tasks.txt
│   ├── step2_student_system.txt
│   └── step3_grading.txt
│
├── digital_jobs.csv             # 114 pre-filtered digital occupations
├── Task Statements.xlsx         # O*NET task data (source of truth)
│
├── configs/                     # Cached LLM-generated job configs (auto-created)
└── results/                     # Pipeline results (auto-created)
```

Prompt templates are plain `.txt` files with `{placeholder}` substitution. Edit them to change task generation style, grading strictness, or scoring criteria — no code changes needed.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
