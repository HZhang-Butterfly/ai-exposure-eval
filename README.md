# Job Evaluation Pipeline

Evaluate a **student model** (e.g. code or BI model) on job-relevant tasks. The pipeline uses O*NET job/task data and two LLM roles: a **teacher** (generates tasks and grades) and a **student** (the model under evaluation).

---

## Flow

1. **Job & task selection** — Load O*NET Excel, filter to IT jobs, user picks a job (or pass `--job`). An LLM selects **N** representative O*NET tasks for that job.
2. **Job config** — Another LLM call produces a job configuration (role, output structure, grading notes). Saved to `job_config_interactive.json`.
3. **3-step evaluation**
   - **Step 1 (Teacher):** Generate N evaluation task objects (prompt, context, criteria).
   - **Step 2 (Student):** The model under evaluation answers each task.
   - **Step 3 (Teacher):** Grade each answer (score + reason).

Results are written to `{job_id}_results_auto.json`.

---

## How to run

```bash
pip install -r requirements.txt
python qwen_main_pipeline.py
```

- **Non-interactive:** `python qwen_main_pipeline.py --job "Software Developers" --tasks 5`
- **Config only (no eval):** add `--skip-eval`
- **Custom Excel:** `--excel path/to/file.xlsx`
- **Custom config output:** `--output my_config.json`

**Data:** Default input is `Task Statements.xlsx` in the project root (O*NET format: columns `Title`, `Task`, `O*NET-SOC Code`).

---

## Files this pipeline uses

| File | Role |
|------|------|
| `qwen_main_pipeline.py` | Main script: job selection, config generation, 3-step eval |
| `utils.py` | Shared helpers: LLM calls, config load, prompt template load/render |
| `pipeline_config.json` | Server IP/ports, teacher & student model names, `max_tokens`, timeouts, output suffix |
| `Task Statements.xlsx` | O*NET task data (default; override with `--excel`) |
| `prompt_templates/task_selector.txt` | LLM selects N representative O*NET tasks |
| `prompt_templates/generate_configuration.txt` | LLM generates job config |
| `prompt_templates/step1_teacher_tasks.txt` | Step 1: teacher generates task objects |
| `prompt_templates/step2_student_system.txt` | Step 2: student model system prompt |
| `prompt_templates/step3_grading.txt` | Step 3: teacher grades answer (score + reason) |

Editing the `prompt_templates/*.txt` files changes behavior without touching code. Optional per-job overrides: `step1_teacher_tasks_{job_id}.txt` (same for step2/step3) are tried before the generic template.

---

## Config (`pipeline_config.json`)

- **server:** `ip`, `teacher_port`, `student_port`
- **models:** `teacher`, `student` (model names for the API)
- **task.num_tasks:** default number of tasks (e.g. 5)
- **max_tokens:** per step (`step1_generate`, `step2_student`, `step3_grading`) — keep `step1_generate` within the model’s context limit (e.g. 4096 − input tokens)
- **timeouts**, **delays**, **output.file_suffix_auto**

APIs are OpenAI-compatible `POST /v1/chat/completions`.

---

## Outputs

- **Job config:** `job_config_interactive.json` (overwritten each run unless you set `--output`).
- **Eval results:** `{job_id}_results_auto.json` — meta (job_title, models, average_score, total_time_seconds) + per-task scores and grade reasons.
