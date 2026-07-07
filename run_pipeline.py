#!/usr/bin/env python3
"""
Job Evaluation Pipeline — main entry point.

Flow (per job):
  1. Load ALL O*NET tasks for the job from Task Statements.xlsx
  2. Generate (or load cached) job config via LLM  →  configs/<onet_slug>_config.json
  3. Override rubric with the fixed 5 evaluation dimensions
  4. Process all tasks in batches of 5:
       Step 1 (Teacher)  → generate evaluation task objects from each batch
       Step 2 (Student)  → student model answers each task
       Step 3 (Judge)    → grade each answer on 5 dimensions (1–5)
  5. Compute per-dimension averages and overall exposure score
  6. Save to results/{job_id}_results_auto.json

Usage:
  python run_pipeline.py --batch-all                    # run all 114 jobs
  python run_pipeline.py --batch-all --resume           # skip already-completed jobs
  python run_pipeline.py --job "Software Dev"           # run a single job by title fragment
  python run_pipeline.py                                # interactive job picker
  python run_pipeline.py --config saved.json            # use saved config, skip config LLM call
  python run_pipeline.py --skip-eval --job "X"          # generate config only, no evaluation

Model / server can be set via (highest priority first):
  CLI args:   --server-ip, --port, --teacher-model, --student-model
  Env vars:   EVAL_SERVER_IP, EVAL_SERVER_PORT, EVAL_TEACHER_MODEL, EVAL_STUDENT_MODEL
  JSON file:  pipeline_config.json  (copy from pipeline_config.example.json)
"""

import json
import os
import re
import time
import argparse
import requests
import pandas as pd
from typing import List, Dict, Tuple
import sys

import utils
from utils import (
    call_llm, extract_json, strip_thinking, load_onet_excel,
    load_prompt_template, render_prompt_template,
)


# ─────────────────────────────────────────────
# FIXED RUBRIC — 5 dimensions applied to all jobs
# ─────────────────────────────────────────────

FIXED_RUBRIC = [
    "1. Correctness: Does the solution correctly address all task requirements?",
    "2. Completeness: Are all parts of the task fully addressed with sufficient detail?",
    "3. Best Practices: Does the solution follow professional and industry standards?",
    "4. Domain Accuracy: Are domain-specific concepts, facts, calculations, and terminology used correctly?",
    "5. Clarity: Is the output well-structured, clear, and easy to understand?",
]
FIXED_DIMENSION_LABELS = ["Correctness", "Completeness", "Best Practices", "Domain Accuracy", "Clarity"]

CONFIGS_DIR = "configs"


# SECTION 1: EVALUATION PIPELINE  (Steps 1–3)

def _normalize_job_config(job_config):  # -> Dict
    """Accept a single config dict or a list of one dict (e.g. from saved JSON)."""
    if isinstance(job_config, list) and len(job_config) == 1:
        return job_config[0]
    if isinstance(job_config, dict):
        return job_config
    raise TypeError("job_config must be a dict or a list of one dict, got " + type(job_config).__name__)


def generate_teacher_prompt(job_config: Dict, num_tasks: int) -> str:
    """
    Build the Step 1 prompt for the teacher model.
    Fills the step1_teacher_tasks.txt template with:
      - role, title, num_tasks, onet_list, requirements, output_structure
    """
    job_config = _normalize_job_config(job_config)
    title = job_config["job_title"]
    role = job_config["role_description"]
    onet_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(job_config["onet_tasks"]))
    reqs_raw = job_config["specific_requirement"]
    requirements = "\n".join(reqs_raw) if isinstance(reqs_raw, list) else reqs_raw
    output_structure = json.dumps(job_config["output_structure"], indent=2)

    template = load_prompt_template("step1_teacher_tasks", job_config.get("job_id"))
    if not template:
        raise FileNotFoundError("prompt_templates/step1_teacher_tasks.txt not found.")

    return render_prompt_template(
        template,
        role=role, title=title, num_tasks=num_tasks,
        onet_list=onet_list, requirements=requirements,
        output_structure=output_structure,
    )


def step_1_generate_tasks(job_config: Dict, num_tasks: int = None) -> List[Dict]:
    """
    Step 1: Teacher model generates N evaluation task objects.
    Each task contains: task_id, task_type, onet_source,
    user_prompt, reference_context, evaluation_criteria.

    Returns a list of task dicts, or [] on failure.
    """
    num_tasks = num_tasks if num_tasks is not None else utils.CONFIG.get("task", {}).get("batch_size", 5)
    max_tok   = utils.CONFIG.get("max_tokens", {}).get("step1_generate", 2500)
    timeout   = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 120)

    print(f"\n--- STEP 1: Teacher generates {num_tasks} tasks for [{job_config['job_title']}] ---")
    prompt   = generate_teacher_prompt(job_config, num_tasks)
    response = call_llm(
        utils.TEACHER_API_URL, utils.TEACHER_MODEL_NAME,
        [
            {"role": "system", "content": "Output ONLY a valid JSON array. No explanation, no markdown."},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tok, timeout=timeout, enable_thinking=False,
    )
    if not response:
        print("  Step 1: no response from teacher API — skipping this batch.")
        return []
    try:
        # Strip any thinking prefix before JSON parsing
        cleaned = strip_thinking(response)
        tasks = extract_json(cleaned)
        if isinstance(tasks, dict):
            tasks = [tasks]
        print(f"  Generated {len(tasks)} tasks:")
        for i, t in enumerate(tasks, 1):
            print(f"    [{i}] {t.get('task_id','?')} | {t.get('task_type','?')}")
        return tasks
    except Exception:
        print("  JSON parse failed. Raw output (first 500 chars):")
        print(response[:500])
        return []


def step_2_student_inference(tasks: List[Dict], job_config: Dict = None) -> List[Dict]:
    """
    Step 2: Student model answers each task.
    Builds a system prompt from step2_student_system.txt and sends
    (user_prompt + reference_context) as the user message.

    Returns task list with 'student_answer' appended to each item.
    Returns [] if the student API is unreachable.
    """
    print(f"\n--- STEP 2: Student [{utils.STUDENT_MODEL_NAME}] answers tasks ---")

    conn_timeout = utils.CONFIG.get("timeouts", {}).get("connection_check_seconds", 5)
    base_url     = utils.STUDENT_API_URL.replace("/v1/chat/completions", "")
    print(f"  Checking connection to {utils.STUDENT_API_URL}...")
    try:
        requests.get(base_url, timeout=conn_timeout)
    except Exception as e:
        print(f"  Student API unreachable: {e}")
        return []

    jc        = job_config or {}
    template  = load_prompt_template("step2_student_system", jc.get("job_id"))
    raw_sys   = render_prompt_template(template or "", job_title=jc.get("job_title", "")) if template else f"/no_think\nYou are a {jc.get('job_title', 'professional')}. Complete the task directly and professionally. Provide only the final work product."
    # Ensure /no_think is always present to suppress thinking mode
    if "/no_think" not in raw_sys:
        raw_sys = "/no_think\n" + raw_sys
    max_tok   = utils.CONFIG.get("max_tokens", {}).get("step2_student", 4000)
    temp      = utils.CONFIG.get("temperature", {}).get("step2_student", 0.2)
    timeout   = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 300)
    delay     = utils.CONFIG.get("delays", {}).get("between_tasks_seconds", 3)
    results   = []

    for i, item in enumerate(tasks):
        ref_ctx = item.get("reference_context") or ""
        if not isinstance(ref_ctx, str):
            ref_ctx = json.dumps(ref_ctx, indent=2, ensure_ascii=False)
        user_content = str(item.get("user_prompt") or "") + "\n\n--- Context ---\n" + ref_ctx
        print(f"  Task {i+1}/{len(tasks)}: {item.get('task_id', '?')}...")
        resp = call_llm(
            utils.STUDENT_API_URL, utils.STUDENT_MODEL_NAME,
            [{"role": "system", "content": raw_sys}, {"role": "user", "content": user_content}],
            max_tokens=max_tok, temperature=temp, timeout=timeout, enable_thinking=False,
        )
        item["student_answer"] = resp.strip() if resp else "(no response)"
        results.append(item)
        if delay and i < len(tasks) - 1:
            time.sleep(delay)

    return results


def _get_rubric(item: Dict, job_config: Dict) -> Tuple[List[str], str]:
    """
    Get rubric for grading this task. Prefer:
      1. job_config["grading_rubric_template"]  — shared rubric for all tasks in this job
      2. item["evaluation_criteria"]            — per-task criteria
    Returns (list of criterion strings, joined text for prompt).
    """
    jc = job_config or {}
    r  = jc.get("grading_rubric_template")
    if r is not None:
        rubric_list = r if isinstance(r, list) else [r]
        return rubric_list, "\n".join(rubric_list)
    criteria = item.get("evaluation_criteria") or []
    rubric_list = criteria if isinstance(criteria, list) else [criteria]
    return rubric_list, json.dumps(rubric_list, indent=2)


def _extract_dimension_labels(rubric_list: List[str]) -> List[str]:
    """
    Extract short dimension names from rubric strings.
    e.g. "1. Correctness: Does the solution..." → "Correctness"
         "Criterion 1: Security — ..."          → "Security"
    Falls back to "Dim N" if no clear label is found.
    """
    labels = []
    for i, line in enumerate(rubric_list):
        # Match patterns like "1. Correctness:", "Criterion 1: Security", "Correctness:"
        m = re.search(r'(?:\d+\.\s*)?([A-Za-z][A-Za-z /]+?)(?:\s*[:\-—])', line)
        labels.append(m.group(1).strip() if m else f"Dim {i+1}")
    return labels


def step_3_grading(results: List[Dict], job_config: Dict = None) -> Tuple[List[Dict], float, int]:
    """
    Step 3: Teacher (Judge) grades each student answer.
    Uses job_config grading_rubric_template or per-task evaluation_criteria.
    Scale from job_config grading_scale_max or pipeline_config grading.scale_max (default 5).
    Returns (graded_results, average_score, scale_max).
    """
    if not utils.CONFIG:
        utils.load_config()

    jc       = job_config or {}
    scale_max = 5
    if jc.get("grading_scale_max") is not None:
        scale_max = int(jc["grading_scale_max"])
    elif utils.CONFIG:
        scale_max = int(utils.CONFIG.get("grading", {}).get("scale_max", 5))

    judge_role = jc.get("role_description") or "Expert grader"
    additional_checks = jc.get("grading_notes") or "Score fairly and consistently."
    template = load_prompt_template("step3_grading", jc.get("job_id"))
    if not template:
        raise FileNotFoundError("prompt_templates/step3_grading.txt not found.")

    max_tok  = utils.CONFIG.get("max_tokens", {}).get("step3_grading", 1600)
    temp     = utils.CONFIG.get("temperature", {}).get("step3_grading", 0.1)
    timeout  = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 120)
    graded_results = []
    total_score = 0.0
    valid_grades = 0

    print(f"\n--- STEP 3: Judge [{utils.JUDGE_MODEL_NAME}] grades {len(results)} tasks ---")

    for i, item in enumerate(results):
        task_text    = item.get("user_prompt") or ""
        context_text = item.get("reference_context") or ""
        rubric_list, rubric_text = _get_rubric(item, job_config)
        num_dimensions   = max(1, len(rubric_list))
        dimension_labels = _extract_dimension_labels(rubric_list)
        student_answer_text = item.get("student_answer") or ""

        print(f"\n  Task {i+1}/{len(results)}: {item.get('task_id', '?')} | {item.get('task_type', '?')}")

        # Skip grading if student produced no answer (API timeout) — don't pollute scores
        if not student_answer_text or student_answer_text.strip() in ("(no response)", ""):
            print("    Skipped (no student answer — API timeout)")
            item["score"] = None
            item["dimension_scores"] = []
            item["dimension_labels"] = dimension_labels
            item["reason"] = "Skipped: student API timed out, no answer produced."
            results[i] = item
            continue

        grading_prompt = render_prompt_template(
            template,
            judge_role=judge_role,
            task=task_text,
            context=context_text,
            rubric=rubric_text,
            student_answer=student_answer_text,
            additional_checks=additional_checks,
            score_min=1,
            score_max=scale_max,
            num_dimensions=num_dimensions,
        )
        response = call_llm(
            utils.JUDGE_API_URL, utils.JUDGE_MODEL_NAME,
            [
                {"role": "system", "content": "Output a valid JSON object with dimension scores. No markdown, no code fences."},
                {"role": "user", "content": grading_prompt},
            ],
            max_tokens=max_tok, temperature=temp, timeout=timeout,
        )
        if not response:
            item["score"] = 0
            item["dimension_scores"] = []
            item["dimension_labels"] = dimension_labels
            item["grade_reason"] = "No response from judge."
            graded_results.append(item)
            continue

        try:
            grade_data = extract_json(strip_thinking(response))
            dim_scores = grade_data.get("dimension_scores")
            if isinstance(dim_scores, list) and len(dim_scores) > 0:
                clamped = [max(1, min(scale_max, int(s))) for s in dim_scores if isinstance(s, (int, float))]
                if clamped:
                    dimension_scores = clamped[:num_dimensions]
                    score = round(sum(dimension_scores) / len(dimension_scores), 1)
                else:
                    dimension_scores = []
                    score = 0
            else:
                raw_score = grade_data.get("score")
                if raw_score is not None and isinstance(raw_score, (int, float)):
                    score = max(1, min(scale_max, int(round(float(raw_score)))))
                    dimension_scores = [score]
                else:
                    score = 0
                    dimension_scores = []

            item["dimension_labels"] = dimension_labels
            item["dimension_scores"] = dimension_scores
            item["score"] = score
            item["grade_reason"] = grade_data.get("reason", "") or "No reason given."
            total_score += score
            valid_grades += 1

            for label, s in zip(dimension_labels, dimension_scores):
                print(f"    {label:<20} {s}/{scale_max}")
            print(f"    {'Average':<20} {score}/{scale_max}")
            print(f"    Reason: {item['grade_reason'][:80]}...")

        except (json.JSONDecodeError, KeyError, TypeError) as e:
            item["score"] = 0
            item["dimension_scores"] = []
            item["dimension_labels"] = dimension_labels
            item["grade_reason"] = f"Parse error: {e}"
            graded_results.append(item)
            continue
        graded_results.append(item)

    average_score = total_score / valid_grades if valid_grades > 0 else 0
    return graded_results, average_score, scale_max


def compute_exposure_summary(
    graded_tasks: List[Dict],
    dimension_labels: List[str] = None,
    scale_max: int = 5,
) -> Dict:
    """
    Compute per-dimension average scores and an overall exposure score for a job.

    exposure_score.overall = mean of all per-dimension averages
                           = mean over every (task × dimension) score pair.
    """
    if dimension_labels is None:
        dimension_labels = FIXED_DIMENSION_LABELS

    dim_scores: Dict[str, List[float]] = {d: [] for d in dimension_labels}

    for task in graded_tasks:
        scores = task.get("dimension_scores", [])
        labels = task.get("dimension_labels", dimension_labels)
        for i, score in enumerate(scores):
            if i < len(labels):
                lbl = labels[i]
                if lbl in dim_scores and isinstance(score, (int, float)):
                    dim_scores[lbl].append(float(score))

    dim_avgs = {
        d: round(sum(v) / len(v), 2) if v else 0.0
        for d, v in dim_scores.items()
    }
    valid_avgs = [v for v in dim_avgs.values() if v > 0]
    overall = round(sum(valid_avgs) / len(valid_avgs), 2) if valid_avgs else 0.0

    return {
        "overall": overall,
        "dimensions": dim_avgs,
        "evaluated_tasks": len(graded_tasks),
        "scale_max": scale_max,
    }


def run_batched_pipeline_for_job(
    job_config: Dict,
    all_onet_tasks: List[str],
    batch_size: int = 5,
) -> List[Dict]:
    """
    Process ALL O*NET tasks for a job in batches of batch_size.

    For each batch:
      - Step 1: Teacher generates batch_size evaluation tasks
      - Step 2: Student answers each task
      - Step 3: Judge grades each answer on the 5 fixed dimensions

    Returns a flat list of all graded task dicts.
    """
    batches = [
        all_onet_tasks[i : i + batch_size]
        for i in range(0, len(all_onet_tasks), batch_size)
    ]
    all_graded: List[Dict] = []
    delay = utils.CONFIG.get("delays", {}).get("between_tasks_seconds", 1)

    for b_idx, batch_onet in enumerate(batches):
        n = len(batch_onet)
        print(f"\n  ── Batch {b_idx + 1}/{len(batches)}  ({n} tasks) ──")

        try:
            batch_cfg = dict(job_config)
            batch_cfg["onet_tasks"] = batch_onet

            eval_tasks = step_1_generate_tasks(batch_cfg, num_tasks=n)
            if not eval_tasks:
                print(f"  Batch {b_idx + 1}: Step 1 produced no tasks — skipping.")
                continue

            answered = step_2_student_inference(eval_tasks, job_config=batch_cfg)
            if not answered:
                print(f"  Batch {b_idx + 1}: Step 2 returned no answers — skipping.")
                continue

            graded, _, _ = step_3_grading(answered, job_config=batch_cfg)
            all_graded.extend(graded)

        except Exception as e:
            print(f"  Batch {b_idx + 1}: unexpected error — {e}  (skipping)")

        if delay and b_idx < len(batches) - 1:
            time.sleep(delay)

    return all_graded


# ─────────────────────────────────────────────
# SECTION 2: JOB CONFIG & JOB SELECTION
# ─────────────────────────────────────────────

def generate_job_config(job_title: str, onet_code: str,
                        selected_tasks: List[str], num_eval_tasks: int) -> Dict:
    """
    Call the meta-prompt LLM with the generate_configuration template to produce
    a full job config dict.
    """
    print(f"\n  Generating job config for [{job_title}]...")
    template = load_prompt_template("generate_configuration")
    if not template:
        raise FileNotFoundError("prompt_templates/generate_configuration.txt not found.")

    prompt = render_prompt_template(
        template,
        job_title=job_title,
        onet_code=onet_code,
        tasks_list="\n".join(f"{i+1}. {t}" for i, t in enumerate(selected_tasks)),
        num_eval_tasks=num_eval_tasks,
        onet_tasks_json=json.dumps(selected_tasks),
    )
    response = call_llm(
        utils.META_PROMPT_API_URL, utils.META_PROMPT_MODEL,
        [
            {
                "role": "system",
                "content": (
                    "You are a JSON-only API. Output ONLY a valid JSON object — "
                    "no explanation, no markdown, no code fences."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        max_tokens=8000, temperature=0.3, enable_thinking=False,
    )
    response = (response or "").strip()
    if not response:
        raise Exception(
            "Failed to generate job config — LLM API returned nothing (connection refused or error). "
            "Check that the server is running at the URL in pipeline_config.json. "
            "To run without the server, use a saved config: python qwen_main_pipeline.py --config job_config_interactive.json"
        )

    config = extract_json(response)
    if isinstance(config, list) and len(config) > 0:
        config = config[0]
    if not isinstance(config, dict):
        raise ValueError("Generated job config is not a JSON object. Got: " + type(config).__name__)
    print(f"  Config generated: job_id={config.get('job_id')} | task_types={config.get('task_types')}")
    return config


def _onet_slug(onet_code: str) -> str:
    """e.g. '13-2011.00' → '13_2011_00'  (used as cache-file key)"""
    return re.sub(r"[^a-z0-9]+", "_", onet_code.lower()).strip("_")


def generate_or_load_config(job_title: str, onet_code: str, all_tasks: List[str]) -> Dict:
    """
    Load a cached job config from configs/<onet_slug>_config.json, or generate one
    via LLM and cache it for future runs.
    Always overrides grading_rubric_template with FIXED_RUBRIC before returning.
    """
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f"{_onet_slug(onet_code)}_config.json")

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if isinstance(config, list):
            config = config[0]
        print(f"  Loaded cached config: {config_path}")
    else:
        # Use up to 10 O*NET tasks as a representative sample for config generation
        sample = all_tasks[: min(10, len(all_tasks))]
        config = generate_job_config(job_title, onet_code, sample, num_eval_tasks=len(all_tasks))
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  Config cached → {config_path}")

    # Always enforce the fixed rubric and scale
    config["grading_rubric_template"] = FIXED_RUBRIC
    config["grading_scale_max"] = 5
    return config


def load_digital_jobs(digital_jobs_csv: str) -> List[Tuple[str, str]]:
    """Load the pre-filtered 114 digital jobs from digital_jobs.csv."""
    if not os.path.isfile(digital_jobs_csv):
        print(f"Error: '{digital_jobs_csv}' not found. Run filter_digital_jobs.py first.")
        sys.exit(1)
    ddf = pd.read_csv(digital_jobs_csv)
    return [(str(r["Title"]), str(r["O*NET-SOC Code"])) for _, r in ddf.iterrows()]


def display_job_menu(it_jobs: List[Tuple[str, str]]):
    print("\n  Digital / knowledge-worker jobs (AI-evaluable):")
    for i, (title, code) in enumerate(it_jobs, 1):
        print(f"    {i}. {title}  ({code})")
    print("    q. Quit")


def select_job_interactive(it_jobs: List[Tuple[str, str]]) -> Tuple[str, str]:
    """Let user pick a job by number, or 'q' to quit."""
    while True:
        choice = input("\n  Enter job number (or q to quit): ").strip().lower()
        if choice == "q":
            print("\n  Exiting. Goodbye.")
            sys.exit(0)
        try:
            idx = int(choice)
            if 1 <= idx <= len(it_jobs):
                return it_jobs[idx - 1][0], it_jobs[idx - 1][1]
        except ValueError:
            pass
        print(f"  Invalid input. Enter a number between 1 and {len(it_jobs)}, or q to quit.")


# ── Commented out: interactive task-count selection (now always uses all tasks) ──
# def select_num_tasks() -> int:
#     """Prompt for number of tasks (1–20); return int."""
#     while True:
#         try:
#             n = input("\n  Number of tasks (1–20) [default 5]: ").strip() or "5"
#             n = int(n)
#             if 1 <= n <= 20:
#                 return n
#         except ValueError:
#             pass
#         print("  Enter a number between 1 and 20.")


# ── Commented out: LLM-based task subset selection (now always uses all tasks) ──
# def llm_select_tasks(job_title: str, all_tasks: List[str], num_tasks: int) -> List[str]:
#     """Ask LLM to select num_tasks representative tasks from all_tasks."""
#     template = load_prompt_template("task_selector")
#     if not template:
#         return all_tasks[:num_tasks]
#     tasks_list = "\n".join(f"{i}. {t}" for i, t in enumerate(all_tasks, 1))
#     prompt = render_prompt_template(
#         template,
#         job_title=job_title,
#         total_count=len(all_tasks),
#         tasks_list=tasks_list,
#         num_tasks=num_tasks,
#     )
#     print(f"  Asking LLM to select {num_tasks} representative tasks from {len(all_tasks)} O*NET tasks...")
#     timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 120)
#     response = call_llm(
#         utils.META_PROMPT_API_URL, utils.META_PROMPT_MODEL,
#         [{"role": "user", "content": prompt}],
#         max_tokens=500, temperature=0.2, timeout=timeout,
#     )
#     if not response:
#         print("  ⚠ LLM selection failed — using first N tasks as fallback")
#         return all_tasks[:num_tasks]
#     try:
#         indices = extract_json(response)
#         if isinstance(indices, list):
#             selected = [all_tasks[int(i) - 1] for i in indices
#                         if isinstance(i, (int, float)) and 1 <= int(i) <= len(all_tasks)]
#             if selected:
#                 return selected
#     except Exception as e:
#         print(f"  ⚠ Parse error ({e}) — using first N tasks as fallback")
#     return all_tasks[:num_tasks]


# ─────────────────────────────────────────────
# SECTION 3: PER-JOB RUNNER + PATCH
# ─────────────────────────────────────────────

def patch_job(
    job_title: str,
    onet_code: str,
    df: pd.DataFrame,
    batch_size: int,
    results_dir: str,
    suffix: str,
) -> bool:
    """
    Patch an existing result file:
      1. Re-grade tasks that have a student answer but no dimension scores.
      2. Evaluate O*NET tasks that were never covered (Step 1 batch failures).
    Rewrites the result file in-place with updated scores and exposure summary.
    """
    import glob as _glob

    # Find existing result file for this job
    out_path = None
    existing_data = None
    for fpath in _glob.glob(os.path.join(results_dir, "*" + suffix)):
        try:
            d = json.load(open(fpath, encoding="utf-8"))
        except Exception:
            continue
        if d.get("meta", {}).get("onet_code") == onet_code or \
           d.get("meta", {}).get("job_title") == job_title:
            out_path = fpath
            existing_data = d
            break

    if not existing_data:
        print(f"  No existing result file for [{job_title}] — run normally instead.")
        return False

    all_tasks = df[df["O*NET-SOC Code"].astype(str) == onet_code]["Task"].dropna().tolist()
    if not all_tasks:
        all_tasks = df[df["Title"] == job_title]["Task"].dropna().tolist()
    if not all_tasks:
        print(f"  No O*NET tasks found for [{job_title}].")
        return False

    try:
        job_config = generate_or_load_config(job_title, onet_code, all_tasks)
    except Exception as e:
        print(f"  Config load failed: {e}")
        return False

    existing_tasks = existing_data["tasks"]
    modified = False

    # ── Pass 1: re-grade tasks that have an answer but no scores ─────────────
    unscored = [
        (i, t) for i, t in enumerate(existing_tasks)
        if t.get("student_answer", "").strip() not in ("(no response)", "")
        and not t.get("dimension_scores")
    ]
    if unscored:
        print(f"  Re-grading {len(unscored)} unscored tasks...")
        regraded_tasks, _, _ = step_3_grading([t for _, t in unscored], job_config)
        for (orig_idx, _), patched in zip(unscored, regraded_tasks):
            if patched.get("dimension_scores"):
                existing_tasks[orig_idx] = patched
                modified = True

    # ── Pass 2: evaluate O*NET tasks that were never batched ─────────────────
    # Use fuzzy matching: an O*NET task is considered "covered" if its text appears
    # as a substring of any existing onet_source, or vice versa (handles Teacher rewrites).
    covered_sources = [t.get("onet_source", "").strip().lower() for t in existing_tasks]

    def _is_covered(onet_task: str) -> bool:
        needle = onet_task.strip().lower()
        for src in covered_sources:
            if needle in src or src in needle:
                return True
        return False

    unevaluated = [t for t in all_tasks if not _is_covered(t)]
    if unevaluated:
        print(f"  Running evaluation for {len(unevaluated)} unevaluated O*NET tasks...")
        new_tasks = run_batched_pipeline_for_job(job_config, unevaluated, batch_size=batch_size)
        existing_tasks.extend(new_tasks)
        modified = True

    if not modified:
        print(f"  [{job_title}] — no gaps found, nothing to patch.")
        return True

    # ── Recompute exposure and save ───────────────────────────────────────────
    exposure = compute_exposure_summary(existing_tasks, FIXED_DIMENSION_LABELS)
    existing_data["meta"]["evaluated_tasks"] = len(existing_tasks)
    existing_data["meta"]["exposure_score"] = exposure
    existing_data["tasks"] = existing_tasks
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    scored_count = sum(1 for t in existing_tasks if t.get("dimension_scores"))
    print(f"  Patched → {out_path}  (scored: {scored_count}/{len(existing_tasks)}, "
          f"exposure: {exposure['overall']:.2f}/5)")
    return True

def run_job(
    job_title: str,
    onet_code: str,
    df: pd.DataFrame,
    batch_size: int,
    results_dir: str,
    suffix: str,
    resume: bool,
    skip_eval: bool = False,
    output_config: str = "job_config_interactive.json",
) -> bool:
    """
    Full pipeline for one job: config → batched eval → save results.
    Returns True on success, False on skip or failure.
    """
    # Get ALL O*NET tasks for this job from the Excel data
    all_tasks = df[df["O*NET-SOC Code"].astype(str) == onet_code]["Task"].dropna().tolist()
    if not all_tasks:
        all_tasks = df[df["Title"] == job_title]["Task"].dropna().tolist()
    if not all_tasks:
        print(f"  No O*NET tasks found for [{job_title}] — skipping.")
        return False

    print(f"\n  O*NET tasks for this job: {len(all_tasks)}")

    # Generate or load cached job config
    try:
        job_config = generate_or_load_config(job_title, onet_code, all_tasks)
    except Exception as e:
        print(f"  Config generation failed: {e}")
        return False

    # Optionally save the config for inspection
    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(job_config, f, indent=2, ensure_ascii=False)

    if skip_eval:
        print(f"  (--skip-eval: config saved to {output_config}, evaluation skipped)")
        return True

    job_id   = job_config.get("job_id") or re.sub(r"[^a-z0-9]+", "_", job_title.lower()).strip("_")[:50]
    out_path = os.path.join(results_dir, f"{job_id}{suffix}")

    if resume and os.path.isfile(out_path):
        print(f"  Already completed ({out_path}) — skipping.")
        return True

    num_batches = (len(all_tasks) + batch_size - 1) // batch_size
    start = time.time()

    print(f"\n{'='*70}")
    print(f"  Job    : {job_title}")
    print(f"  SOC    : {onet_code}")
    print(f"  Tasks  : {len(all_tasks)}  |  Batch size: {batch_size}  |  Batches: {num_batches}")
    print(f"  Rubric : {FIXED_DIMENSION_LABELS}")
    print(f"{'='*70}")

    graded = run_batched_pipeline_for_job(job_config, all_tasks, batch_size=batch_size)
    if not graded:
        print(f"  No tasks were graded for [{job_title}].")
        return False

    exposure = compute_exposure_summary(graded, FIXED_DIMENSION_LABELS)

    os.makedirs(results_dir, exist_ok=True)
    output = {
        "meta": {
            "job_title":          job_title,
            "job_id":             job_id,
            "onet_code":          onet_code,
            "teacher_model":      utils.TEACHER_MODEL_NAME,
            "student_model":      utils.STUDENT_MODEL_NAME,
            "judge_model":        utils.JUDGE_MODEL_NAME,
            "total_onet_tasks":   len(all_tasks),
            "evaluated_tasks":    len(graded),
            "batches":            num_batches,
            "batch_size":         batch_size,
            "grading_scale_max":  5,
            "grading_dimensions": FIXED_DIMENSION_LABELS,
            "exposure_score":     exposure,
            "total_time_seconds": round(time.time() - start, 2),
        },
        "tasks": graded,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  Exposure score (overall): {exposure['overall']:.2f} / 5")
    print(f"  Per-dimension averages:")
    for dim, avg in exposure["dimensions"].items():
        print(f"    {dim:<25} {avg:.2f}")
    print(f"  Results saved → {out_path}")
    print(f"{'='*70}")
    return True


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def validate_config():
    """
    Check that the loaded config doesn't contain placeholder values.
    Prints a helpful error and exits if the server is not configured.
    """
    placeholders = {"YOUR_SERVER_IP", "YOUR_MODEL_NAME", "default", "localhost"}
    errors = []

    if utils.SERVER_IP in placeholders:
        errors.append(
            "  Server IP is not set.\n"
            "  Fix: copy pipeline_config.example.json → pipeline_config.json and set 'server.ip',\n"
            "       or set the EVAL_SERVER_IP environment variable,\n"
            "       or pass --server-ip on the command line."
        )
    if utils.TEACHER_MODEL_NAME in placeholders:
        errors.append(
            "  Teacher model is not set.\n"
            "  Fix: set 'models.teacher' in pipeline_config.json,\n"
            "       or set the EVAL_TEACHER_MODEL environment variable,\n"
            "       or pass --teacher-model on the command line."
        )
    if utils.STUDENT_MODEL_NAME in placeholders:
        errors.append(
            "  Student model is not set.\n"
            "  Fix: set 'models.student' in pipeline_config.json,\n"
            "       or set the EVAL_STUDENT_MODEL environment variable,\n"
            "       or pass --student-model on the command line."
        )
    if errors:
        print("\nConfiguration error — cannot start pipeline:\n")
        for e in errors:
            print(e)
        print("\nSee pipeline_config.example.json and .env.example for setup instructions.\n")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Job Evaluation Pipeline — model-agnostic, works with any OpenAI-compatible server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_pipeline.py --batch-all                         # run all 114 jobs
  python run_pipeline.py --batch-all --resume                # skip already-done jobs
  python run_pipeline.py --patch-all                         # patch all existing results
  python run_pipeline.py --patch --job "Budget"              # patch one job
  python run_pipeline.py --job "Software Developers"         # single job
  python run_pipeline.py                                     # interactive picker
  python run_pipeline.py --config saved.json                 # use saved config
  python run_pipeline.py --skip-eval --job "X"               # config only, no eval

Override server/model without editing config files:
  python run_pipeline.py --server-ip 10.0.0.1 --port 8000 --teacher-model llama3 --job "Budget"
  EVAL_SERVER_IP=10.0.0.1 EVAL_TEACHER_MODEL=llama3 python run_pipeline.py --job "Budget"
        """,
    )
    parser.add_argument("--excel",          default="Task Statements.xlsx",
                        help="O*NET Task Statements Excel file")
    parser.add_argument("--digital-jobs",   default="digital_jobs.csv",
                        help="Pre-filtered digital jobs CSV (114 jobs)")
    parser.add_argument("--job",            default=None,
                        help="Run a single job whose title contains this string")
    parser.add_argument("--batch-all",      action="store_true",
                        help="Run all jobs from digital_jobs.csv")
    parser.add_argument("--resume",         action="store_true",
                        help="Skip jobs that already have a results file (--batch-all)")
    parser.add_argument("--patch",          action="store_true",
                        help="Patch an existing result file: re-grade unscored tasks and fill gaps")
    parser.add_argument("--patch-all",      action="store_true",
                        help="Apply --patch to every existing result file")
    parser.add_argument("--batch-size",     type=int, default=None,
                        help="Tasks per batch (default: pipeline_config batch_size = 5)")
    parser.add_argument("--skip-eval",      action="store_true",
                        help="Generate config only, skip evaluation steps")
    parser.add_argument("--output",         default="job_config_interactive.json",
                        help="Path to write the last job config JSON")
    parser.add_argument("--config",         default=None,
                        help="Use an existing job config JSON (skips config LLM call)")
    # ── Model / server overrides (CLI tier — highest priority) ───────────────
    parser.add_argument("--server-ip",      default=None,
                        help="Override server IP (highest priority over config/env)")
    parser.add_argument("--port",           type=int, default=None,
                        help="Override server port for both teacher and student")
    parser.add_argument("--teacher-model",  default=None,
                        help="Override teacher model name")
    parser.add_argument("--student-model",  default=None,
                        help="Override student model name")
    parser.add_argument("--pipeline-config", default="pipeline_config.json",
                        help="Path to pipeline config JSON (default: pipeline_config.json)")
    # ── Commented out: --tasks is no longer used; all O*NET tasks are evaluated ──
    # parser.add_argument("--tasks", type=int, default=None,
    #                     help="Number of tasks (1–20); now replaced by running all tasks")
    args = parser.parse_args()

    # Load config (JSON → env vars applied inside load_config)
    utils.load_config(args.pipeline_config)

    # Apply CLI overrides (Tier 3 — highest priority)
    if args.server_ip:
        utils.SERVER_IP = args.server_ip
        port = args.port or 8000
        base = f"http://{utils.SERVER_IP}:{port}/v1/chat/completions"
        utils.TEACHER_API_URL = base
        utils.STUDENT_API_URL = base
        utils.META_PROMPT_API_URL = base
    elif args.port:
        base = f"http://{utils.SERVER_IP}:{args.port}/v1/chat/completions"
        utils.TEACHER_API_URL = base
        utils.STUDENT_API_URL = base
        utils.META_PROMPT_API_URL = base
    if args.teacher_model:
        utils.TEACHER_MODEL_NAME = args.teacher_model
        utils.META_PROMPT_MODEL  = args.teacher_model
    if args.student_model:
        utils.STUDENT_MODEL_NAME = args.student_model

    # Validate config before doing anything else
    validate_config()

    batch_size  = args.batch_size or utils.CONFIG.get("task", {}).get("batch_size", 5)
    results_dir = utils.CONFIG.get("output", {}).get("results_dir", "results")
    suffix      = utils.CONFIG.get("output", {}).get("file_suffix_auto", "_results_auto.json")

    print("\nJob Evaluation Pipeline\n")

    # ── Mode A: use a saved config (skip config LLM call) ───────────────────
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            job_config = _normalize_job_config(json.load(f))
        job_config["grading_rubric_template"] = FIXED_RUBRIC
        job_config["grading_scale_max"] = 5

        job_title = job_config.get("job_title", "Unknown")
        onet_code = job_config.get("onet_code", "")
        print(f"  Loaded config: {job_title}  (job_id={job_config.get('job_id')})")

        # Load all O*NET tasks from Excel (fall back to onet_tasks in config)
        df = load_onet_excel(args.excel)
        all_tasks = df[df["O*NET-SOC Code"].astype(str) == onet_code]["Task"].dropna().tolist()
        if not all_tasks:
            all_tasks = df[df["Title"] == job_title]["Task"].dropna().tolist()
        if not all_tasks:
            all_tasks = job_config.get("onet_tasks", [])
        print(f"  O*NET tasks: {len(all_tasks)}")

        if args.skip_eval:
            print("  (--skip-eval: no evaluation run)")
            return

        graded = run_batched_pipeline_for_job(job_config, all_tasks, batch_size=batch_size)
        exposure = compute_exposure_summary(graded, FIXED_DIMENSION_LABELS)
        job_id   = job_config.get("job_id") or "unknown"
        out_path = os.path.join(results_dir, f"{job_id}{suffix}")
        os.makedirs(results_dir, exist_ok=True)
        output = {
            "meta": {
                "job_title":          job_title,
                "job_id":             job_id,
                "onet_code":          onet_code,
                "teacher_model":      utils.TEACHER_MODEL_NAME,
                "student_model":      utils.STUDENT_MODEL_NAME,
                "total_onet_tasks":   len(all_tasks),
                "evaluated_tasks":    len(graded),
                "batches":            (len(all_tasks) + batch_size - 1) // batch_size,
                "batch_size":         batch_size,
                "grading_scale_max":  5,
                "grading_dimensions": FIXED_DIMENSION_LABELS,
                "exposure_score":     exposure,
                "total_time_seconds": 0,
            },
            "tasks": graded,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Exposure score: {exposure['overall']:.2f}/5  →  {out_path}")
        print("\nPipeline complete.")
        return

    # ── Load O*NET data and the 114 digital jobs ─────────────────────────────
    df       = load_onet_excel(args.excel)
    all_jobs = load_digital_jobs(args.digital_jobs)
    print(f"\n  Loaded {len(all_jobs)} digital jobs from '{args.digital_jobs}'")

    # ── Mode B2: --patch / --patch-all  (fix gaps in existing results) ──────
    if args.patch_all or args.patch:
        jobs_to_patch = all_jobs
        if args.patch and args.job:
            jobs_to_patch = [(t, c) for t, c in all_jobs if args.job.lower() in t.lower()]
            if not jobs_to_patch:
                print(f"  No job matching '{args.job}' found.")
                sys.exit(1)
        total = len(jobs_to_patch)
        patched = 0
        for idx, (job_title, onet_code) in enumerate(jobs_to_patch, 1):
            print(f"\n[{idx}/{total}] Patching: {job_title}  ({onet_code})")
            ok = patch_job(job_title, onet_code, df, batch_size, results_dir, suffix)
            if ok:
                patched += 1
        print(f"\n  Patch complete: {patched}/{total} jobs processed.")
        return

    # ── Mode B: --batch-all  (run every job) ─────────────────────────────────
    if args.batch_all:
        total     = len(all_jobs)
        succeeded = 0
        failed: List[str] = []
        between_jobs = utils.CONFIG.get("delays", {}).get("between_jobs_seconds", 5)

        for idx, (job_title, onet_code) in enumerate(all_jobs, 1):
            print(f"\n[{idx}/{total}] {job_title}  ({onet_code})")
            ok = run_job(
                job_title, onet_code, df,
                batch_size, results_dir, suffix,
                resume=args.resume,
                skip_eval=args.skip_eval,
                output_config=args.output,
            )
            if ok:
                succeeded += 1
            else:
                failed.append(job_title)
            if between_jobs and idx < total:
                time.sleep(between_jobs)

        print(f"\n{'='*70}")
        print(f"  Batch complete: {succeeded}/{total} jobs succeeded.")
        if failed:
            print(f"  Failed / skipped ({len(failed)}):")
            for j in failed:
                print(f"    - {j}")
        print(f"{'='*70}")
        return

    # ── Mode C: single job  (--job flag or interactive picker) ───────────────
    if args.job:
        matches = [j for j in all_jobs if args.job.lower() in j[0].lower()]
        if not matches:
            print(f"  Job '{args.job}' not found in {args.digital_jobs}.")
            display_job_menu(all_jobs)
            sys.exit(1)
        job_title, onet_code = matches[0]
        print(f"\n  Selected: {job_title}  ({onet_code})")
    else:
        display_job_menu(all_jobs)
        job_title, onet_code = select_job_interactive(all_jobs)

    run_job(
        job_title, onet_code, df,
        batch_size, results_dir, suffix,
        resume=False,
        skip_eval=args.skip_eval,
        output_config=args.output,
    )
    print("\nPipeline complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye.")
        sys.exit(0)
