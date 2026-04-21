#!/usr/bin/env python3
"""
Job Evaluation Pipeline V2 — improved version with difficulty tagging and enhanced grading.

Improvements over V1:
  1. Tasks include difficulty levels (easy / medium / hard) and edge case tagging
  2. Difficulty-weighted exposure score alongside the standard average
  3. Enhanced step1 prompt: complete context requirements, edge case requirements
  4. Enhanced step3 prompt: critique-then-score with edge case awareness
  5. Task quality statistics in results (difficulty distribution, edge case coverage)

Flow (per job):
  1.  Load ALL O*NET tasks for the job from Task Statements.xlsx
  2.  Generate (or load cached) job config via LLM  →  configs/<onet_slug>_config.json
  3.  Override rubric with the fixed 5 evaluation dimensions
  4.  Process all tasks in batches of 5:
        Step 1  (Teacher)   → generate evaluation task objects with difficulty + edge cases
        Step 2  (Student)   → student model answers each task
        Step 3  (Judge)     → critique-then-score on 5 dimensions
  5.  Compute per-dimension averages, difficulty-weighted score, and overall exposure score
  6.  Save to results_v2/{job_id}_results_v2.json

Usage:
  python qwen_main_pipeline_v2.py --batch-all              # run all 114 jobs
  python qwen_main_pipeline_v2.py --batch-all --resume     # skip already-completed jobs
  python qwen_main_pipeline_v2.py --job "Software Dev"     # run a single job by title fragment
  python qwen_main_pipeline_v2.py                          # interactive job picker
  python qwen_main_pipeline_v2.py --patch-all              # patch existing results
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
)

# ─────────────────────────────────────────────
# V2 PROMPT TEMPLATE DIR — separate from v1
# ─────────────────────────────────────────────
PROMPT_TEMPLATE_DIR_V2 = "prompt_templates_v2"

def _load_template_v2(step_name: str, job_id: str = None) -> str:
    """Load a prompt template from prompt_templates_v2/."""
    for name in ([f"{step_name}_{job_id}", step_name] if job_id else [step_name]):
        path = os.path.join(PROMPT_TEMPLATE_DIR_V2, f"{name}.txt")
        if os.path.isfile(path):
            with open(path, "r", encoding="utf-8") as f:
                return f.read()
    return None

def _render(template_str: str, **kwargs) -> str:
    """Replace {key} placeholders with kwargs values."""
    out = template_str
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value) if value is not None else "")
    out = re.sub(r"\{[a-zA-Z0-9_]+\}", "", out)
    return out


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

DIFFICULTY_WEIGHTS = {"easy": 0.7, "medium": 1.0, "hard": 1.5}

CONFIGS_DIR = "configs"


# ═══════════════════════════════════════════════
# SECTION 1: EVALUATION PIPELINE (Steps 1, 1b, 2, 3)
# ═══════════════════════════════════════════════

def _normalize_job_config(job_config):
    if isinstance(job_config, list) and len(job_config) == 1:
        return job_config[0]
    if isinstance(job_config, dict):
        return job_config
    raise TypeError("job_config must be a dict or a list of one dict, got " + type(job_config).__name__)


def generate_teacher_prompt(job_config: Dict, num_tasks: int) -> str:
    """Build the Step 1 prompt for v2 teacher (with difficulty + edge case requirements)."""
    job_config = _normalize_job_config(job_config)
    title = job_config["job_title"]
    role = job_config["role_description"]
    onet_list = "\n".join(f"{i+1}. {t}" for i, t in enumerate(job_config["onet_tasks"]))
    reqs_raw = job_config["specific_requirement"]
    requirements = "\n".join(reqs_raw) if isinstance(reqs_raw, list) else reqs_raw

    output_structure = json.dumps(job_config["output_structure"], indent=2)
    edge_case_count = max(1, int(num_tasks * 0.3))

    template = _load_template_v2("step1_teacher_tasks", job_config.get("job_id"))
    if not template:
        raise FileNotFoundError("prompt_templates_v2/step1_teacher_tasks.txt not found.")

    return _render(
        template,
        role=role, title=title, num_tasks=num_tasks,
        onet_list=onet_list, requirements=requirements,
        output_structure=output_structure,
        edge_case_count=edge_case_count,
    )


def step_1_generate_tasks(job_config: Dict, num_tasks: int = None) -> List[Dict]:
    """
    Step 1: Teacher generates N evaluation tasks with difficulty and edge case tags.
    Returns a list of task dicts, or [] on failure.
    """
    num_tasks = num_tasks if num_tasks is not None else utils.CONFIG.get("task", {}).get("batch_size", 5)
    max_tok   = utils.CONFIG.get("max_tokens", {}).get("step1_generate", 12000)
    timeout   = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 300)

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
        print("  Step 1: no response from teacher API -- skipping this batch.")
        return []
    try:
        cleaned = strip_thinking(response)
        tasks = extract_json(cleaned)
        if isinstance(tasks, dict):
            tasks = [tasks]
        print(f"  Generated {len(tasks)} tasks:")
        for i, t in enumerate(tasks, 1):
            diff = t.get("difficulty", "?")
            edge = " [EDGE]" if t.get("has_edge_case") else ""
            print(f"    [{i}] {t.get('task_id','?')} | {t.get('task_type','?')} | {diff}{edge}")
        return tasks
    except Exception:
        print("  JSON parse failed. Raw output (first 500 chars):")
        print(response[:500])
        return []


def step_1b_generate_reference_answers(tasks: List[Dict], job_config: Dict) -> List[Dict]:
    """
    Step 1b (NEW): Teacher generates a gold-standard reference answer for each task.
    This reference is stored in task["reference_answer"] and used by the Judge.
    """
    print(f"\n--- STEP 1b: Teacher generates reference answers for {len(tasks)} tasks ---")

    template = _load_template_v2("step1b_reference_answer", job_config.get("job_id"))
    if not template:
        print("  WARNING: step1b template not found, skipping reference answers.")
        return tasks

    max_tok = utils.CONFIG.get("max_tokens", {}).get("step1b_reference", 6000)
    timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 300)
    temp    = utils.CONFIG.get("temperature", {}).get("step1b_reference", 0.2)
    delay   = utils.CONFIG.get("delays", {}).get("between_tasks_seconds", 3)

    jc    = _normalize_job_config(job_config)
    role  = jc.get("role_description", "Domain Expert")
    title = jc.get("job_title", "Professional")

    for i, task in enumerate(tasks):
        ref_ctx = task.get("reference_context") or ""
        if not isinstance(ref_ctx, str):
            ref_ctx = json.dumps(ref_ctx, indent=2, ensure_ascii=False)

        eval_criteria = task.get("evaluation_criteria", [])
        if isinstance(eval_criteria, list):
            eval_criteria = "\n".join(f"- {c}" for c in eval_criteria)

        prompt = _render(
            template,
            role=role,
            title=title,
            user_prompt=task.get("user_prompt", ""),
            reference_context=ref_ctx,
            evaluation_criteria=eval_criteria,
        )

        print(f"  Reference {i+1}/{len(tasks)}: {task.get('task_id', '?')}...")
        resp = call_llm(
            utils.TEACHER_API_URL, utils.TEACHER_MODEL_NAME,
            [
                {"role": "system", "content": "You are a domain expert. Produce a complete, high-quality answer."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=max_tok, temperature=temp, timeout=timeout,
            enable_thinking=None,  # thinking ON for quality, strip later
        )
        if resp:
            task["reference_answer"] = strip_thinking(resp).strip()
        else:
            task["reference_answer"] = "(no reference answer generated)"

        if delay and i < len(tasks) - 1:
            time.sleep(delay)

    return tasks


def step_2_student_inference(tasks: List[Dict], job_config: Dict = None) -> List[Dict]:
    """
    Step 2: Student answers each task (blind — no reference answer).
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

    jc       = job_config or {}
    template = _load_template_v2("step2_student_system", jc.get("job_id"))
    raw_sys  = _render(template or "", job_title=jc.get("job_title", "")) if template else f"/no_think\nYou are a {jc.get('job_title', 'professional')}. Complete the task directly and professionally. Provide only the final work product."
    if "/no_think" not in raw_sys:
        raw_sys = "/no_think\n" + raw_sys

    max_tok = utils.CONFIG.get("max_tokens", {}).get("step2_student", 4000)
    temp    = utils.CONFIG.get("temperature", {}).get("step2_student", 0.2)
    timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 300)
    delay   = utils.CONFIG.get("delays", {}).get("between_tasks_seconds", 3)
    results = []

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
    jc = job_config or {}
    r  = jc.get("grading_rubric_template")
    if r is not None:
        rubric_list = r if isinstance(r, list) else [r]
        return rubric_list, "\n".join(rubric_list)
    criteria = item.get("evaluation_criteria") or []
    rubric_list = criteria if isinstance(criteria, list) else [criteria]
    return rubric_list, json.dumps(rubric_list, indent=2)


def _extract_dimension_labels(rubric_list: List[str]) -> List[str]:
    labels = []
    for i, line in enumerate(rubric_list):
        m = re.search(r'(?:\d+\.\s*)?([A-Za-z][A-Za-z /]+?)(?:\s*[:\-\u2014])', line)
        labels.append(m.group(1).strip() if m else f"Dim {i+1}")
    return labels


def step_3_grading(results: List[Dict], job_config: Dict = None) -> Tuple[List[Dict], float, int]:
    """
    Step 3 (V2): Critique-then-score grading with edge case awareness.
    """
    if not utils.CONFIG:
        utils.load_config()

    jc        = job_config or {}
    scale_max = 5
    if jc.get("grading_scale_max") is not None:
        scale_max = int(jc["grading_scale_max"])
    elif utils.CONFIG:
        scale_max = int(utils.CONFIG.get("grading", {}).get("scale_max", 5))

    judge_role = jc.get("role_description") or "Expert grader"
    additional_checks = jc.get("grading_notes") or "Score fairly and consistently."
    template = _load_template_v2("step3_grading", jc.get("job_id"))
    if not template:
        raise FileNotFoundError("prompt_templates_v2/step3_grading.txt not found.")

    max_tok = utils.CONFIG.get("max_tokens", {}).get("step3_grading", 14000)
    temp    = utils.CONFIG.get("temperature", {}).get("step3_grading", 0.1)
    timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 300)
    graded_results = []
    total_score    = 0.0
    valid_grades   = 0

    print(f"\n--- STEP 3: Judge [{utils.TEACHER_MODEL_NAME}] grades {len(results)} tasks ---")

    for i, item in enumerate(results):
        task_text       = item.get("user_prompt") or ""
        context_text    = item.get("reference_context") or ""
        rubric_list, rubric_text = _get_rubric(item, job_config)
        num_dimensions   = max(1, len(rubric_list))
        dimension_labels = _extract_dimension_labels(rubric_list)
        student_answer_text = item.get("student_answer") or ""

        print(f"\n  Task {i+1}/{len(results)}: {item.get('task_id', '?')} | {item.get('task_type', '?')}")

        if not student_answer_text or student_answer_text.strip() in ("(no response)", ""):
            print("    Skipped (no student answer)")
            item["score"] = None
            item["dimension_scores"] = []
            item["dimension_labels"] = dimension_labels
            item["reason"] = "Skipped: student API timed out, no answer produced."
            results[i] = item
            continue

        if not isinstance(context_text, str):
            context_text = json.dumps(context_text, indent=2, ensure_ascii=False)

        grading_prompt = _render(
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
            utils.TEACHER_API_URL, utils.TEACHER_MODEL_NAME,
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
            print(f"    Reason: {item['grade_reason'][:100]}...")

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


# ═══════════════════════════════════════════════
# SECTION 2: EXPOSURE SCORE (V2 — difficulty-weighted)
# ═══════════════════════════════════════════════

def compute_exposure_summary(
    graded_tasks: List[Dict],
    dimension_labels: List[str] = None,
    scale_max: int = 5,
) -> Dict:
    """
    V2 exposure score: computes both standard and difficulty-weighted averages.

    Difficulty weighting: easy=0.7, medium=1.0, hard=1.5
    This penalizes models that ace easy tasks but fail hard ones.
    """
    if dimension_labels is None:
        dimension_labels = FIXED_DIMENSION_LABELS

    dim_scores: Dict[str, List[float]] = {d: [] for d in dimension_labels}
    dim_weighted: Dict[str, List[Tuple[float, float]]] = {d: [] for d in dimension_labels}

    difficulty_counts = {"easy": 0, "medium": 0, "hard": 0, "unknown": 0}
    edge_case_count = 0

    for task in graded_tasks:
        scores = task.get("dimension_scores", [])
        labels = task.get("dimension_labels", dimension_labels)
        diff   = task.get("difficulty", "medium")
        weight = DIFFICULTY_WEIGHTS.get(diff, 1.0)

        if diff in difficulty_counts:
            difficulty_counts[diff] += 1
        else:
            difficulty_counts["unknown"] += 1
        if task.get("has_edge_case"):
            edge_case_count += 1

        for idx, score in enumerate(scores):
            if idx < len(labels):
                lbl = labels[idx]
                if lbl in dim_scores and isinstance(score, (int, float)):
                    dim_scores[lbl].append(float(score))
                    dim_weighted[lbl].append((float(score), weight))

    dim_avgs = {
        d: round(sum(v) / len(v), 2) if v else 0.0
        for d, v in dim_scores.items()
    }

    dim_weighted_avgs = {}
    for d, pairs in dim_weighted.items():
        if pairs:
            w_sum = sum(s * w for s, w in pairs)
            w_total = sum(w for _, w in pairs)
            dim_weighted_avgs[d] = round(w_sum / w_total, 2) if w_total > 0 else 0.0
        else:
            dim_weighted_avgs[d] = 0.0

    valid_avgs = [v for v in dim_avgs.values() if v > 0]
    overall = round(sum(valid_avgs) / len(valid_avgs), 2) if valid_avgs else 0.0

    valid_w_avgs = [v for v in dim_weighted_avgs.values() if v > 0]
    overall_weighted = round(sum(valid_w_avgs) / len(valid_w_avgs), 2) if valid_w_avgs else 0.0

    return {
        "overall": overall,
        "overall_difficulty_weighted": overall_weighted,
        "dimensions": dim_avgs,
        "dimensions_difficulty_weighted": dim_weighted_avgs,
        "evaluated_tasks": len(graded_tasks),
        "scale_max": scale_max,
        "task_statistics": {
            "difficulty_distribution": difficulty_counts,
            "edge_case_tasks": edge_case_count,
        },
    }


# ═══════════════════════════════════════════════
# SECTION 3: BATCHED PIPELINE
# ═══════════════════════════════════════════════

def run_batched_pipeline_for_job(
    job_config: Dict,
    all_onet_tasks: List[str],
    batch_size: int = 5,
) -> List[Dict]:
    """
    V2 pipeline: Step 1 → Step 2 → Step 3 (critique-then-score).
    """
    batches = [
        all_onet_tasks[i : i + batch_size]
        for i in range(0, len(all_onet_tasks), batch_size)
    ]
    all_graded: List[Dict] = []
    delay = utils.CONFIG.get("delays", {}).get("between_tasks_seconds", 1)

    for b_idx, batch_onet in enumerate(batches):
        n = len(batch_onet)
        print(f"\n  -- Batch {b_idx + 1}/{len(batches)}  ({n} tasks) --")

        try:
            batch_cfg = dict(job_config)
            batch_cfg["onet_tasks"] = batch_onet

            # Step 1: Generate tasks
            eval_tasks = step_1_generate_tasks(batch_cfg, num_tasks=n)
            if not eval_tasks:
                print(f"  Batch {b_idx + 1}: Step 1 produced no tasks -- skipping.")
                continue

            # Step 2: Student answers
            answered = step_2_student_inference(eval_tasks, job_config=batch_cfg)
            if not answered:
                print(f"  Batch {b_idx + 1}: Step 2 returned no answers -- skipping.")
                continue

            # Step 3: Grading
            graded, _, _ = step_3_grading(answered, job_config=batch_cfg)
            all_graded.extend(graded)

        except Exception as e:
            print(f"  Batch {b_idx + 1}: unexpected error -- {e}  (skipping)")

        if delay and b_idx < len(batches) - 1:
            time.sleep(delay)

    return all_graded


# ═══════════════════════════════════════════════
# SECTION 4: JOB CONFIG & JOB SELECTION
# ═══════════════════════════════════════════════

def generate_job_config(job_title: str, onet_code: str,
                        selected_tasks: List[str], num_eval_tasks: int) -> Dict:
    print(f"\n  Generating job config for [{job_title}]...")
    template = _load_template_v2("generate_configuration")
    if not template:
        raise FileNotFoundError("prompt_templates_v2/generate_configuration.txt not found.")

    prompt = _render(
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
                    "You are a JSON-only API. Output ONLY a valid JSON object -- "
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
            "Failed to generate job config -- LLM API returned nothing. "
            "Check server at pipeline_config_v2.json URL."
        )

    config = extract_json(response)
    if isinstance(config, list) and len(config) > 0:
        config = config[0]
    if not isinstance(config, dict):
        raise ValueError("Generated job config is not a JSON object. Got: " + type(config).__name__)
    print(f"  Config generated: job_id={config.get('job_id')} | task_types={config.get('task_types')}")
    return config


def _onet_slug(onet_code: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", onet_code.lower()).strip("_")


def generate_or_load_config(job_title: str, onet_code: str, all_tasks: List[str]) -> Dict:
    os.makedirs(CONFIGS_DIR, exist_ok=True)
    config_path = os.path.join(CONFIGS_DIR, f"{_onet_slug(onet_code)}_config.json")

    if os.path.isfile(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        if isinstance(config, list):
            config = config[0]
        print(f"  Loaded cached config: {config_path}")
    else:
        sample = all_tasks[: min(10, len(all_tasks))]
        config = generate_job_config(job_title, onet_code, sample, num_eval_tasks=len(all_tasks))
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f, indent=2, ensure_ascii=False)
        print(f"  Config cached -> {config_path}")

    config["grading_rubric_template"] = FIXED_RUBRIC
    config["grading_scale_max"] = 5
    return config


def load_digital_jobs(digital_jobs_csv: str) -> List[Tuple[str, str]]:
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


# ═══════════════════════════════════════════════
# SECTION 5: PER-JOB RUNNER + PATCH
# ═══════════════════════════════════════════════

def patch_job(
    job_title: str, onet_code: str, df: pd.DataFrame,
    batch_size: int, results_dir: str, suffix: str,
) -> bool:
    """Patch an existing V2 result file: re-grade + fill gaps."""
    import glob as _glob

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
        print(f"  No existing result file for [{job_title}] -- run normally instead.")
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

    # Pass 1: re-grade tasks with answer but no scores
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

    # Pass 2: evaluate missing O*NET tasks
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
        print(f"  [{job_title}] -- no gaps found, nothing to patch.")
        return True

    exposure = compute_exposure_summary(existing_tasks, FIXED_DIMENSION_LABELS)
    existing_data["meta"]["evaluated_tasks"] = len(existing_tasks)
    existing_data["meta"]["exposure_score"] = exposure
    existing_data["tasks"] = existing_tasks
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(existing_data, f, indent=2, ensure_ascii=False)

    scored_count = sum(1 for t in existing_tasks if t.get("dimension_scores"))
    print(f"  Patched -> {out_path}  (scored: {scored_count}/{len(existing_tasks)}, "
          f"exposure: {exposure['overall']:.2f}/5)")
    return True


def run_job(
    job_title: str, onet_code: str, df: pd.DataFrame,
    batch_size: int, results_dir: str, suffix: str,
    resume: bool, skip_eval: bool = False,
    output_config: str = "job_config_interactive.json",
) -> bool:
    """Full V2 pipeline for one job."""
    all_tasks = df[df["O*NET-SOC Code"].astype(str) == onet_code]["Task"].dropna().tolist()
    if not all_tasks:
        all_tasks = df[df["Title"] == job_title]["Task"].dropna().tolist()
    if not all_tasks:
        print(f"  No O*NET tasks found for [{job_title}] -- skipping.")
        return False

    print(f"\n  O*NET tasks for this job: {len(all_tasks)}")

    try:
        job_config = generate_or_load_config(job_title, onet_code, all_tasks)
    except Exception as e:
        print(f"  Config generation failed: {e}")
        return False

    with open(output_config, "w", encoding="utf-8") as f:
        json.dump(job_config, f, indent=2, ensure_ascii=False)

    if skip_eval:
        print(f"  (--skip-eval: config saved to {output_config}, evaluation skipped)")
        return True

    job_id   = job_config.get("job_id") or re.sub(r"[^a-z0-9]+", "_", job_title.lower()).strip("_")[:50]
    out_path = os.path.join(results_dir, f"{job_id}{suffix}")

    if resume and os.path.isfile(out_path):
        print(f"  Already completed ({out_path}) -- skipping.")
        return True

    num_batches = (len(all_tasks) + batch_size - 1) // batch_size
    start = time.time()

    print(f"\n{'='*70}")
    print(f"  Job    : {job_title}")
    print(f"  SOC    : {onet_code}")
    print(f"  Tasks  : {len(all_tasks)}  |  Batch size: {batch_size}  |  Batches: {num_batches}")
    print(f"  Rubric : {FIXED_DIMENSION_LABELS}")
    print(f"  Mode   : V2 (difficulty tagging, edge cases, critique-then-score)")
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
            "pipeline_version":   "v2",
            "teacher_model":      utils.TEACHER_MODEL_NAME,
            "student_model":      utils.STUDENT_MODEL_NAME,
            "total_onet_tasks":   len(all_tasks),
            "evaluated_tasks":    len(graded),
            "batches":            num_batches,
            "batch_size":         batch_size,
            "grading_scale_max":  5,
            "grading_dimensions": FIXED_DIMENSION_LABELS,
            "grading_method":     "critique-then-score",
            "exposure_score":     exposure,
            "total_time_seconds": round(time.time() - start, 2),
        },
        "tasks": graded,
    }
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*70}")
    print(f"  Exposure score (overall):              {exposure['overall']:.2f} / 5")
    print(f"  Exposure score (difficulty-weighted):   {exposure['overall_difficulty_weighted']:.2f} / 5")
    print(f"  Per-dimension averages:")
    for dim in FIXED_DIMENSION_LABELS:
        avg = exposure["dimensions"].get(dim, 0)
        w_avg = exposure["dimensions_difficulty_weighted"].get(dim, 0)
        print(f"    {dim:<25} {avg:.2f}  (weighted: {w_avg:.2f})")
    stats = exposure.get("task_statistics", {})
    dd = stats.get("difficulty_distribution", {})
    print(f"  Task statistics:")
    print(f"    Difficulty: easy={dd.get('easy',0)} medium={dd.get('medium',0)} hard={dd.get('hard',0)}")
    print(f"    Edge cases: {stats.get('edge_case_tasks', 0)}")
    print(f"  Results saved -> {out_path}")
    print(f"{'='*70}")
    return True


# ═══════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Job Evaluation Pipeline V2 -- difficulty tagging & enhanced grading",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen_main_pipeline_v2.py --batch-all                   # run all 114 jobs
  python qwen_main_pipeline_v2.py --batch-all --resume          # skip already-done jobs
  python qwen_main_pipeline_v2.py --patch-all                   # patch all existing results
  python qwen_main_pipeline_v2.py --job "Accountants"           # single job
  python qwen_main_pipeline_v2.py                               # interactive picker
        """,
    )
    parser.add_argument("--excel",        default="Task Statements.xlsx",
                        help="O*NET Task Statements Excel file")
    parser.add_argument("--digital-jobs", default="digital_jobs.csv",
                        help="Pre-filtered digital jobs CSV (114 jobs)")
    parser.add_argument("--job",          default=None,
                        help="Run a single job whose title contains this string")
    parser.add_argument("--batch-all",    action="store_true",
                        help="Run all jobs from digital_jobs.csv")
    parser.add_argument("--resume",       action="store_true",
                        help="Skip jobs that already have a results file")
    parser.add_argument("--patch",        action="store_true",
                        help="Patch an existing result file")
    parser.add_argument("--patch-all",    action="store_true",
                        help="Apply --patch to every existing result file")
    parser.add_argument("--batch-size",   type=int, default=None,
                        help="Tasks per batch (default: 5)")
    parser.add_argument("--skip-eval",    action="store_true",
                        help="Generate config only, skip evaluation")
    parser.add_argument("--output",       default="job_config_interactive.json",
                        help="Path to write the last job config JSON")
    parser.add_argument("--config",       default=None,
                        help="Use an existing job config JSON")
    parser.add_argument("--pipeline-config", default="pipeline_config_v2.json",
                        help="Pipeline config file (default: pipeline_config_v2.json)")
    args = parser.parse_args()

    # Load V2 config
    utils.load_config(args.pipeline_config)

    # Override prompt template dir from config
    global PROMPT_TEMPLATE_DIR_V2
    PROMPT_TEMPLATE_DIR_V2 = utils.CONFIG.get("prompt_template_dir", "prompt_templates_v2")

    batch_size  = args.batch_size or utils.CONFIG.get("task", {}).get("batch_size", 5)
    results_dir = utils.CONFIG.get("output", {}).get("results_dir", "results_v2")
    suffix      = utils.CONFIG.get("output", {}).get("file_suffix_auto", "_results_v2.json")

    print("\nJob Evaluation Pipeline V2")
    print("  (difficulty tagging, edge cases, critique-then-score)\n")

    # Mode A: saved config
    if args.config:
        with open(args.config, "r", encoding="utf-8") as f:
            job_config = _normalize_job_config(json.load(f))
        job_config["grading_rubric_template"] = FIXED_RUBRIC
        job_config["grading_scale_max"] = 5

        job_title = job_config.get("job_title", "Unknown")
        onet_code = job_config.get("onet_code", "")
        print(f"  Loaded config: {job_title}")

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
                "job_title": job_title, "job_id": job_id, "onet_code": onet_code,
                "pipeline_version": "v2",
                "teacher_model": utils.TEACHER_MODEL_NAME, "student_model": utils.STUDENT_MODEL_NAME,
                "total_onet_tasks": len(all_tasks), "evaluated_tasks": len(graded),
                "batches": (len(all_tasks) + batch_size - 1) // batch_size,
                "batch_size": batch_size, "grading_scale_max": 5,
                "grading_dimensions": FIXED_DIMENSION_LABELS,
                "grading_method": "critique-then-score",
                "exposure_score": exposure, "total_time_seconds": 0,
            },
            "tasks": graded,
        }
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"\n  Exposure score: {exposure['overall']:.2f}/5  ->  {out_path}")
        print("\nPipeline V2 complete.")
        return

    # Load data
    df       = load_onet_excel(args.excel)
    all_jobs = load_digital_jobs(args.digital_jobs)
    print(f"\n  Loaded {len(all_jobs)} digital jobs from '{args.digital_jobs}'")

    # Mode B: patch
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

    # Mode C: batch all
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
                resume=args.resume, skip_eval=args.skip_eval,
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

    # Mode D: single job
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
        resume=False, skip_eval=args.skip_eval,
        output_config=args.output,
    )
    print("\nPipeline V2 complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted. Goodbye.")
        sys.exit(0)
