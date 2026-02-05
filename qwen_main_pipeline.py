#!/usr/bin/env python3
"""
Interactive Job Selector and Evaluation Pipeline (main entry).
Usage: python qwen_main_pipeline.py
"""

import json
import time
import requests
import argparse
import pandas as pd
from typing import List, Dict, Tuple
import sys
import utils
from utils import (
    call_llm, extract_json, load_onet_excel,
    load_prompt_template, render_prompt_template,
    META_PROMPT_API_URL, META_PROMPT_MODEL,
)

# ================= EVALUATION PIPELINE=================

def generate_teacher_prompt(job_config, num_tasks):
    """Build the Step 1 prompt for the teacher model: job role, O*NET tasks, requirements, and desired output structure."""
    title = job_config['job_title']
    role = job_config['role_description']
    onet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(job_config['onet_tasks'])])
    reqs_raw = job_config['specific_requirement']
    requirements = "\n".join(reqs_raw) if isinstance(reqs_raw, list) else reqs_raw
    output_structure = json.dumps(job_config['output_structure'], indent=2)
    template = load_prompt_template("step1_teacher_tasks", job_config.get("job_id"))
    if not template:
        raise FileNotFoundError(
            "prompt_templates/step1_teacher_tasks.txt not found. Add the template file or create step1_teacher_tasks_{job_id}.txt."
        )
    return render_prompt_template(
        template,
        role=role,
        title=title,
        num_tasks=num_tasks,
        onet_list=onet_list,
        requirements=requirements,
        output_structure=output_structure,
    )


def step_1_generate_tasks(job_config, num_tasks=None):
    """Step 1: Call teacher model to generate N evaluation tasks (with task_id, user_prompt, criteria, etc.). Returns list of task dicts or [] on failure."""
    num_tasks = num_tasks if num_tasks is not None else utils.CONFIG.get("task", {}).get("num_tasks", 5)
    max_tok = utils.CONFIG.get("max_tokens", {}).get("step1_generate", 2500)
    print(f"\n--- STEP 1: Generating Tasks for {job_config['job_title']} ---")
    prompt = generate_teacher_prompt(job_config, num_tasks)
    messages = [{"role": "user", "content": prompt}]
    api_timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 120)
    response = call_llm(utils.TEACHER_API_URL, utils.TEACHER_MODEL_NAME, messages, max_tokens=max_tok, timeout=api_timeout)
    if not response:
        raise Exception(
            "Failed to generate tasks (teacher API returned no content). "
            "Check the error above: connection, timeout, or max_tokens vs model context limit."
        )
    try:
        tasks = extract_json(response)
        print(f"Successfully generated {len(tasks)} tasks.")
        print("\n--- Task Details ---")
        for i, task in enumerate(tasks, 1):
            print(f"\n[Task {i}]")
            print(f"  Task ID: {task.get('task_id', 'N/A')}")
            print(f"  Type: {task.get('task_type', 'N/A')}")
            print(f"  O*NET Source: {task.get('onet_source', 'N/A')[:80]}...")
            print(f"  User Prompt: {task.get('user_prompt', 'N/A')[:100]}...")
            print(f"  Evaluation Criteria: {len(task.get('evaluation_criteria', []))} criteria")
        return tasks
    except json.JSONDecodeError:
        print("JSON Parsing failed. Raw output:")
        print(response)
        return []


def step_2_student_inference(tasks, job_config=None):
    """Step 2: Run student model on each task; append student_answer to each task. Returns list of task dicts with student_answer (or [] if connection fails)."""
    print(f"\n--- STEP 2: Student Model ({utils.STUDENT_MODEL_NAME}) Answering Tasks ---")
    conn_timeout = utils.CONFIG.get("timeouts", {}).get("connection_check_seconds", 5)
    print(f"Checking connection to {utils.STUDENT_API_URL}...")
    try:
        test_response = requests.get(utils.STUDENT_API_URL.replace('/v1/chat/completions', ''), timeout=conn_timeout)
        print("Connection successful")
    except requests.exceptions.ConnectionError:
        print(f"Connection failed: Cannot connect to {utils.STUDENT_API_URL}")
        return []
    except Exception as e:
        print(f"Connection check warning: {e}")
    results = []
    job_title = job_config.get('job_title', 'professional') if job_config else 'professional'
    template = load_prompt_template("step2_student_system", job_config.get("job_id") if job_config else None)
    if not template:
        raise FileNotFoundError(
            "prompt_templates/step2_student_system.txt not found. Add the template file or create step2_student_system_{job_id}.txt."
        )
    system_content = render_prompt_template(template, job_title=job_title)
    for i, task in enumerate(tasks):
        print(f"Processing Task {i+1}/{len(tasks)}: {task.get('task_type', 'Unknown Type')}...")
        student_prompt = f"""
{task.get('user_prompt', '')}

Context/Code:
{task.get('reference_context', '')}
        """
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": student_prompt}
        ]
        max_tok = utils.CONFIG.get("max_tokens", {}).get("step2_student", 1400)
        temp = utils.CONFIG.get("temperature", {}).get("step2_student", 0.2)
        api_timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 120)
        answer = call_llm(utils.STUDENT_API_URL, utils.STUDENT_MODEL_NAME, messages, max_tokens=max_tok, temperature=temp, timeout=api_timeout)
        task_result = task.copy()
        task_result['student_answer'] = answer if answer else "ERROR_GENERATING_ANSWER"
        results.append(task_result)
        delay = utils.CONFIG.get("delays", {}).get("between_tasks_seconds", 1)
        time.sleep(delay)
    return results


def step_3_grading(results, job_config=None):
    """Step 3: Use teacher model as judge to grade each student answer; expect JSON with score and reason. Returns (graded_results, average_score)."""
    print(f"\n--- STEP 3: Judge Model ({utils.TEACHER_MODEL_NAME}) Grading Answers ---")
    graded_results = []
    total_score = 0
    valid_grades = 0
    if job_config:
        judge_role = f"strict {job_config.get('job_title', 'Code')} Reviewer"
        additional_checks = job_config.get('grading_notes', '')
    else:
        judge_role = "strict Code Reviewer"
        additional_checks = ""
    for i, item in enumerate(results):
        print(f"Grading Task {i+1}/{len(results)}")
        task_text = item.get('user_prompt', '')
        context_text = item.get('reference_context', '')
        criteria_text = json.dumps(item.get('evaluation_criteria', []))
        student_answer_text = item.get('student_answer', '')
        template = load_prompt_template("step3_grading", job_config.get("job_id") if job_config else None)
        if not template:
            raise FileNotFoundError(
                "prompt_templates/step3_grading.txt not found. Add the template file or create step3_grading_{job_id}.txt."
            )
        grading_prompt = render_prompt_template(
            template,
            judge_role=judge_role,
            task=task_text,
            context=context_text,
            criteria=criteria_text,
            student_answer=student_answer_text,
            additional_checks=additional_checks,
        )
        messages = [{"role": "user", "content": grading_prompt}]
        max_tok = utils.CONFIG.get("max_tokens", {}).get("step3_grading", 1600)
        temp = utils.CONFIG.get("temperature", {}).get("step3_grading", 0.1)
        api_timeout = utils.CONFIG.get("timeouts", {}).get("api_request_seconds", 120)
        judge_response = call_llm(utils.TEACHER_API_URL, utils.TEACHER_MODEL_NAME, messages, max_tokens=max_tok, temperature=temp, timeout=api_timeout)
        try:
            grade_data = extract_json(judge_response)
            item['score'] = grade_data['score']
            item['grade_reason'] = grade_data['reason']
            total_score += grade_data['score']
            valid_grades += 1
            print(f" -> Score: {grade_data['score']} | Reason: {grade_data['reason'][:50]}...")
        except (json.JSONDecodeError, KeyError, TypeError):
            print(" -> Grading format error. Assigning 0.")
            item['score'] = 0
            item['grade_reason'] = "Parsing Error"
        graded_results.append(item)
    average_score = total_score / valid_grades if valid_grades > 0 else 0
    return graded_results, average_score


def run_pipeline_for_job(job_config, num_tasks=None):
    """Run the full 3-step evaluation pipeline for one job: generate tasks → student inference → grading. Loads config from utils if needed; writes results to {job_id}_results_auto.json."""
    if not utils.CONFIG:
        utils.load_config()
    num_tasks = num_tasks if num_tasks is not None else utils.CONFIG.get("task", {}).get("num_tasks", 5)
    start_time = time.time()
    suffix = utils.CONFIG.get("output", {}).get("file_suffix_auto", "_results.json")
    output_file = f"{job_config['job_id']}{suffix}"
    print(f"\n{'='*70}")
    print(f"Starting Pipeline for: {job_config['job_title']}")
    print(f"{'='*70}")
    tasks = step_1_generate_tasks(job_config, num_tasks=num_tasks)
    if tasks:
        answered_tasks = step_2_student_inference(tasks, job_config=job_config)
        final_data, avg_score = step_3_grading(answered_tasks, job_config=job_config)
        output = {
            "meta": {
                "job_title": job_config['job_title'],
                "job_id": job_config['job_id'],
                "teacher_model": utils.TEACHER_MODEL_NAME,
                "student_model": utils.STUDENT_MODEL_NAME,
                "average_score": avg_score,
                "total_time_seconds": round(time.time() - start_time, 2)
            },
            "tasks": final_data
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
        print(f"Finished {job_config['job_title']}. Saved to {output_file}")
        print(f"Average Score: {avg_score:.1f}/100")
    else:
        print(f"Failed to generate tasks for {job_config['job_title']}")


# ================= JOB SELECTION (O*NET + interactive + LLM) =================

def get_it_jobs(df: pd.DataFrame) -> List[Tuple[str, str, int]]:
    """Filter O*NET DataFrame to IT-related job titles (keywords include software, developer, data, etc.; excludes many non-IT roles). Returns list of (job_title, onet_code, task_count)."""
    it_keywords = [
        r'\bsoftware\b', r'\bdeveloper\b', r'\bprogrammer\b',
        r'\bcomputer\b', r'information technology', r'\bit\b',
        r'\bdatabase\b', r'\bweb\b', r'\bnetwork\b', r'systems analyst',
        r'\bdevops\b', r'\bcybersecurity\b', r'information security',
        r'\bcloud\b', r'\bai\b', r'artificial intelligence',
        r'data scientist', r'data engineer', r'data analyst',
        r'computer science', r'computer systems', r'computer network',
        r'software quality', r'qa engineer', r'test engineer',
        r'\bblockchain\b', r'machine learning engineer',
        r'computer programmer', r'computer hardware', r'computer user',
        r'computer and information', r'computer network',
        r'web developer', r'web administrator', r'web designer',
        r'database administrator', r'database architect',
        r'network administrator', r'network architect',
        r'information security', r'security analyst',
        r'business intelligence', r'data warehousing'
    ]
    exclude_keywords = [
        r'aerospace', r'chemical', r'civil engineer', r'mechanical engineer',
        r'electrical engineer', r'automotive engineer', r'biomedical engineer',
        r'environmental engineer', r'petroleum engineer', r'industrial engineer',
        r'manufacturing engineer', r'nuclear engineer', r'marine engineer',
        r'agricultural engineer', r'mining engineer', r'materials engineer',
        r'aircraft', r'automotive', r'railroad', r'rail\b', r'truck',
        r'repairer', r'mechanic', r'installer', r'operator',
        r'pilot', r'driver', r'trainer', r'teacher', r'instructor',
        r'air traffic', r'airline', r'airfield', r'aircraft',
        r'automotive body', r'automotive glass',
        r'electric motor', r'power tool',
        r'heating.*air conditioning', r'refrigeration',
        r'telecommunications equipment installer', r'telecommunications line installer',
        r'telecommunications engineering specialist'
    ]
    title_col = 'Title'
    title_lower = df[title_col].str.lower()
    it_pattern = '|'.join(it_keywords)
    it_mask = title_lower.str.contains(it_pattern, na=False, regex=True)
    exclude_pattern = '|'.join(exclude_keywords)
    exclude_mask = ~title_lower.str.contains(exclude_pattern, na=False, regex=True)
    df_filtered = df[it_mask & exclude_mask]
    job_groups = df_filtered.groupby(title_col).agg({
        'O*NET-SOC Code': 'first',
        'Task': 'count'
    }).reset_index()
    job_groups.columns = ['Title', 'Code', 'TaskCount']
    job_groups = job_groups.sort_values('Title', ascending=True)
    return [(row['Title'], row['Code'], row['TaskCount']) for _, row in job_groups.iterrows()]


def display_job_menu(jobs: List[Tuple[str, str, int]]):
    """Print a numbered menu of (job_title, onet_code, task_count) for interactive selection."""
    print("\nAvailable IT Jobs from O*NET:\n")
    for idx, (title, code, count) in enumerate(jobs, 1):
        print(f"{idx:2d}. {title:50s} ({count} tasks")


def select_job_interactive(jobs: List[Tuple[str, str, int]]) -> Tuple[str, str]:
    """Prompt user to enter a job number (or 'q' to quit). Returns (job_title, onet_code) for the chosen job."""
    while True:
        try:
            choice = input("\n Enter job number (or 'q' to quit): ").strip()
            if choice.lower() == 'q':
                print("Goodbye!")
                sys.exit(0)
            idx = int(choice) - 1
            if 0 <= idx < len(jobs):
                job_title, onet_code, task_count = jobs[idx]
                print(f"\n Selected: {job_title}")
                print(f"  O*NET Code: {onet_code}")
                print(f"  Available Tasks: {task_count}")
                return job_title, onet_code
            else:
                print(f"Invalid choice. Please enter 1-{len(jobs)}")
        except ValueError:
            print("Please enter a number")
        except KeyboardInterrupt:
            print("\nGoodbye")
            sys.exit(0)


def select_num_tasks() -> int:
    """Prompt user for number of tasks to generate (1–20). Default 5 if empty; repeats until valid input."""
    while True:
        try:
            num = input("\n How many tasks to generate? (default: 5): ").strip()
            if num == "":
                return 5
            num = int(num)
            if 1 <= num <= 20:
                return num
            else:
                print("Please enter a number between 1-20")
        except ValueError:
            print("Please enter a valid number")
        except KeyboardInterrupt:
            print("\nGoodbye")
            sys.exit(0)


def llm_select_tasks(job_title: str, all_tasks: List[str], num_tasks: int) -> List[str]:
    """Use meta-prompt LLM to choose N most representative tasks from all_tasks for the given job_title. Uses task_selector template; falls back to first N on parse/API failure."""
    print(f"\nUsing LLM to select {num_tasks} most representative tasks.")
    tasks_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(all_tasks)])
    total_count = len(all_tasks)
    template = load_prompt_template("task_selector")
    if not template:
        raise FileNotFoundError(
            "prompt_templates/task_selector.txt not found. Add the template to customize task selection."
        )
    prompt = render_prompt_template(
        template,
        job_title=job_title,
        tasks_list=tasks_list,
        num_tasks=num_tasks,
        total_count=total_count,
    )
    messages = [{"role": "user", "content": prompt}]
    response = call_llm(META_PROMPT_API_URL, META_PROMPT_MODEL, messages, max_tokens=500, temperature=0.3)
    if not response:
        print("LLM selection failed, using first N tasks as fallback")
        return all_tasks[:num_tasks]
    try:
        selected_indices = extract_json(response)
        selected_tasks = [all_tasks[i-1] for i in selected_indices if 1 <= i <= len(all_tasks)]
        print(f"LLM selected {len(selected_tasks)} tasks:")
        for i, task in enumerate(selected_tasks, 1):
            print(f"  {i}. {task[:70]}...")
        return selected_tasks
    except Exception as e:
        print(f"Error parsing LLM response: {e}")
        print(f"   Using first {num_tasks} tasks as fallback")
        return all_tasks[:num_tasks]


def generate_job_config(
    job_title: str,
    onet_code: str,
    selected_tasks: List[str],
    num_eval_tasks: int
) -> Dict:
    """Call meta-prompt LLM with generate_configuration template to produce job config dict (job_id, role_description, onet_tasks, output_structure, etc.) for the evaluation pipeline."""
    print(f"\nGenerating configuration for {job_title}:")
    tasks_list = "\n".join([f"{i+1}. {task}" for i, task in enumerate(selected_tasks)])
    onet_tasks_json = json.dumps(selected_tasks)
    template = load_prompt_template("generate_configuration")
    if not template:
        raise FileNotFoundError(
            "prompt_templates/generate_configuration.txt not found.")
    meta_prompt = render_prompt_template(
        template,
        job_title=job_title,
        onet_code=onet_code,
        tasks_list=tasks_list,
        num_eval_tasks=num_eval_tasks,
        onet_tasks_json=onet_tasks_json,
    )
    messages = [{"role": "user", "content": meta_prompt}]
    response = call_llm(META_PROMPT_API_URL, META_PROMPT_MODEL, messages, max_tokens=2000, temperature=0.3)
    if not response:
        raise Exception("Failed to generate job config")
    config = extract_json(response)
    print(f"Configuration generated:")
    print(f"Job ID: {config.get('job_id')}")
    print(f"Task Types: {config.get('task_types')}")
    return config


# ================= MAIN WORKFLOW =================

def main():
    """Entry point: load O*NET data, (optionally) interactively select job and task count, use LLM to select tasks and generate job config, then optionally run the 3-step evaluation pipeline and save results."""
    parser = argparse.ArgumentParser(
        description='Interactive AI Model Evaluation Pipeline (main entry)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python qwen_main_pipeline.py
  python qwen_main_pipeline.py --job "Software Developers" --tasks 10
  python qwen_main_pipeline.py --excel data.xlsx --skip-eval
        """
    )
    parser.add_argument('--excel', type=str, default='Task Statements.xlsx',
                        help='Path to O*NET Excel file')
    parser.add_argument('--job', type=str, help='Job title (skips interactive selection)')
    parser.add_argument('--tasks', type=int, default=None,
                        help='Number of tasks to select and generate (1-20). If omitted, you will be prompted.')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Only generate config, skip evaluation pipeline')
    parser.add_argument('--output', type=str, default='job_config_interactive.json',
                        help='Output config file path')
    args = parser.parse_args()

    print("\nAI Model Evaluation Pipeline")
    print("="*70)

    df = load_onet_excel(args.excel)
    it_jobs = get_it_jobs(df)
    if len(it_jobs) == 0:
        print("No IT jobs found in the Excel file")
        sys.exit(1)
    print(f"\nFound {len(it_jobs)} IT-related jobs")

    if args.job:
        matching_jobs = [j for j in it_jobs if args.job.lower() in j[0].lower()]
        if not matching_jobs:
            print(f"Job '{args.job}' not found")
            display_job_menu(it_jobs)
            sys.exit(1)
        job_title, onet_code = matching_jobs[0][0], matching_jobs[0][1]
        print(f"\n Selected: {job_title}")
    else:
        display_job_menu(it_jobs)
        job_title, onet_code = select_job_interactive(it_jobs)

    num_tasks = args.tasks if args.tasks is not None else select_num_tasks()

    title_col = 'Title'
    task_col = 'Task'
    job_df = df[df[title_col] == job_title]
    all_tasks = job_df[task_col].dropna().tolist()
    print(f"\n Found {len(all_tasks)} tasks in O*NET for this job")

    selected_tasks = llm_select_tasks(job_title, all_tasks, num_tasks)
    job_config = generate_job_config(job_title, onet_code, selected_tasks, num_tasks)

    config_data = [job_config]
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(config_data, f, indent=2, ensure_ascii=False)
    print(f"\nConfiguration saved to: {args.output}")

    if not args.skip_eval:
        print("\nStarting Evaluation Pipeline\n")
        confirm = input("\nContinue with evaluation? (y/n): ").strip().lower()
        if confirm == 'y':
            run_pipeline_for_job(job_config, num_tasks=num_tasks)
            print("Pipeline Complete!")
        else:
            print("\nConfig saved. Run evaluation later with:")
            print(f"   python qwen_main_pipeline.py --job \"...\" --tasks N  (and load {args.output})")
    else:
        print("\nConfig generation complete (evaluation skipped)")
        print("  Run evaluation by re-running and choosing y, or load the config elsewhere.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Goodbye")
        sys.exit(0)
