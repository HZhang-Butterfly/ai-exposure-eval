import json
import requests
import time
import re

# configuration
SERVER_IP = "172.27.146.129"

TEACHER_API_URL = f"http://{SERVER_IP}:8500/v1/chat/completions"
TEACHER_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

STUDENT_API_URL = f"http://{SERVER_IP}:8600/v1/chat/completions"
STUDENT_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"


# ================= UTILITY FUNCTIONS =================

def call_llm(api_url, model_name, messages, max_tokens, temperature=0.2):
    """
    Generic function to send requests to API on our DGX server
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error calling API {api_url}: {e}")
        if hasattr(e, 'response') and e.response is not None:
            try:
                error_detail = e.response.json()
                print(f"Error details: {json.dumps(error_detail, indent=2)}")
            except:
                print(f"Response text: {e.response.text[:500]}")
        return None
    except requests.exceptions.ConnectionError as e:
        print(f"Connection Error: Cannot connect to {api_url}")
        print(f"  Details: {e}")
        print(f"  Please check if the server is running and the port is correct.")
        return None
    except Exception as e:
        print(f"Error calling API {api_url}: {e}")
        return None

def extract_json(text):
    """
    Extracts JSON array/object from text, handling Markdown code blocks.
    """
    # Remove markdown code fences if present
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    return json.loads(text)

def generate_teacher_prompt(job_config, num_tasks):
    title = job_config['job_title']
    role = job_config['role_description']
    onet_list = "\n".join([f"{i+1}. {t}" for i, t in enumerate(job_config['onet_tasks'])])
    task_types = job_config['task_types']
    reqs = job_config['specific_requirement']
    output_structure=job_config['output_structure']

    prompt = f"""
# Role
You are a {role}.

# Goal
I have provided a list of O*NET task descriptions for "{title}" below.
Your job is to **generate {num_tasks} distinct tasks**.

# Source Material: O*NET Task List
{onet_list}

# Requirements
{reqs}

# Output JSON Structure
{output_structure}
    """
    return prompt
# ================= STEP 1: GENERATE TASKS =================

def step_1_generate_tasks(job_config, num_tasks=5):
    print(f"\n--- STEP 1: Generating Tasks for {job_config['job_title']} ---")
    
    prompt = generate_teacher_prompt(job_config, num_tasks)
    
    messages = [{"role": "user", "content": prompt}]
    
    response = call_llm(TEACHER_API_URL, TEACHER_MODEL_NAME, messages, max_tokens=2500)
    if not response:
        raise Exception("Failed to generate tasks.")
        
    try:
        tasks = extract_json(response)
        print(f"Successfully generated {len(tasks)} tasks.")
        
        # Print details for each task
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

# ================= STEP 2: STUDENT INFERENCE =================

def step_2_student_inference(tasks, job_config=None):
    print(f"\n--- STEP 2: Student Model ({STUDENT_MODEL_NAME}) Answering Tasks ---")
    
    # Check connection before processing
    print(f"Checking connection to {STUDENT_API_URL}...")
    try:
        test_response = requests.get(STUDENT_API_URL.replace('/v1/chat/completions', ''), timeout=5)
        print("Connection successful")
    except requests.exceptions.ConnectionError:
        print(f"Connection failed: Cannot connect to {STUDENT_API_URL}")
        return []
    except Exception as e:
        print(f"Connection check warning: {e}")

    results = []
    
    # Generate system prompt based on job config
    if job_config:
        job_title = job_config.get('job_title', 'professional')
        system_content = f"You are a {job_title}. Solve the following task efficiently. Keep your answer concise. Do NOT exceed 150 lines of code."
    else:
        system_content = "You are a software engineer. Solve the following task efficiently. Keep your answer concise. Do NOT exceed 150 lines of code."
    
    for i, task in enumerate(tasks):
        print(f"Processing Task {i+1}/{len(tasks)}: {task.get('task_type', 'Unknown Type')}...")
        
        # Construct the prompt for the student
        student_prompt = f"""
        {task.get('user_prompt', '')}
        
        Context/Code:
        {task.get('reference_context', '')}
        """
        
        messages = [
            {"role": "system", "content": system_content},
            {"role": "user", "content": student_prompt}
        ]
        # Call Student Model 
        answer = call_llm(STUDENT_API_URL, STUDENT_MODEL_NAME, messages, max_tokens=1400, temperature=0.2)
        
        # Save the result
        task_result = task.copy()
        task_result['student_answer'] = answer if answer else "ERROR_GENERATING_ANSWER"
        results.append(task_result)
        
        # Small delay to be gentle on the server
        time.sleep(1)
        
    return results

# ================= STEP 3: GRADING =================

def step_3_grading(results, job_config=None):
    print(f"\n--- STEP 3: Judge Model ({TEACHER_MODEL_NAME}) Grading Answers ---")
    
    graded_results = []
    total_score = 0
    valid_grades = 0
    
    # Generate judge role based on job config
    if job_config:
        judge_role = f"strict {job_config.get('job_title', 'Code')} Reviewer"
        additional_checks = job_config.get('grading_notes', '')
    else:
        judge_role = "strict Code Reviewer"
        additional_checks = ""
    
    for i, item in enumerate(results):
        print(f"Grading Task {i+1}/{len(results)}")
        
        # Construct the grading prompt
        grading_prompt = f"""
        You are a {judge_role}. Grade the following student submission.
        
        --- TASK ---
        {item.get('user_prompt', '')}
        
        --- CONTEXT ---
        {item.get('reference_context', '')}
        
        --- EXPECTED CRITERIA ---
        {json.dumps(item.get('evaluation_criteria', []))}
        
        --- STUDENT SUBMISSION ---
        {item.get('student_answer', '')}
        
        --- INSTRUCTIONS ---
        Evaluate if the student met the criteria.
        {additional_checks}
        Output a strict JSON object:
        {{
            "score": (0-100 integer),
            "reason": "Short explanation of the score"
        }}
        """
        
        messages = [{"role": "user", "content": grading_prompt}]
        
        # Call Teacher Model as Judge
        judge_response = call_llm(TEACHER_API_URL, TEACHER_MODEL_NAME, messages, max_tokens=1600, temperature=0.1)
        
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



def run_pipeline_for_job(job_config):
    start_time = time.time()
    output_file = f"{job_config['job_id']}_results_1.json"
    print(f"Starting Pipeline for: {job_config['job_title']}")
    
    tasks = step_1_generate_tasks(job_config, num_tasks=5)
    
    if tasks:
        answered_tasks = step_2_student_inference(tasks, job_config=job_config)
        
        # 3. Grade
        final_data, avg_score = step_3_grading(answered_tasks, job_config=job_config)
        
        output = {
            "meta": {
                "teacher_model": TEACHER_MODEL_NAME,
                "student_model": STUDENT_MODEL_NAME,
                "average_score": avg_score,
                "total_time_seconds": round(time.time() - start_time, 2)
            },
            "tasks": final_data
        }
        # Save
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=2, ensure_ascii=False)
            
        print(f"Finished {job_config['job_title']}. Saved to {output_file}")
    else:
        print(f" Failed to generate tasks for {job_config['job_title']}")


# ================= MAIN EXECUTION =================

if __name__ == "__main__":
    with open("job_config.json", "r", encoding='utf-8') as f:
        all_jobs = json.load(f)
        
    for job in all_jobs:
        run_pipeline_for_job(job)
        time.sleep(5)


