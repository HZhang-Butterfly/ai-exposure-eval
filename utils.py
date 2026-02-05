#!/usr/bin/env python3
"""
Shared utilities module - functions used by all pipeline scripts.
"""

import json
import os
import requests
import pandas as pd
import re
from typing import List, Dict, Optional

# Directory for prompt template files (relative to cwd)
PROMPT_TEMPLATE_DIR = "prompt_templates"

# ================= Configuration =================
SERVER_IP = "172.27.146.129"
META_PROMPT_API_URL = f"http://{SERVER_IP}:8500/v1/chat/completions"
META_PROMPT_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

TEACHER_API_URL = f"http://{SERVER_IP}:8500/v1/chat/completions"
TEACHER_MODEL_NAME = "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8"

STUDENT_API_URL = f"http://{SERVER_IP}:8600/v1/chat/completions"
STUDENT_MODEL_NAME = "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8"

# Pipeline config (loaded from pipeline_config.json); used by evaluation steps
CONFIG = {}

def load_config(path: str = "pipeline_config.json") -> dict:
    """
    Load pipeline config from JSON and update SERVER_IP, TEACHER_*, STUDENT_* in this module.
    If the file is missing, CONFIG is {} and callers rely on .get(key, default) for fallbacks.
    """
    global CONFIG, SERVER_IP, TEACHER_API_URL, TEACHER_MODEL_NAME, STUDENT_API_URL, STUDENT_MODEL_NAME
    try:
        with open(path, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        CONFIG = {}
    s = CONFIG.get("server", {})
    m = CONFIG.get("models", {})
    SERVER_IP = s.get("ip", "172.27.146.129")
    TEACHER_API_URL = f"http://{SERVER_IP}:{s.get('teacher_port', 8500)}/v1/chat/completions"
    STUDENT_API_URL = f"http://{SERVER_IP}:{s.get('student_port', 8600)}/v1/chat/completions"
    TEACHER_MODEL_NAME = m.get("teacher", "Qwen/Qwen3-30B-A3B-Instruct-2507-FP8")
    STUDENT_MODEL_NAME = m.get("student", "Qwen/Qwen3-Coder-30B-A3B-Instruct-FP8")
    return CONFIG


# ================= Utility functions =================

def call_llm(api_url, model_name, messages, max_tokens, temperature=0.2, timeout=120):
    """
    Send a request to the LLM API.

    Args:
        api_url: API endpoint URL
        model_name: Model name
        messages: List of messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds

    Returns:
        Text content of the API response, or None on failure.
    """
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    
    try:
        response = requests.post(api_url, json=payload, timeout=timeout)
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
    Extract JSON from text, stripping Markdown code fences.
    Tries to repair truncated JSON (e.g. cut off by max_tokens) by closing brackets.

    Args:
        text: Text containing JSON

    Returns:
        Parsed JSON object or array
    """
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to fix truncated JSON: close unclosed string, then close brackets in reverse order
        repaired = text
        if repaired.count('"') % 2 == 1:
            repaired = repaired + '"'
        stack = []
        in_string = False
        escape = False
        for i, c in enumerate(repaired):
            if escape:
                escape = False
                continue
            if c == "\\" and in_string:
                escape = True
                continue
            if (c == '"') and not escape:
                in_string = not in_string
                continue
            if not in_string:
                if c == "[":
                    stack.append("]")
                elif c == "{":
                    stack.append("}")
                elif c in "]}":
                    if stack and stack[-1] == c:
                        stack.pop()
        repaired = repaired + "".join(reversed(stack))
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            raise

def load_onet_excel(file_path: str, job_title_filter: Optional[str] = None) -> pd.DataFrame:
    """
    Load O*NET Excel file and return a DataFrame.

    Args:
        file_path: Path to the Excel file
        job_title_filter: Optional job title filter (case-insensitive)

    Returns:
        DataFrame with O*NET data
    """
    print(f"Loading O*NET data from: {file_path}")

    try:
        df = pd.read_excel(file_path)
        print(f"Loaded {len(df)} rows")
        print(f"Columns found: {df.columns.tolist()}")

        if job_title_filter:
            df = df[df['Title'].str.contains(job_title_filter, case=False, na=False)]
            print(f"Filtered to {len(df)} rows for '{job_title_filter}'")
        
        return df
    except FileNotFoundError:
        print(f"Error: File '{file_path}' not found!")
        raise
    except Exception as e:
        print(f"Error loading Excel: {e}")
        raise


# ================= Prompt template helpers =================

def load_prompt_template(step_name: str, job_id: Optional[str] = None) -> Optional[str]:
    """
    Load a prompt template from prompt_templates/.
    Tries step_name_{job_id}.txt first, then step_name.txt.
    Returns file content or None if not found.
    """
    for name in ([f"{step_name}_{job_id}", step_name] if job_id else [step_name]):
        path = os.path.join(PROMPT_TEMPLATE_DIR, f"{name}.txt")
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    return f.read()
            except Exception as e:
                print(f"Warning: could not read template {path}: {e}")
                return None
    return None


def render_prompt_template(template_str: str, **kwargs) -> str:
    """
    Replace placeholders {key} in template_str with kwargs.get(key, "").
    """
    out = template_str
    for key, value in kwargs.items():
        out = out.replace("{" + key + "}", str(value) if value is not None else "")
    # Replace any remaining {x} with empty string to avoid KeyError
    out = re.sub(r"\{[a-zA-Z0-9_]+\}", "", out)
    return out
