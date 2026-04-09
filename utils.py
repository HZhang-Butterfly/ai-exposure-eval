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
# All models now served from the same endpoint (port 8600)
META_PROMPT_API_URL = f"http://{SERVER_IP}:8600/v1/chat/completions"
META_PROMPT_MODEL = "Qwen/Qwen3.5-35B-A3B-FP8"

TEACHER_API_URL = f"http://{SERVER_IP}:8600/v1/chat/completions"
TEACHER_MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"

STUDENT_API_URL = f"http://{SERVER_IP}:8600/v1/chat/completions"
STUDENT_MODEL_NAME = "Qwen/Qwen3.5-35B-A3B-FP8"

# Pipeline config (loaded from pipeline_config.json); used by evaluation steps
CONFIG = {}

def load_config(path: str = "pipeline_config.json") -> dict:
    """
    Load pipeline config from JSON and update SERVER_IP, TEACHER_*, STUDENT_* in this module.
    If the file is missing, CONFIG is {} and callers rely on .get(key, default) for fallbacks.
    """
    global CONFIG, SERVER_IP, \
        TEACHER_API_URL, TEACHER_MODEL_NAME, \
        STUDENT_API_URL, STUDENT_MODEL_NAME, \
        META_PROMPT_API_URL, META_PROMPT_MODEL
    try:
        with open(path, "r", encoding="utf-8") as f:
            CONFIG = json.load(f)
    except FileNotFoundError:
        CONFIG = {}
    s = CONFIG.get("server", {})
    m = CONFIG.get("models", {})
    SERVER_IP = s.get("ip", "172.27.146.129")
    # All roles (meta-prompt / teacher / student) now use the same port and model
    port = s.get("student_port", 8600)
    api_base = f"http://{SERVER_IP}:{port}/v1/chat/completions"
    TEACHER_API_URL = api_base
    STUDENT_API_URL = api_base
    META_PROMPT_API_URL = api_base
    # Default model for all roles unless overridden in pipeline_config.json
    default_model = "Qwen/Qwen3.5-35B-A3B-FP8"
    META_PROMPT_MODEL = m.get("meta_prompt", default_model)
    TEACHER_MODEL_NAME = m.get("teacher", default_model)
    STUDENT_MODEL_NAME = m.get("student", default_model)
    return CONFIG


# ================= Utility functions =================

def call_llm(api_url, model_name, messages, max_tokens, temperature=0.2, timeout=120,
             enable_thinking=None, max_retries=3, retry_delay=10):
    """
    Send a request to the LLM API with automatic retry on timeout.

    Args:
        api_url: API endpoint URL
        model_name: Model name
        messages: List of messages
        max_tokens: Maximum tokens to generate
        temperature: Sampling temperature
        timeout: Request timeout in seconds per attempt
        enable_thinking: If False, disables chain-of-thought for Qwen3 models (pass to
                         vLLM via chat_template_kwargs). None = use server default.
        max_retries: Number of retry attempts on timeout (default 3)
        retry_delay: Base delay in seconds between retries (doubles each attempt)

    Returns:
        Text content of the API response, or None on failure.
    """
    import time as _time
    payload = {
        "model": model_name,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature
    }
    if enable_thinking is not None:
        payload["chat_template_kwargs"] = {"enable_thinking": enable_thinking}

    for attempt in range(1, max_retries + 1):
        try:
            response = requests.post(api_url, json=payload, timeout=timeout)
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]
        except requests.exceptions.Timeout:
            if attempt < max_retries:
                wait = retry_delay * (2 ** (attempt - 1))
                print(f"  Timeout on attempt {attempt}/{max_retries}, retrying in {wait}s...")
                _time.sleep(wait)
            else:
                print(f"  Timeout after {max_retries} attempts — giving up.")
                return None
        except requests.exceptions.HTTPError as e:
            print(f"HTTP Error calling API {api_url}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                try:
                    error_detail = e.response.json()
                    print(f"Error details: {json.dumps(error_detail, indent=2)}")
                except Exception:
                    print(f"Response text: {e.response.text[:500]}")
            return None
        except requests.exceptions.ConnectionError as e:
            print(f"Connection Error: Cannot connect to {api_url}")
            print(f"  Details: {e}")
            return None
        except Exception as e:
            print(f"Error calling API {api_url}: {e}")
            return None
    return None

def strip_thinking(text: str) -> str:
    """
    Remove chain-of-thought / thinking content from a model response,
    keeping only the final answer.

    Handles formats produced by Qwen3 thinking mode:
      1. <think>...</think>\\n\\nActual answer    → strips the <think> block
      2. Thinking Process:\\n...\\n</think>\\n\\nActual answer  → takes text after </think>
      3. No tags at all (thinking was suppressed)             → returns as-is
      4. Plain "Thinking Process:" header without tags        → finds JSON/answer after thinking
    """
    # Case 1 & 2: if </think> is present, take everything after it
    if "</think>" in text:
        after = text.split("</think>", 1)[1].strip()
        if after:
            return after
    # Case 1 (well-formed): strip <think>...</think> blocks
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    if cleaned != text.strip():
        return cleaned if cleaned else text.strip()
    # Case 4: plain "Thinking Process:" without tags — only extract if a JSON block follows.
    # (Free-form text answers cannot be reliably separated from thinking in this format;
    #  callers should use enable_thinking=False instead for non-JSON outputs.)
    if re.match(r"^\s*Thinking Process:", text, re.IGNORECASE):
        positions = [m.start(1) for m in re.finditer(r"(?:^|\n)[ \t]*([{\[])", text)]
        if positions:
            candidate = text[positions[-1]:]
            try:
                json.loads(candidate)
                return candidate
            except (json.JSONDecodeError, ValueError):
                pass
    return text.strip()


def _repair_and_parse(text: str):
    """
    Try to parse text as JSON, repairing truncated output by closing open brackets.
    For truncated arrays, also attempts to salvage complete elements only.
    Raises json.JSONDecodeError if still invalid after all repair attempts.
    """
    # Attempt 1: close open brackets / quotes
    repaired = text
    if repaired.count('"') % 2 == 1:
        repaired = repaired + '"'
    stack = []
    in_string = False
    escape = False
    for c in repaired:
        if escape:
            escape = False
            continue
        if c == "\\" and in_string:
            escape = True
            continue
        if c == '"' and not escape:
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
        pass

    # Attempt 2: for truncated arrays, salvage only the complete elements.
    # Find the last complete object boundary: the last '}' followed only by
    # optional whitespace/comma before the text ends.
    if text.lstrip().startswith("["):
        last_close = text.rfind("}")
        if last_close != -1:
            salvaged = text[:last_close + 1].rstrip().rstrip(",") + "\n]"
            try:
                return json.loads(salvaged)
            except json.JSONDecodeError:
                pass

    raise json.JSONDecodeError("Could not repair JSON", text, 0)


def extract_json(text):
    """
    Extract JSON from LLM output, handling:
      - <think>...</think> tags (Qwen3 structured thinking mode)
      - Plain-text chain-of-thought ("Thinking Process: ...") before the JSON
      - Markdown code fences (```json ... ```)
      - Truncated JSON (closed automatically)

    Strategy: JSON is always at the END of the response (after any thinking text).
    We collect all line-starting '{' / '[' positions and try them from LAST to FIRST:
      - Nested objects fail (extra trailing content) → we skip them
      - The outermost JSON block (which extends to the end of text) parses cleanly
    """
    # Strip <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # Strip markdown fences
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text)
    text = text.strip()

    if not text:
        raise ValueError("LLM response is empty after stripping formatting. "
                         "The model may have returned nothing — check server status.")

    # 1. Try the full text first (model obeyed JSON-only instruction)
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Collect positions of '{' or '[' that appear at the start of a line.
    #    These are the most likely candidates for a JSON block boundary.
    #    Try from LAST to FIRST so the outermost (final) JSON block wins.
    line_start_positions = [
        m.start(1)
        for m in re.finditer(r'(?:^|\n)[ \t]*([{\[])', text)
    ]
    for idx in reversed(line_start_positions):
        candidate = text[idx:]
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            try:
                return _repair_and_parse(candidate)
            except json.JSONDecodeError:
                continue

    # 3. Fall back: try the first '{' or '[' anywhere in the text
    for start_char in ('{', '['):
        idx = text.find(start_char)
        if idx != -1:
            candidate = text[idx:]
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                try:
                    return _repair_and_parse(candidate)
                except json.JSONDecodeError:
                    pass

    # 4. Last resort: repair whatever we have
    try:
        return _repair_and_parse(text)
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
