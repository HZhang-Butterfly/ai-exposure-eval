import os
import subprocess

occupations = ["Accountants", "Web Developers", "Computer Programmers", "Financial Quantitative", "Market Research", "Lawyers", "Technical Writers", "Statisticians", "Economists", "Logistics Analysts"]

print("Running 1.5b model with 6 dimensions (results_1.5b_v2/)")
for job in occupations:
    print(f"--- Running {job} for qwen2.5:1.5b ---")
    subprocess.run([
        "python", "run_pipeline.py",
        "--job", job,
        "--student-model", "qwen2.5:1.5b",
        "--results-dir", "results_1.5b_v2"
    ])

print("Running 7b model with 6 dimensions (results_7b/)")
for job in occupations:
    print(f"--- Running {job} for qwen2.5:7b ---")
    subprocess.run([
        "python", "run_pipeline.py",
        "--job", job,
        "--student-model", "qwen2.5:7b",
        "--results-dir", "results_7b"
    ])
