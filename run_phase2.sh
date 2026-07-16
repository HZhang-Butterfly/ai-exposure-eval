#!/usr/bin/env bash
OCCUPATIONS=("Accountants" "Software Dev" "Computer Programmer" "Financial Analyst" "Market Research" "Lawyer" "Technical Writer" "Statistician" "Web Dev" "Economist")

echo "Running 1.5b model with 6 dimensions (results_1.5b_v2/)"
for job in "${OCCUPATIONS[@]}"; do
    python run_pipeline.py \
        --job "$job" \
        --student-model "qwen2.5:1.5b" \
        --results-dir "results_1.5b_v2"
done

echo "Running 7b model with 6 dimensions (results_7b/)"
for job in "${OCCUPATIONS[@]}"; do
    python run_pipeline.py \
        --job "$job" \
        --student-model "qwen2.5:7b" \
        --results-dir "results_7b"
done
