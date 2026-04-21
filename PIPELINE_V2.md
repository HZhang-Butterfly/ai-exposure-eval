# AI Exposure Evaluation Pipeline — 项目说明文档

> 本文档面向第一次接触该项目的读者，介绍项目背景、整体流程、V2 版本的改进点，以及如何运行。

---

## 1. 项目背景与目标

### 业务问题

企业在推进数字化转型时，常常面临一个难题：**哪些岗位的日常工作更容易被 AI 替代？**

靠主观判断太随意，也说不清楚"为什么这个岗位比那个岗位更危险"。我们的目标是用一套**可量化、可复现、基于真实任务数据**的方法，给出客观的答案。

### 解决思路

用 LLM 模拟"出题 → 答题 → 评分"这个过程：

- **出题**：从 O\*NET 数据库中提取每个岗位的真实工作任务，让 LLM 教师模型将其转化成可评测的题目
- **答题**：让 LLM 学生模型扮演该岗位从业者，完成题目
- **评分**：让 LLM 裁判模型从五个维度打分，量化答题质量

分数越高，说明 AI 在该岗位任务上表现越好，即该岗位**被 AI 替代的可能性越高**（AI Exposure Score）。

### 数据来源

- **O\*NET（Occupational Information Network）**：美国劳工部维护的职业任务数据库，包含 114 个数字化相关岗位的真实工作任务描述（Task Statements）
- 本项目共评测 **114 个岗位**，每个岗位平均约 20-30 条任务

---

## 2. 整体流程（V1 与 V2 通用）

```
O*NET Task Statements.xlsx
         │
         ▼
┌─────────────────────────────────────────────────────────┐
│  Per-Job Pipeline                                       │
│                                                         │
│  [Config] LLM 生成该岗位的评测配置（首次运行，之后缓存）   │
│      │                                                  │
│      ▼                                                  │
│  [Step 1] Teacher Model                                 │
│    - 输入：5 条 O*NET 任务描述（一批）                   │
│    - 输出：5 道评测题（含题目、情境数据、评分标准）        │
│      │                                                  │
│      ▼                                                  │
│  [Step 2] Student Model                                 │
│    - 输入：题目 + 情境数据                               │
│    - 输出：自由文本答案（模拟从业者完成工作任务）          │
│      │                                                  │
│      ▼                                                  │
│  [Step 3] Judge Model (Teacher)                         │
│    - 输入：题目 + 学生答案                               │
│    - 输出：5 个维度各 1-5 分 + 评分理由                  │
│      │                                                  │
│      ▼                                                  │
│  [Exposure Score] 汇总所有任务的分数                     │
│    - 每个维度的平均分                                    │
│    - 所有维度、所有任务的综合曝光分                       │
└─────────────────────────────────────────────────────────┘
         │
         ▼
results/{job_id}_results_auto.json
```

### 批处理逻辑

每个岗位有 20-30 条 O\*NET 任务，每次批处理 5 条，依次走完 Step 1-3，结果合并后计算整体曝光分。

---

## 3. 三个模型的角色

| 角色 | 模型 | 职责 |
|---|---|---|
| Teacher | Qwen3.5-35B-A3B-FP8 | 出题（生成评测任务）+ 评分（Judge） |
| Student | Qwen3.5-35B-A3B-FP8 | 答题（模拟岗位从业者） |
| Judge | Qwen3.5-35B-A3B-FP8 | 对学生答案打分（同 Teacher 模型） |

> **注**：当前三个角色使用同一模型，属于"自评"架构。为降低自评偏差，在 Judge 的 prompt 中使用了 critique-then-score（先批判再打分）和显式分数锚定策略。

---

## 4. 五维评分维度

每道题从以下 5 个维度独立打分，分值 1-5：

| # | 维度 | 含义 |
|---|---|---|
| 1 | **Correctness（正确性）** | 答案是否正确解决了任务要求？ |
| 2 | **Completeness（完整性）** | 任务的所有部分是否都被覆盖，且有足够细节？ |
| 3 | **Best Practices（最佳实践）** | 是否符合行业规范和专业标准？ |
| 4 | **Domain Accuracy（领域准确性）** | 领域专业概念、术语、计算是否正确？ |
| 5 | **Clarity（清晰度）** | 输出是否结构清晰、易于理解？ |

**评分锚定（所有维度通用）：**
- 5 = 卓越，完全正确且专业，达到或超越行业标准
- 4 = 良好，基本正确，有小瑕疵但不影响实用性
- 3 = 及格，有明显缺口或模糊之处，实用性受影响
- 2 = 较弱，重大错误或遗漏关键内容
- 1 = 不合格，未解决任务，或基本判断错误

**综合曝光分（Exposure Score）** = 所有维度均分 × 所有任务均分，取值范围 1-5。

---

## 5. 输出文件结构

每个岗位生成一个 JSON 文件，路径为 `results/{job_id}_results_auto.json`，结构如下：

```json
{
  "meta": {
    "job_title": "Accountants and Auditors",
    "onet_code": "13-2011.00",
    "total_onet_tasks": 30,
    "evaluated_tasks": 30,
    "exposure_score": {
      "overall": 4.04,
      "dimensions": {
        "Correctness": 3.6,
        "Completeness": 4.23,
        "Best Practices": 3.73,
        "Domain Accuracy": 3.77,
        "Clarity": 4.87
      }
    }
  },
  "tasks": [
    {
      "task_id": "gen_task_01",
      "onet_source": "原始 O*NET 任务描述",
      "task_type": "任务类别",
      "user_prompt": "学生收到的完整题目",
      "reference_context": "题目附带的情境数据（表格、JSON 等）",
      "student_answer": "学生的完整回答",
      "dimension_scores": [4, 5, 4, 5, 5],
      "score": 4.6,
      "grade_reason": "Judge 的评分理由"
    }
  ]
}
```

---

## 6. V2 版本改进内容

V2 在 V1 基础上进行了以下改进，**原有 V1 代码不受影响**，V2 独立运行并输出到 `results_v2/` 目录。

### 6.1 任务难度标签（Difficulty Tagging）

**问题**：V1 所有题目难度相同，高分只能说明模型"整体不错"，无法区分模型在简单任务和困难任务上的差距。

**改进**：Teacher 出题时必须为每道题标注难度：

| 难度 | 含义 | 预期得分范围 |
|---|---|---|
| `easy` | 标准知识的直接应用 | 4-5 |
| `medium` | 需要多步骤推理或概念综合 | 3-4 |
| `hard` | 涉及歧义、权衡或罕见边界情况 | 2-3 |

**要求**：每批 5 道题中至少包含 1 道 easy、1 道 hard，其余 medium。

### 6.2 边缘情况要求（Edge Case Requirements）

**问题**：V1 的题目情境数据过于"干净"，不包含现实中常见的异常值或边界条件，导致学生答案看起来都很好，但实际上缺乏对异常的处理能力。

**改进**：
- 每批 5 道题中至少 30%（即 2 道）必须包含边缘情况，用 `"has_edge_case": true` 标记
- Prompt 明确要求 Teacher 加入的边缘情况类型：异常数值、缺失字段、政策冲突、多步依赖等
- Config 生成中新增 `domain_edge_cases` 字段，列出 3-5 个领域特有的边缘情况

### 6.3 难度加权曝光分（Difficulty-Weighted Exposure Score）

**问题**：V1 的曝光分是简单均值，模型在 10 道 easy 题上满分、在 5 道 hard 题上 0 分，曝光分会虚高。

**改进**：引入难度权重，硬题的分数贡献更大：

```
easy   权重 = 0.7
medium 权重 = 1.0
hard   权重 = 1.5
```

V2 结果同时输出两个分数：
- `overall`：原始均分（与 V1 可比）
- `overall_difficulty_weighted`：难度加权分（更能反映真实能力）

### 6.4 更严格的评分 Prompt（Enhanced Critique-then-Score）

**问题**：V1 的 Judge 有时不够严格，倾向于给 4-5 分，评分区分度不足。

**改进**：
- 明确指令：大多数能力普通的答案应得 3-4 分，5 分留给真正卓越的回答
- 新增对边缘情况处理的专项审查："学生是否处理了数据中的异常值？"
- 要求 Judge 先写批评（critique）再打分（score），强制其找出答案的不足

### 6.5 更完整的情境数据要求

**问题**：V1 部分题目的 `reference_context` 只是描述性文字，缺少具体数据，学生需要凭空想象数据，评分也缺乏客观依据。

**改进**：Step 1 prompt 明确要求：
- 情境数据必须是**完整、具体**的（表格、JSON 记录，至少 5-10 行）
- 学生不应需要自己发明数据
- 多步骤任务的数据必须覆盖每一步的输入

### 6.6 任务统计字段

V2 的结果文件中新增 `task_statistics`：

```json
"task_statistics": {
  "difficulty_distribution": {"easy": 5, "medium": 18, "hard": 7},
  "edge_case_tasks": 9
}
```

---

## 7. V1 vs V2 配置对比

| 参数 | V1 (`pipeline_config.json`) | V2 (`pipeline_config_v2.json`) |
|---|---|---|
| Step 1 max_tokens | 10000 | 10000 |
| Step 2 max_tokens | 4000 | 4000 |
| Step 3 max_tokens | **12000** | **8000**（去掉 reference answer 后输入更短，不需要那么大） |
| 输出目录 | `results/` | `results_v2/` |
| 任务难度标签 | 无 | 有 |
| 边缘情况要求 | 无 | 有（≥30% per batch） |
| 难度加权分 | 无 | 有 |
| Prompt 模板目录 | `prompt_templates/` | `prompt_templates_v2/` |

---

## 8. 如何运行

### 环境要求

- Python 3.9+
- 服务器运行 vLLM，模型：`Qwen/Qwen3.5-35B-A3B-FP8`，端口：8600
- 数据文件：`Task Statements.xlsx`、`digital_jobs.csv`

### V2 运行命令

```bash
# 单个 job 测试
python qwen_main_pipeline_v2.py --job "Accountants"

# 跑全部 114 个 job
python qwen_main_pipeline_v2.py --batch-all

# 断点续跑（跳过已完成的 job）
python qwen_main_pipeline_v2.py --batch-all --resume

# 补丁模式（重跑未评分的任务）
python qwen_main_pipeline_v2.py --patch-all

# 只生成 job config，不跑评测
python qwen_main_pipeline_v2.py --job "Accountants" --skip-eval
```

### V1 运行命令（不变）

```bash
python qwen_main_pipeline.py --batch-all
python qwen_main_pipeline.py --batch-all --resume
python qwen_main_pipeline.py --patch-all
```

### 查看结果 Dashboard

```bash
streamlit run dashboard.py
```

---

## 9. 文件结构

```
job_eval/
├── qwen_main_pipeline.py        # V1 主流程
├── qwen_main_pipeline_v2.py     # V2 主流程（本文档描述）
├── utils.py                     # 共享工具函数（call_llm, extract_json 等）
├── dashboard.py                 # Streamlit 可视化看板
├── pipeline_config.json         # V1 运行参数
├── pipeline_config_v2.json      # V2 运行参数
├── digital_jobs.csv             # 114 个数字化岗位列表
├── Task Statements.xlsx         # O*NET 原始任务数据
│
├── prompt_templates/            # V1 prompt 模板
│   ├── generate_configuration.txt
│   ├── step1_teacher_tasks.txt
│   ├── step2_student_system.txt
│   └── step3_grading.txt
│
├── prompt_templates_v2/         # V2 prompt 模板（改进版）
│   ├── generate_configuration.txt
│   ├── step1_teacher_tasks.txt
│   ├── step1b_reference_answer.txt  # 备用，暂未启用
│   ├── step2_student_system.txt
│   └── step3_grading.txt
│
├── configs/                     # LLM 生成的 job config 缓存（V1/V2 共用）
├── results/                     # V1 评测结果
└── results_v2/                  # V2 评测结果
```

---

## 10. 关键设计决策说明

### 为什么用 Teacher-Student-Judge 架构而非直接让模型自评？

直接问模型"你能做这个任务吗？"会得到高度主观且不可靠的答案。通过让模型**实际完成任务**再由另一个角色评分，能得到更客观的能力证据。

### 为什么选这 5 个评分维度？

这 5 个维度覆盖了知识型工作任务质量的主要方面：
- **Correctness + Domain Accuracy** = 知识是否正确（客观维度）
- **Completeness** = 是否全面（覆盖维度）
- **Best Practices** = 是否专业（规范维度）
- **Clarity** = 是否可用（可读维度）

这 5 个维度在所有 114 个岗位上统一使用，保证了跨岗位的可比性。

### 为什么使用 O*NET 数据？

O*NET 是美国劳工部维护的权威职业数据库，任务描述来自真实从业者调查，比研究者自行设计任务更具代表性和客观性。

---

*文档版本：V2 | 最后更新：2026-04*
