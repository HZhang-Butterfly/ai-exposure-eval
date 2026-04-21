# DS 面试准备文档 — AI Exposure Evaluation Pipeline

> 本文档用于面试准备，覆盖项目的完整技术细节、设计决策、踩坑经历和可能被问到的所有问题。

---

## 一、30 秒电梯介绍

"我在 Butterfly Labs 做的核心项目是帮企业量化**哪些岗位的工作任务更容易被 AI 替代**。我们从 O\*NET 职业数据库提取了 114 个数字化岗位的真实任务，搭建了一套 Teacher-Student-Judge 架构的 LLM 评测流水线：Teacher 把原始任务描述转化成评测题，Student 模拟从业者答题，Judge 从五个维度打分。最终每个岗位得到一个 1-5 分的 AI 暴露度得分，企业可以用这个数据做岗位优先级排序和培训规划。同时我用 Streamlit 开发了一个交互式数据看板，让结果可解释、可比较。"

---

## 二、项目背景与业务价值

### 业务问题
企业推进数字化转型时，常遇到两个困境：
1. **不知道从哪个岗位入手**：靠主观判断哪个岗位"该优化"，结论因人而异，缺乏说服力
2. **无法量化 AI 的替代程度**：知道"AI 可以辅助"，但不知道能替代多少比例的工作

### 解决思路
用 LLM 模拟真实工作场景中的"出题-答题-评分"过程，把 AI 在该岗位上的实际任务完成质量量化为数字分数，分数越高说明 AI 越能胜任该岗位任务，即**AI 暴露度（Exposure Score）越高**。

### 业务价值
- 企业 HR 可以拿曝光分做**岗位优先级排序**，决定哪些岗位先做 AI 辅助工具落地
- 培训部门可以用**维度分数**定位具体的能力短板，设计针对性的人机协作培训
- 管理层可以用**跨岗位对比图**向董事会汇报数字化转型的 ROI 预期

---

## 三、数据来源

### O\*NET 数据库
- 美国劳工部维护的职业信息数据库，覆盖 900+ 职业
- 每个职业有 **Task Statements**：由真实从业者调查得到的工作任务描述
- 格式：Excel 文件，列为 `Title`（职业名）、`O*NET-SOC Code`（职业代码）、`Task`（任务描述）

### 样本规模
- 从 O\*NET 筛选出 **114 个数字化相关岗位**（通过 SOC 职业分类代码过滤）
- 保存在 `digital_jobs.csv`（114 行，每行一个职业）
- 每个职业平均约 **20-30 条 O\*NET 任务**，全部参与评测（不采样）
- 总任务数量：约 114 × 25 ≈ **2800+ 条原始任务**

### 114 个职业如何筛选
- 依据 SOC 职业大类代码，筛选知识型/数字化工作相关的职业群组
- 例如：13-xxxx（商业运营）、15-xxxx（计算机科学）、17-xxxx（工程师）等
- 通过 `filter_digital_jobs.py` 脚本实现，结果写入 `digital_jobs.csv`

---

## 四、技术架构详解

### 整体流程

```
O*NET Task Statements.xlsx
        │
        ▼
[Config 生成]  LLM 为每个职业生成评测配置（首次运行，后续读缓存）
        │
        ▼
    批处理循环（每批 5 条 O*NET 任务）
        │
        ├── Step 1: Teacher 出题
        │     输入：5条O*NET任务 + 职业描述
        │     输出：5道评测题（题目、情境数据、评分标准）
        │
        ├── Step 2: Student 答题
        │     输入：题目 + 情境数据（不含评分标准）
        │     输出：自由文本答案
        │
        └── Step 3: Judge 评分
              输入：题目 + 学生答案
              输出：5维度分数 + 评分理由（JSON）
        │
        ▼
[汇总] 计算每个维度均分 + 综合 Exposure Score
        │
        ▼
results/{job_id}_results_auto.json
```

### 模型配置
```
Teacher Model: Qwen/Qwen3.5-35B-A3B-FP8（出题 + 评分）
Student Model: Qwen/Qwen3.5-35B-A3B-FP8（答题）
服务方式: vLLM，部署在内部服务器，OpenAI-compatible API（端口 8600）
```

- 模型特征：35B 参数 MoE 架构（每次激活 3B），FP8 量化，上下文窗口 262K tokens
- Teacher 和 Student **使用同一模型**（服务器资源限制），通过不同 system prompt 区分角色

### API 调用方式
```python
# OpenAI-compatible chat completions
POST http://172.27.146.129:8600/v1/chat/completions

payload = {
    "model": "Qwen/Qwen3.5-35B-A3B-FP8",
    "messages": [
        {"role": "system", "content": "..."},
        {"role": "user",   "content": "..."}
    ],
    "max_tokens": 10000,
    "temperature": 0.2,
    "chat_template_kwargs": {"enable_thinking": False}  # 关闭 Chain-of-Thought
}
```

---

## 五、每一步的实现细节

### Config 生成（预处理步骤）

**目的**：为每个职业生成一份"评测配置"，定义该职业的评测框架。

**实现**：
- 取该职业前 10 条 O\*NET 任务作为样本，发给 LLM
- LLM 返回一个 JSON，包含：`job_id`、`role_description`、`task_types`、`output_structure`、`grading_notes`
- 生成后缓存到 `configs/{onet_slug}_config.json`，下次直接读取

**关键设计**：Config 里的 `grading_rubric_template` 字段会被代码层强制覆盖为固定的 5 维度 rubric，保证跨职业可比性：
```python
config["grading_rubric_template"] = FIXED_RUBRIC  # 强制统一
config["grading_scale_max"] = 5
```

---

### Step 1：Teacher 出题

**输入**：当前批次的 5 条 O\*NET 任务描述 + 职业角色描述 + 输出结构模板

**Prompt 设计要点**：
```
要求 Teacher：
1. 每道题必须 grounded in 一条具体的 O*NET 描述（通过 onet_source 字段标注）
2. user_prompt + reference_context 必须自包含，学生不需要任何额外信息
3. reference_context 必须包含具体数据（JSON 表格、数字记录等），不能只有描述
4. 难度定位：专业人员 30-60 分钟能完成
```

**输出格式（JSON array）**：
```json
[
  {
    "task_id": "gen_task_01",
    "onet_source": "原始 O*NET 任务描述",
    "task_type": "fraud_detection",
    "difficulty": "medium",
    "has_edge_case": true,
    "user_prompt": "分析以下交易记录，识别潜在的欺诈行为...",
    "reference_context": {"transactions": [...]},
    "evaluation_criteria": ["Correctness: ...", "Completeness: ..."]
  }
]
```

**max_tokens**：10000（生成 5 道题需要较大空间）

**为什么关闭 thinking**：Step 1 输出 JSON，thinking 模式会在 JSON 前生成大量推理文字，破坏 JSON 解析。通过 `enable_thinking=False` 关闭。

---

### Step 2：Student 答题

**输入**：`user_prompt` + `reference_context`（注意：不含 evaluation_criteria，不能让学生看到打分标准）

**System Prompt 设计**：
```
/no_think
You are a {job_title}. Complete the following task directly and professionally.
Provide ONLY the final work product — no reasoning steps, no thinking process.
Use ALL provided data. Address any anomalies or edge cases explicitly.
```

**关键设计**：
- `/no_think` 指令 + API 层 `enable_thinking=False`：双重确保不输出 thinking 内容
- Temperature=0.2：保持答案稳定性，减少随机性
- max_tokens=4000：自由文本答案约 1000-1600 tokens，4000 有足够余量

**reference_context 处理**：
```python
ref_ctx = item.get("reference_context") or ""
if not isinstance(ref_ctx, str):
    ref_ctx = json.dumps(ref_ctx, indent=2, ensure_ascii=False)
user_content = str(item.get("user_prompt")) + "\n\n--- Context ---\n" + ref_ctx
```

---

### Step 3：Judge 评分

**输入**：题目 + 学生答案 + 5 维度 rubric

**Prompt 核心设计（Critique-then-Score）**：
```
Step 1 — CRITIQUE: 先找弱点
  - What is missing? What is vague? What could a real professional do better?
  - Even a good answer usually has at least one area for improvement.

Step 2 — SCORE: 基于 critique 打分
  - 仅当批评找不出任何问题时才给 5 分
  - Most competent answers deserve 3–4
```

**分数锚定**：
```
5 = Exceptional，无明显缺口
4 = Good，有小瑕疵但不影响实用
3 = Adequate，有明显缺口
2 = Weak，重大错误或遗漏
1 = Inadequate，基本没解决任务
```

**输出格式**：
```json
{
  "dimension_scores": [4, 5, 4, 5, 5],
  "reason": "Critique-based explanation..."
}
```

**为什么 thinking 开着**：Step 3 的多维度推理从 thinking 中受益，且输出是 JSON，用 `strip_thinking()` 先剥离 `<think>` 标签再解析。max_tokens=8000（thinking 约占 2000-3000）。

---

## 六、五维评分维度

| # | 维度 | 核心问题 |
|---|---|---|
| 1 | **Correctness（正确性）** | 答案是否正确解决了所有任务要求？ |
| 2 | **Completeness（完整性）** | 任务的每一部分是否都被充分覆盖？ |
| 3 | **Best Practices（最佳实践）** | 是否符合行业规范和专业标准？ |
| 4 | **Domain Accuracy（领域准确性）** | 专业概念、术语、计算是否正确？ |
| 5 | **Clarity（清晰度）** | 输出是否结构清晰、易于理解？ |

**为什么是这 5 个维度**：
- Correctness + Domain Accuracy = 知识是否正确（客观维度）
- Completeness = 是否全面（覆盖维度）
- Best Practices = 是否专业（规范维度）
- Clarity = 是否可用（可读维度）
- 5 个维度在 114 个职业上**统一使用**，保证跨职业可比性

---

## 七、Exposure Score 计算

```python
# 每个维度收集所有任务的分数
dim_scores = {
    "Correctness":    [4, 5, 3, 4, ...],  # 每道题的该维度分数
    "Completeness":   [5, 5, 4, 5, ...],
    ...
}

# 每个维度均值
dim_avgs = {d: mean(scores) for d, scores in dim_scores.items()}

# Exposure Score = 各维度均值的均值
exposure_overall = mean(dim_avgs.values())
```

**V2 新增：难度加权**
```python
weights = {"easy": 0.7, "medium": 1.0, "hard": 1.5}

# 加权平均：困难题贡献更大
weighted_score = sum(score * weight) / sum(weight)
```

---

## 八、工程实现关键点

### JSON 解析鲁棒性（`extract_json`）

LLM 输出 JSON 时常见问题：
1. 输出被截断（达到 max_tokens 上限）
2. 输出 Markdown code fence（```json ... ```）
3. thinking 模式在 JSON 前输出推理文字

解决方案：
```python
def extract_json(text):
    # 1. 剥离 <think>...</think> 标签
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # 2. 剥离 markdown fence
    text = re.sub(r"```json|```", "", text)
    # 3. 从最后一个行首 { 或 [ 开始解析（JSON 总在最后）
    for idx in reversed(line_start_positions):
        try:
            return json.loads(text[idx:])
        except json.JSONDecodeError:
            pass
    # 4. 修复截断 JSON（补充缺失括号）
    return _repair_and_parse(text)
```

### 重试机制（`call_llm`）

```python
for attempt in range(1, max_retries + 1):  # 默认 3 次
    try:
        response = requests.post(api_url, json=payload, timeout=300)
        return response.json()["choices"][0]["message"]["content"]
    except requests.exceptions.Timeout:
        if attempt < max_retries:
            wait = 10 * (2 ** (attempt - 1))  # 10s, 20s, 40s
            time.sleep(wait)
        else:
            return None  # 放弃，不抛异常
```

### `--patch-all` 补丁模式

用于修复部分任务因 timeout 未完成评分的结果文件：
```
Pass 1（重评分）：找有 student_answer 但无 dimension_scores 的任务 → 重跑 Step 3
Pass 2（补评测）：找从未被覆盖的 O*NET 任务 → 重跑 Step 1-3
```

用 fuzzy substring matching 判断 O\*NET 任务是否被覆盖（因为 Teacher 会轻微改写原文）：
```python
def _is_covered(onet_task):
    needle = onet_task.strip().lower()
    for src in covered_sources:
        if needle in src or src in needle:  # 双向子串匹配
            return True
    return False
```

---

## 九、遇到的技术挑战（面试重点）

### 挑战 1：LLM 输出 JSON 解析失败

**现象**：约 10-15% 的 LLM 响应无法直接 `json.loads()`

**根因分析**：
- Thinking 模式输出：模型先生成 `<think>...推理过程...</think>`，再输出 JSON
- Markdown 包装：模型忽略指令，用 ` ```json ``` ` 包裹
- 截断：max_tokens 不够，JSON 数组中途被切断
- "Thinking Process:" 前缀：部分响应用纯文本推理前缀而非 `<think>` 标签

**解决路径**：
1. 增加 max_tokens（最直接）
2. API 层 `enable_thinking=False`（对 JSON 输出步骤关闭 thinking）
3. 多层 `extract_json` 逻辑（剥离标签 → 找最后一个 JSON 块 → 修复截断）
4. `_repair_and_parse`：自动补全缺失括号，或从截断数组中挽救已完整的元素

**经验教训**：LLM 工程中 JSON 解析的鲁棒性比想象中重要，需要从 prompt、API 参数、后处理三个层次同时防御。

---

### 挑战 2：API 超时导致结果不完整

**现象**：某些任务的 Step 1 或 Step 3 因 API 超时而跳过，导致结果文件里 `evaluated_tasks < total_onet_tasks`

**根因分析**：
- Step 3 评分 prompt 较长（题目 + 学生答案 + rubric），服务器处理时间超过默认 120s timeout
- 服务器在高负载下（多 batch 并发）响应变慢

**解决路径**：
1. timeout 从 120s 增加到 300s
2. 实现 3 次指数退避重试（10s, 20s, 40s）
3. 开发 `--patch-all` 模式：事后补充未完成的任务，而不是重跑整个 job
4. `between_tasks_seconds=3` 限速，避免并发打爆服务器

**数据影响**：通过 patch 机制，将结果完整率从约 85% 提升到接近 100%。

---

### 挑战 3：自评偏差（Self-Evaluation Bias）

**现象**：Judge 给的分数普遍偏高（V1 总分 4.04/5），5 分出现频率过高，评分区分度不足。

**根因**：Teacher、Student、Judge 使用同一模型。模型对自己类似风格的答案有天然偏爱。

**解决路径**：
1. **Critique-then-Score prompt**：强制 Judge 先写批评再打分，破坏"答案合理→直接高分"的捷径
2. **分数锚定**：在 prompt 里明确"大多数答案应得 3-4，5 分只给真正卓越的输出"
3. **V2 难度分层**：设计 hard 题测试能力边界，通过难度分差量化偏差程度

**效果与局限**：prompt 工程让评分理由更具体，但整体分数仍偏高。根本解决需要不同的 Judge 模型（资源限制下暂无法实现），这在项目报告中被诚实标注为局限性。

---

### 挑战 4：结果文件出现重复任务（evaluated_tasks > total_onet_tasks）

**现象**：patch 后某些 job 的 `evaluated_tasks` 比 `total_onet_tasks` 还多。

**根因**：Teacher 在生成任务时会轻微改写 O\*NET 原文（如把"Prepare detailed reports"写成"Generate formal reports"），patch 模式用精确字符串匹配判断任务是否已被覆盖，导致改写后的任务被误判为"未覆盖"，重复评测。

**解决**：改用双向 fuzzy substring matching。运行一次性去重脚本清理已生成的重复数据（保留得分更完整的版本）。

---

## 十、数据结果分析

### 整体结果（V1，114 个职业）

结果文件结构：
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
        "Correctness": 3.60,
        "Completeness": 4.23,
        "Best Practices": 3.73,
        "Domain Accuracy": 3.77,
        "Clarity": 4.87
      }
    }
  },
  "tasks": [...]
}
```

### 观察到的规律

1. **Clarity 维度普遍最高**：模型输出格式清晰，但不代表内容正确
2. **Correctness 和 Best Practices 区分度最好**：这两个维度对领域知识要求高，分数差异更大
3. **不同职业的分数分布差异明显**：技术类职业（SWE、Data Analyst）分数相对稳定，跨领域型职业（如 Management Analyst）分数方差更大

---

## 十一、Dashboard（Streamlit）

**技术栈**：Python + Streamlit + Plotly + Pandas

**四个页面**：

| 页面 | 核心图表 | 业务目的 |
|---|---|---|
| Overview | 职业排名柱状图、分数分布直方图、维度热力图（Top 30） | 全局一眼看清哪些职业 AI 暴露度最高 |
| Job Comparison | 雷达图（2-5 个职业）、平行坐标图、评分对比表 | 同类职业之间横向对比 |
| Job Detail | 维度柱状图 + 雷达图、任务级得分表、Judge 理由展开 | 深入了解某个职业的具体表现 |
| Dimension Analysis | 维度箱线图、散点图（Correctness vs Domain Accuracy）、维度相关矩阵 | 分析五个维度之间的关系和分布 |

**关键实现**：
```python
@st.cache_data  # 缓存数据加载，避免每次交互都重读文件
def load_all_results(results_dir):
    ...
```

**部署**：本地运行 `streamlit run dashboard.py`；也部署到 Streamlit Cloud（需要 `requirements.txt`）。

---

## 十二、V1 vs V2 对比

| 维度 | V1 | V2 |
|---|---|---|
| 题目难度分层 | 无 | 有（easy/medium/hard） |
| 边缘情况要求 | 无 | 有（每批≥30%含 edge case） |
| 评分方法 | critique-then-score | 同 V1，prompt 更强调 edge case 检查 |
| 曝光分 | 简单均值 | + 难度加权版本 |
| 输出统计 | 无 | 难度分布、edge case 数量 |
| Step 3 max_tokens | 12000 | 8000（去掉 reference answer 后输入更短） |
| 结果目录 | results/ | results_v2/ |

**V2 的实际问题**：因为 configs/ 目录有 V1 生成的缓存 config，V2 复用了这些 config，导致难度标签和 edge case 要求没有生效（config 里的 `output_structure` 不含新字段）。需要删除缓存重新生成才能完全激活 V2 功能。

---

## 十三、面试问题准备

### 基础问题

**Q：这个项目的核心创新点是什么？**

A：核心创新有两个。第一，用 LLM 自动化了"出题-答题-评分"这个传统上需要领域专家手工完成的过程，把岗位 AI 暴露度的评估扩展到 114 个职业、2800+ 条任务，这在规模上是人工无法实现的。第二，通过固定的五维评分框架保证了跨职业的可比性——所有职业用同一套标准打分，结果可以直接横向比较，而不是各自一套体系。

---

**Q：为什么选择 Teacher-Student-Judge 架构？**

A：这个架构来自 LLM 评测领域的 LLM-as-a-Judge 范式。相比于直接问模型"你能做这个任务吗"，这种方式有三个优势：第一，通过让模型实际**完成**任务再评分，得到的是能力的行为证据而不是自我声称；第二，把出题和答题分离，可以控制题目难度和情境；第三，Judge 可以给出结构化的多维度评分和文字理由，结果可解释。

---

**Q：五个评分维度是怎么选的？**

A：这五个维度覆盖了知识型工作任务质量的主要方面，而且在不同职业间都有实际意义。Correctness 和 Domain Accuracy 是客观知识的两个侧面；Completeness 测量覆盖广度；Best Practices 测量专业规范性；Clarity 测量可读性和可用性。重要的是，这五个维度不是针对某一个职业设计的，放在会计师、数据分析师、软件工程师身上都能打出有意义的分数，所以能保证跨职业的横向可比性。

---

**Q：如何验证你的评分结果是可信的？**

A：这是项目的局限性之一，我们没有一个完全客观的 ground truth 来验证。我们做了几件事来提高可信度：一是用固定的评分框架和 prompt 模板，保证评分过程的一致性；二是 Judge 的每个分数都附带详细的评分理由，可以人工抽查是否合理；三是从结果分布看，不同职业之间确实有差异，而不是所有职业都打一样的分，说明框架有区分度。如果要进一步验证，理想做法是找领域专家对一个职业的评分进行人工对标。

---

**Q：自评偏差有多严重，你怎么量化它？**

A：从结果上看，V1 总分均值 4.04/5，V2 是 4.78/5，都偏高，说明偏差确实存在。我们没有一个无偏的基准来精确量化偏差大小，但有一个间接指标：Clarity 这个维度的分数明显高于其他维度（接近 4.9/5），而 Clarity 是最容易评判的维度——输出格式是否清晰一眼就能看出来，Judge 评这个维度基本不会错。相比之下 Correctness（3.6/5）低很多，说明 Judge 在需要真正领域判断的维度上确实会给低分，不是完全无脑高分。偏差更多体现在把"基本合格"的答案打到了 4 而不是 3 的区间。

---

**Q：如果资源允许，你会怎么改进这个项目？**

A：有三个方向。第一，用不同的模型做 Judge，最理想是一个更大的专有模型，从结构上消除自评偏差。第二，接入 RAG 知识库，让 Teacher 出题时能检索最新的行业规范文档（比如最新版 GAAP、IFRS），而不是靠模型记忆，减少幻觉。第三，把 Student 改造成 ReAct Agent，配备计算器、规范查询等工具，这样评测的不只是答案质量，还有多步推理能力，更接近真实工作场景。

---

### 深度技术问题

**Q：你是怎么处理 LLM 输出 JSON 解析失败的？**

A：这是项目里踩坑最多的地方。我实现了一个多层防御的 `extract_json` 函数。第一层：用正则表达式剥离 `<think>` 标签和 Markdown code fence。第二层：从文本末尾开始找行首的 `{` 或 `[`（因为 JSON 总在 thinking 之后），逐个尝试解析。第三层：对截断的 JSON，用 `_repair_and_parse` 自动补全缺失的括号，或者从截断数组中挽救已完整的元素。同时在 API 层，对必须输出 JSON 的步骤（Step 1 出题、Config 生成）设置 `enable_thinking=False`，从源头减少干扰。

---

**Q：batch size=5 是怎么确定的？有没有测试过其他值？**

A：5 是基于几个因素权衡的经验值。首先，O\*NET 的任务描述平均长度约 100-150 字，5 条约 600-750 字，加上任务生成的指令，总 prompt 约 1500-2000 tokens，在 max_tokens=10000 的设置下有充足的生成空间。其次，每批生成 5 道题，Step 1 到 Step 3 走一遍约需 10-15 分钟，如果 batch 太大一旦某一步失败，损失更多。更小的 batch（比如 1-2）会导致配置生成的职业角色上下文信息不够丰富，题目类型会重复。

---

**Q：Exposure Score 的业务含义是什么，企业怎么用它做决策？**

A：Exposure Score 代表 AI 在该职业的典型工作任务上的平均完成质量，满分 5 分。4 分以上意味着 AI 在该职业的主要任务上能达到"良好"以上的质量，具有较高的替代潜力；3 分左右意味着能"及格"但有明显缺口，可能需要人机协作；2 分以下则说明 AI 还不能胜任该职业的核心任务。企业可以用维度分数做更细的决策：比如某个职业 Correctness 高但 Best Practices 低，说明 AI 知道"该做什么"但不知道"该怎么做才专业"，适合做初稿生成但不能独立输出。

---

**Q：你的流水线是批处理还是流式处理？如果要 scale up 怎么做？**

A：目前是串行批处理，一个 job 跑完再跑下一个，每批 5 个任务串行进行 Step 1-3。这主要是因为服务器资源有限，如果并发太多会 OOM 崩溃（已经碰到过）。如果 scale up，可以做几个改进：并行化不同 job 的评测（不同 job 之间没有依赖）；用消息队列管理 Step 1-3 的流水线，避免某一步失败影响整批；在服务器侧配置 vLLM 的并发请求限制（`--max-num-seqs`），用 back-pressure 而不是崩溃来控制流量。

---

## 十四、简历表述（精炼版）

**中文版**：
针对 114 个数字化岗位的 O\*NET 职业任务数据，主导开发基于 Teacher-Student-Judge 架构的 LLM 评测流水线。通过多维度打分（正确性/完整性/最佳实践/领域准确性/清晰度）量化各岗位的 AI 暴露度，处理 2,800+ 条原始任务，输出可直接用于业务决策的岗位排名数据。设计 critique-then-score 评分机制和分数锚定策略，有效抑制 LLM 自评偏差。利用 Streamlit 开发多维分析看板，集成雷达图、维度热力图和 Judge 推理详情展示，将复杂评测数据转化为直观的业务洞察。

**英文版（面试英语环境）**：
Led development of an LLM evaluation pipeline using a Teacher-Student-Judge architecture to quantify AI exposure risk across 114 digitalized occupations. Processed 2,800+ O\*NET task statements through a 3-step pipeline (task generation → student inference → multi-dimensional grading), producing per-occupation AI exposure scores across 5 dimensions. Implemented critique-then-score grading prompts and explicit score anchoring to mitigate self-evaluation bias. Built a Streamlit analytics dashboard with radar charts, heatmaps, and judge reasoning drill-downs for business stakeholder communication.

---

*文档版本：面试准备版 | 最后更新：2026-04*
