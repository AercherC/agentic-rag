# EvidenceFlow — Project Vision

**Status:** Long-term Direction  
**Purpose:** 描述 EvidenceFlow 的长期目标、演进原则和阶段关系。  
**Important:** 本文档不是当前阶段的实现规范。当前代码实现以 `PROJECT_SPEC.md` 为准。

---

# 1. 项目定位

EvidenceFlow 是一个面向知识检索与技术研究任务的 LLM 应用工程项目。

项目希望从一个基础、本地、可评测的 RAG 系统开始，逐步演进为能够结合：

- 本地知识库；
- 外部信息检索；
- LLM 推理；
- 工具调用；
- 证据引用；
- Evaluation；
- 工程化运行能力；

完成复杂信息查询和技术研究任务的智能系统。

项目不以“使用尽可能多的 Agent 技术”为目标。

核心目标是：

> 从最简单、完整、可测量的系统开始，根据 Evaluation 暴露的问题逐步增加能力。

---

# 2. 核心设计思想

EvidenceFlow 的长期开发遵循：

```text
Build
  ↓
Evaluate
  ↓
Find Problems
  ↓
Analyze Causes
  ↓
Add the Smallest Useful Technique
  ↓
Evaluate Again
```

每一项新技术都应该对应一个已经观察到的问题。

例如：

```text
Retrieval Recall 不足
    ↓
优化 Chunk / Hybrid Retrieval / Reranker
```

而不是直接增加 Agent。

又例如：

```text
复杂研究任务覆盖不足
    ↓
Query Decomposition
    ↓
如果仍不足
    ↓
再评估是否需要并行 Agent
```

因此项目的复杂度应随着真实问题增长，而不是提前设计成复杂系统。

---

# 3. 长期能力方向

EvidenceFlow 最终希望覆盖以下几类能力。

## 3.1 Knowledge Retrieval

能够从本地文档中获取相关信息。

主要涉及：

```text
Document Parsing
Chunking
Embedding
Vector Retrieval
Metadata
Context Construction
Citation
```

这是整个项目最基础的数据获取能力。

---

## 3.2 Retrieval Quality

在基础 Dense Retrieval 可运行后，通过 Evaluation 分析检索失败原因。

可能逐步研究：

```text
Chunk Strategy
BM25
Hybrid Retrieval
Reranker
Query Rewrite
Metadata Filter
```

是否加入这些能力，由实际评测结果决定。

---

## 3.3 External Research

系统后续可以增加 Web Search，使其能够处理：

- 本地知识库不存在的信息；
- 最新信息；
- 官方文档更新；
- 技术选型资料；
- 外部事实验证。

本地知识库与 Web Search 应作为不同的信息来源，由后续系统统一管理。

---

## 3.4 Single-Agent Workflow

当系统拥有多个工具后，可以引入单 Agent 负责：

```text
理解问题
    ↓
判断需要什么信息
    ↓
选择工具
    ↓
读取结果
    ↓
决定是否继续
    ↓
生成回答
```

第一版 Agent 应保持简单。

Agent 的意义是解决：

> 系统无法提前通过固定流程确定下一步应该调用哪个工具的问题。

---

## 3.5 Evidence and Citation

随着系统从简单问答进入研究任务，需要逐步增强来源追踪能力。

长期可能形成：

```text
Source
  ↓
Evidence
  ↓
Claim
  ↓
Final Answer / Decision
```

系统应该能够回答：

> 这个结论依据什么？

而不仅仅输出一段无法追踪来源的文本。

---

## 3.6 Evaluation

Evaluation 是项目的长期核心能力。

项目不仅需要判断：

```text
系统能不能运行
```

还需要判断：

```text
系统是否做对了
为什么做错
新技术是否真的改善了结果
增加的成本是否值得
```

长期评测可能包括：

### Retrieval

```text
Recall@K
MRR
nDCG
```

### Generation

```text
Answer Correctness
Faithfulness
Citation Correctness
Citation Completeness
```

### Agent

```text
Task Success
Tool Selection
Tool Arguments
Trajectory
Unnecessary Tool Calls
```

### Engineering

```text
Latency
Cost
Failure Rate
Recovery Rate
```

---

# 4. 项目大致演进阶段

下面只是长期路线，不代表当前必须依次全部实现。

---

## Phase 1 — Local RAG

目标：

构建完整的本地知识库 RAG 基线。

主要能力：

```text
Document
→ Chunk
→ Embedding
→ Vector Store
→ Retrieval
→ Context
→ LLM
→ Citation
```

同时建立 Retrieval Evaluation。

这一阶段的详细实现要求由当前 `PROJECT_SPEC.md` 定义。

---

## Phase 2 — Retrieval Quality Improvement

使用 Phase 1 的 Evaluation 结果分析：

```text
哪些问题检索失败？
为什么失败？
```

根据失败原因选择需要测试的技术。

可能包括：

```text
Chunk Strategy
Hybrid Retrieval
Reranker
Query Rewrite
Metadata Filtering
```

目标不是增加功能数量，而是提高 Retrieval Quality。

---

## Phase 3 — External Information Retrieval

增加外部搜索能力。

系统能够同时使用：

```text
Local Knowledge Base
+
Web Search
```

需要明确：

- 什么问题优先查本地知识库；
- 什么问题需要最新外部信息；
- 如何保存外部 Source；
- 如何处理来源时间和可信度。

---

## Phase 4 — Single-Agent Research Workflow

当系统拥有多个工具后，引入一个单 Agent。

Agent 可以根据问题动态决定：

```text
Local Retrieval
Web Search
继续搜索
生成回答
```

同时增加：

```text
State
Tool Result
Step Limit
Budget
Error Handling
```

是否需要更复杂 Agent Architecture，应由 Evaluation 决定。

---

## Phase 5 — Evidence-Grounded Research

从简单 Citation 进一步发展到结构化 Evidence。

可能引入：

```text
Source
Evidence
Claim
```

研究结果可以追踪：

```text
Answer / Decision
        ↓
Claim
        ↓
Evidence
        ↓
Source
```

该阶段主要解决：

- Unsupported Claim；
- Citation 不匹配；
- 多来源冲突；
- 证据不足但模型仍强行回答。

---

## Phase 6 — Service Engineering

当核心能力稳定后，再增加面向实际使用的工程能力，例如：

```text
FastAPI
Persistence
Task State
Logging
Observability
Docker
Integration Tests
```

重点是：

```text
可靠运行
可恢复
可调试
可部署
```

而不是简单增加基础设施数量。

---

# 5. Multi-Agent 的定位

Multi-Agent 不是 EvidenceFlow 的默认目标。

只有当 Evaluation 表明：

1. 单 Agent 在复杂、可并行拆解的任务中存在稳定 Coverage 问题；
2. Query Decomposition、Iterative Research 等更简单方案不能解决；
3. 并行研究能够通过实验证明明显改善质量；

才考虑引入 Multi-Agent。

因此：

```text
Multi-Agent
```

是潜在优化方案，而不是项目起点。

---

# 6. RAG 与 Agent 的关系

EvidenceFlow 将两者视为不同层次。

RAG 解决：

> 从大量知识中找到应该提供给模型的信息。

Agent 解决：

> 当前下一步应该执行哪个动作或调用哪个工具。

例如：

```text
Agent
  │
  ├── KnowledgeBaseTool
  │       ↓
  │      RAG
  │
  └── WebSearchTool
```

因此 RAG 可以独立存在，并作为后续 Agent 的一个工具能力。

---

# 7. 项目长期评价标准

EvidenceFlow 最终不以：

```text
用了多少框架
有多少 Agent
代码有多少行
```

作为主要完成标准。

更重要的是：

```text
是否有清晰的问题定义
是否有稳定的数据合同
是否可以重复运行
是否有真实 Evaluation
是否能够解释失败案例
新技术是否有指标支持
系统是否能够稳定运行
```

---

# 8. 项目文档关系

项目使用三类核心文档：

```text
AGENTS.md
```

说明：

> Codex 应该如何参与项目开发。

---

```text
PROJECT_VISION.md
```

说明：

> 项目长期为什么做，以及大致可能发展到哪里。

---

```text
PROJECT_SPEC.md
```

说明：

> 当前阶段具体需要实现什么，以及当前已经冻结的设计规则。

优先关系：

```text
用户当前任务
    ↓
PROJECT_SPEC.md
    ↓
PROJECT_VISION.md
```

`PROJECT_VISION.md` 不应被直接用于推导当前实现需求。

---

# 9. 项目长期原则

EvidenceFlow 长期坚持以下原则：

### 1. Simple First

先完成最简单但完整的 Baseline。

### 2. Evaluation First

没有 Evaluation，不轻易增加复杂技术。

### 3. Problem Driven

技术必须解决明确问题。

### 4. Stable Contracts

先稳定数据和接口合同，再扩展实现。

### 5. Reproducible Experiments

实验配置、数据和指标必须能够重复。

### 6. Separate Domain from Infrastructure

业务模型与第三方基础设施保持边界。

### 7. Learn Through Implementation

项目不仅追求代码可运行，还要求能够解释每一个核心设计为什么存在。

---

# 10. 最终愿景

EvidenceFlow 最终希望形成一个可以清楚展示以下能力的完整工程项目：

```text
RAG
+
Retrieval Evaluation
+
LLM Application
+
Tool Using Agent
+
Evidence Grounding
+
Software Engineering
+
System Evaluation
```

项目的价值不来自复杂度本身，而来自：

> 每一层能力都有明确的问题背景、工程实现和 Evaluation 依据。
