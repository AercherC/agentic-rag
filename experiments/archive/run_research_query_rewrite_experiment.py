"""已冻结的 Query Rewrite / Query Expansion 历史实验。

问题背景
--------
失败归因显示真正的瓶颈在 recall：12 个 `candidate_retrieval_failure`
（Gold 从未进入 Top-20），多于排序失败（7）与分块失败（2）。重排/增强
实验已经证明：在候选集合不变的前提下，仅调整排序无法把覆盖增益转化为
Top-5 命中。因此下一步干预必须针对"召回"本身——本实验检验
**查询改写 / 查询扩展**能否把 Gold 拉进候选集合。

实验设计（受控 A/B，不改 Runtime）
----------------------------------
- 控制组 `control`：当前基线机制 = 中文原始分支 + 冻结人工英文翻译分支。
- 处理组 `treatment`：中文原始分支保持不变，仅把**翻译分支的英文 Query**
  换成 LLM 扩展后的英文检索 Query（Query Rewrite / Expansion，HyDE 风格）。
- 复用 `run_research_dense_baseline` 的共享契约（`_load_and_verify` /
  `_all_chunks` / `_gold_groups` / `_metrics` / `_source_ndcg` / `FrozenTranslationProvider`），
  只读冻结 40Q Gold，不修改任何生产代码。
- LLM 改写结果缓存到 `evals/research_retrieval/query_rewrite_cache_v1.json`，
  复跑时直接离线加载，保证实验产物可复现。
- 无 `LLM_API_KEY` 或网络不可达时优雅降级：对应 Query 回退到冻结翻译并标记
  `rewrite_failed`，此时该 Query 的 treatment 退化为 control（不会崩溃）。

判定
----
- Phase A（定向）：在控制组判定的 12 个候选失败题上，统计处理组把 Gold 拉进
  Top-20（不再 `candidate_retrieval_failure`）以及进一步拉进 Top-5 的题数。
- Phase B（全量）：对比控制组 / 处理组的 Hit@5、MRR、nDCG@20 与失败分布。
- Gate：`recovered_in_phase_a >= PHASE_A_GATE`（且全量 Hit@5 不劣化超过容差）
  才建议进入 Runtime 集成候选，否则停止。
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

from evidenceflow.config import Settings
from evidenceflow.domain.generation import LLMMessage
from evidenceflow.embedding import BgeM3EmbeddingProvider
from evidenceflow.eval.gold import gold_group_rank
from evidenceflow.llm.base import LLMConfigurationError, LLMError
from evidenceflow.llm.openai_compatible import OpenAICompatibleLLMClient
from evidenceflow.retrieval import DenseRetriever, MultilingualRetriever
from evidenceflow.storage import ChromaVectorStore
from evidenceflow.translation.base import TranslationError, TranslationResult

# 复用冻结基线的共享契约，避免各实验维护多份数据合同。
from experiments.run_research_dense_baseline import (  # noqa: E402
    DEPTHS,
    EXPECTED_COLLECTION,
    FrozenTranslationProvider,
    _all_chunks,
    _breakdowns,
    _gold_groups,
    _load_and_verify,
    _metrics,
    _source_ndcg,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = ROOT / "evals" / "research_retrieval" / "research_questions_v1.json"
OUTPUT_JSON = ROOT / "evals" / "research_retrieval" / "query_rewrite_experiment_v1.json"
OUTPUT_MD = ROOT / "evals" / "research_retrieval" / "query_rewrite_experiment_v1_report.md"
REWRITE_CACHE_PATH = ROOT / "evals" / "research_retrieval" / "query_rewrite_cache_v1.json"

PHASE_A_GATE = 4          # 12 个定向失败题中至少救回 4 个才放行（与增强实验阈值一致）。
HIT5_REGRESS_TOLERANCE = 0.025  # 全量 Hit@5 允许的最大劣化幅度。

_REWRITE_SYSTEM_PROMPT = """You are a retrieval query optimizer for an academic search engine over point-cloud completion papers.
Rewrite the user's search query into a single, richer English search query that:
1) preserves the original intent and question;
2) adds 2-4 high-signal technical synonyms or terms likely to appear verbatim in relevant academic abstracts or section titles (e.g. expand 点云补全 -> "point cloud completion", 缺失区域 -> "missing regions / incomplete scans", 对抗 -> "adversarial robustness");
3) does NOT attempt to answer the question;
4) returns ONLY the rewritten query, with no quotes, no numbering and no explanation."""


def _normalize(text: str) -> str:
    """与基线一致的确定性规范化，用于复用 chunk 匹配逻辑。"""

    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"(?<=\w)-\s*\n\s*(?=\w)", "", text)
    return re.sub(r"\s+", " ", text).strip().casefold()


def _runtime_fingerprint() -> dict[str, str]:
    """记录正式文件 Hash，证明本实验未接入 Runtime 且代码未漂移。"""

    files = (
        "PROJECT_SPEC.md",
        "src/evidenceflow/runtime.py",
        "src/evidenceflow/retrieval/retriever.py",
        "src/evidenceflow/retrieval/multilingual.py",
        "src/evidenceflow/retrieval/fusion.py",
        "src/evidenceflow/eval/lexical.py",
    )
    return {
        relative: hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        for relative in files
    }


class QueryRewriteProvider:
    """用 LLM 把翻译分支的英文 Query 改写为扩展检索 Query。

    只影响 MultilingualRetriever 的翻译（第二）分支文本；中文原始分支不变。
    无 LLM 能力或调用失败时回退到冻结人工翻译，并标记该 Query 为 rewrite_failed，
    确保实验在缺少 API Key / 离线环境下仍能跑完且结果可解释。
    """

    def __init__(
        self,
        llm_client: OpenAICompatibleLLMClient | None,
        translations_by_query: Mapping[str, str],
        cache: Mapping[str, str],
        cache_path: Path,
    ) -> None:
        """注入可选 LLM Client、冻结译文表与改写缓存。"""

        self._llm = llm_client
        self._frozen = dict(translations_by_query)
        self._cache = dict(cache)
        self._cache_path = cache_path
        # query -> "llm_rewrite" / "cache_hit" / "frozen_fallback"
        self.status: dict[str, str] = {}

    def _persist_cache(self) -> None:
        """把改写缓存落盘，供复跑离线加载。"""

        self._cache_path.write_text(
            json.dumps(self._cache, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )

    def translate_to_english(self, text: str) -> TranslationResult:
        """返回 LLM 扩展英文 Query；失败时安全回退冻结译文。"""

        if not isinstance(text, str) or not text.strip():
            raise TranslationError("query must not be blank")

        frozen_english = self._frozen.get(text, text)
        # 无 LLM 能力：直接回退冻结翻译，不调用任何外部服务。
        if self._llm is None:
            self.status[text] = "frozen_fallback"
            return TranslationResult(
                original_text=text, translated_text=frozen_english, was_translated=True
            )

        # 缓存命中：复跑时完全离线、结果可复现。
        if text in self._cache:
            self.status[text] = "cache_hit"
            return TranslationResult(
                original_text=text, translated_text=self._cache[text], was_translated=True
            )

        messages = [
            LLMMessage(role="system", content=_REWRITE_SYSTEM_PROMPT),
            LLMMessage(
                role="user",
                content=(
                    f"原始查询（中文）: {text}\n"
                    f"参考英文直译（仅作语义参考，不要照抄）: {frozen_english}\n"
                    "请输出扩展后的英文检索查询。"
                ),
            ),
        ]
        try:
            rewritten = self._llm.generate(messages).strip()
        except (LLMError, LLMConfigurationError) as exc:
            # 单题 LLM 失败不影响整体实验；回退冻结翻译并标记。
            self.status[text] = "frozen_fallback"
            return TranslationResult(
                original_text=text, translated_text=frozen_english, was_translated=True
            )
        if not rewritten or rewritten == text.strip():
            self.status[text] = "frozen_fallback"
            return TranslationResult(
                original_text=text, translated_text=frozen_english, was_translated=True
            )
        self._cache[text] = rewritten
        self.status[text] = "llm_rewrite"
        self._persist_cache()
        return TranslationResult(
            original_text=text, translated_text=rewritten, was_translated=True
        )


def _build_llm_client(settings: Settings) -> OpenAICompatibleLLMClient | None:
    """仅在配置了 API Key 时构造 LLM Client；否则返回 None（触发降级）。"""

    if not settings.llm_api_key or not settings.llm_api_key.strip():
        return None
    try:
        return OpenAICompatibleLLMClient(
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            model=settings.llm_model,
            temperature=settings.llm_temperature,
            timeout_seconds=settings.llm_timeout_seconds,
            max_attempts=settings.llm_max_attempts,
        )
    except LLMConfigurationError:
        return None


def _evaluate(
    question: Mapping[str, Any],
    chunks: Sequence[Any],
    results: Sequence[Any],
) -> dict[str, Any]:
    """对一组检索结果计算 Gold 来源排名、失败归因与 nDCG 指标。"""

    groups = _gold_groups(question, chunks)
    extraction_failure = False  # 构建阶段已逐页验证全部原文锚点。
    chunk_failure = any(not group for group in groups)
    ranked_ids = [result.chunk.chunk_id for result in results]
    rank = None if chunk_failure else gold_group_rank(ranked_ids, required_gold_groups=groups)
    if extraction_failure:
        primary_failure = "extraction_failure"
    elif chunk_failure:
        primary_failure = "chunk_failure"
    elif rank is None:
        primary_failure = "candidate_retrieval_failure"
    elif rank > 5:
        primary_failure = "ranking_failure"
    else:
        primary_failure = "success"
    row = {
        "question_id": question["question_id"],
        "query": question["query"],
        "gold_group_rank": rank,
        "primary_failure": primary_failure,
        "rewrite_status": None,  # 由调用方填充。
        # 与冻结基线 `_breakdowns` 同构：补齐分组维度字段，否则聚合阶段 KeyError。
        "language_group": question["language_group"],
        "question_type": question["question_type"],
        "evidence_type": question["evidence_type"],
        "gold_files": [source["file_name"] for source in question["gold_sources"]],
    }
    for depth in DEPTHS:
        row[f"ndcg_at_{depth}"] = _source_ndcg(results, groups, depth)
    return row


def _retriever_rows(
    questions: Sequence[Mapping[str, Any]],
    chunks: Sequence[Any],
    retriever: MultilingualRetriever,
    rewrite_status_map: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """对一组 Query 跑检索并逐题评估，返回与基线同构的逐题行。"""

    rows: list[dict[str, Any]] = []
    for index, question in enumerate(questions, start=1):
        results = retriever.retrieve(question["query"], top_k=20)
        row = _evaluate(question, chunks, results)
        if rewrite_status_map is not None:
            row["rewrite_status"] = rewrite_status_map.get(question["query"])
        rows.append(row)
        print(
            f"[{index:02d}/{len(questions)}] {question['question_id']} "
            f"rank={row['gold_group_rank']} {row['primary_failure']}",
            flush=True,
        )
    return rows


def _load_rewrite_cache() -> dict[str, str]:
    """加载已落盘的 LLM 改写缓存；不存在则返回空表。"""

    if REWRITE_CACHE_PATH.exists():
        return json.loads(REWRITE_CACHE_PATH.read_text(encoding="utf-8"))
    return {}


def _phase_a_table(control_rows: list[dict[str, Any]], treatment_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """构造 Phase A 定向失败题的处理组效果明细。"""

    control_by_id = {r["question_id"]: r for r in control_rows}
    treatment_by_id = {r["question_id"]: r for r in treatment_rows}
    table: list[dict[str, Any]] = []
    for qid, ctl in control_by_id.items():
        if ctl["primary_failure"] != "candidate_retrieval_failure":
            continue
        trt = treatment_by_id[qid]
        table.append(
            {
                "question_id": qid,
                "control_rank": ctl["gold_group_rank"],
                "control_failure": ctl["primary_failure"],
                "treatment_rank": trt["gold_group_rank"],
                "treatment_failure": trt["primary_failure"],
                "rewrite_status": trt.get("rewrite_status"),
                "effect": _effect_label(ctl, trt),
            }
        )
    return table


def _effect_label(ctl: Mapping[str, Any], trt: Mapping[str, Any]) -> str:
    """用简短标签描述处理组相对控制组在候选召回上的变化。"""

    if trt["primary_failure"] == "success" and ctl["primary_failure"] != "success":
        return "RECOVERED→Top5"
    if trt["primary_failure"] != "candidate_retrieval_failure" and ctl["primary_failure"] == "candidate_retrieval_failure":
        return "RECOVERED→Top20"
    if ctl["primary_failure"] == "candidate_retrieval_failure" and trt["primary_failure"] == "candidate_retrieval_failure":
        return "still_missed"
    return "no_change"


def _gate(
    phase_a: list[dict[str, Any]],
    control_overall: Mapping[str, Any],
    treatment_overall: Mapping[str, Any],
) -> dict[str, Any]:
    """根据 Phase A 救回数与全量 Hit@5 劣化给出 Gate 判定。"""

    recovered_top20 = sum(1 for r in phase_a if r["effect"].startswith("RECOVERED"))
    recovered_top5 = sum(1 for r in phase_a if r["effect"] == "RECOVERED→Top5")
    hit5_ctl = control_overall["hit_at_5"]
    hit5_trt = treatment_overall["hit_at_5"]
    hit5_regress = hit5_ctl - hit5_trt
    passed = recovered_top20 >= PHASE_A_GATE and hit5_regress <= HIT5_REGRESS_TOLERANCE
    return {
        "phase_a_targets": len(phase_a),
        "recovered_to_top20": recovered_top20,
        "recovered_to_top5": recovered_top5,
        "phase_a_gate_threshold": PHASE_A_GATE,
        "control_hit_at_5": hit5_ctl,
        "treatment_hit_at_5": hit5_trt,
        "hit5_regression": round(hit5_regress, 4),
        "hit5_regression_tolerance": HIT5_REGRESS_TOLERANCE,
        "passed": passed,
        "decision": (
            "PASSED → 建议作为 Runtime 集成候选（仍需真值抽取/真实端到端复核）"
            if passed
            else "NOT PASSED → 停止，不回写 Runtime"
        ),
    }


def _metric_line(name: str, metric: Mapping[str, Any]) -> str:
    """把一组核心指标渲染为 Markdown 表格行。"""

    return (
        f"| {name} | {metric['N']} | {metric['hit_at_1']:.3f} | {metric['hit_at_3']:.3f} | "
        f"{metric['hit_at_5']:.3f} | {metric['hit_at_10']:.3f} | {metric['hit_at_20']:.3f} | "
        f"{metric['mrr']:.3f} | {metric['ndcg_at_20']:.3f} |"
    )


def render_markdown(payload: Mapping[str, Any]) -> str:
    """生成面向阶段决策的 Query Rewrite 受控实验报告。"""

    header = "| Group | N | Hit@1 | Hit@3 | Hit@5 | Hit@10 | Hit@20 | MRR | nDCG@20 |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    lines = [
        "# Academic Research Query Rewrite Experiment v1",
        "",
        f"- Dataset hash: `{payload['dataset_hash']}`",
        "- 控制组：中文原始分支 + 冻结人工英文翻译分支（当前基线机制）。",
        "- 处理组：中文原始分支不变，仅把翻译分支英文 Query 换成 LLM 扩展检索 Query。",
        f"- LLM 改写状态：成功 {payload['rewrite_summary']['llm_rewrite']} / 缓存命中 "
        f"{payload['rewrite_summary']['cache_hit']} / 回退冻结 {payload['rewrite_summary']['frozen_fallback']} "
        f"（{'有 LLM 能力' if payload['llm_available'] else '无 LLM_API_KEY，处理组退化为控制组'}）。",
        "",
        "## Overall（控制组 vs 处理组）",
        "",
        header,
        divider,
        _metric_line("control", payload["control"]["breakdowns"]["overall"]),
        _metric_line("treatment", payload["treatment"]["breakdowns"]["overall"]),
    ]
    lines += ["", "## Failure distribution", ""]
    lines.append(f"- control: {payload['control']['failure_counts']}")
    lines.append(f"- treatment: {payload['treatment']['failure_counts']}")
    lines += ["", "## Phase A：12 个定向候选失败题的处理组效果", ""]
    lines.append("| question_id | control_rank | treatment_rank | effect | rewrite |")
    lines.append("|---|---:|---:|---|---|")
    for r in payload["phase_a"]:
        lines.append(
            f"| {r['question_id']} | {r['control_rank']} | {r['treatment_rank']} | "
            f"{r['effect']} | {r['rewrite_status']} |"
        )
    g = payload["gate"]
    lines += [
        "",
        "## Gate",
        "",
        f"- Phase A 定向题数：{g['phase_a_targets']}",
        f"- 救回至 Top-20：{g['recovered_to_top20']}（阈值 {g['phase_a_gate_threshold']}）",
        f"- 救回至 Top-5：{g['recovered_to_top5']}",
        f"- 控制组 Hit@5：{g['control_hit_at_5']:.3f}；处理组 Hit@5：{g['treatment_hit_at_5']:.3f}",
        f"- Hit@5 劣化：{g['hit5_regression']}（容差 {g['hit5_regression_tolerance']}）",
        f"- **判定：{g['decision']}**",
        "",
        "结论：瓶颈在 recall（候选生成失败）。本实验检验查询改写能否把 Gold 拉进候选集合；"
        "若 Gate 不通过，说明问题更可能在于语料覆盖 / chunk 切分 / 嵌入表示，而非查询表层改写。",
        "",
    ]
    return "\n".join(lines)


def run() -> dict[str, Any]:
    """执行 Query Rewrite 受控实验，返回带 Gate 判定的完整产物。"""

    dataset, translations_by_id, audit = _load_and_verify()
    os.environ["HF_HOME"] = str((ROOT / ".cache" / "huggingface").resolve())
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    settings = Settings()
    if (settings.collection_name, settings.chunk_size, settings.chunk_overlap) != (
        EXPECTED_COLLECTION,
        512,
        64,
    ):
        raise ValueError("active Settings do not match the frozen research baseline")
    store = ChromaVectorStore(settings.vector_db_path, settings.collection_name)
    if store._collection.count() != 361:
        raise ValueError("research collection must contain exactly 361 frozen chunks")
    chunks = _all_chunks(store)

    translations_by_query = {
        question["query"]: translations_by_id[question["question_id"]]
        for question in dataset["questions"]
    }
    dense = DenseRetriever(
        BgeM3EmbeddingProvider(settings.embedding_model),
        store,
        top_k=settings.multilingual_candidate_top_k,
    )
    frozen_provider = FrozenTranslationProvider(translations_by_query)
    llm_client = _build_llm_client(settings)
    rewrite_provider = QueryRewriteProvider(
        llm_client, translations_by_query, _load_rewrite_cache(), REWRITE_CACHE_PATH
    )

    control_retriever = MultilingualRetriever(
        dense,
        frozen_provider,
        candidate_top_k=settings.multilingual_candidate_top_k,
        final_top_k=settings.multilingual_final_top_k,
        rrf_k=settings.multilingual_rrf_k,
        original_weight=settings.multilingual_original_weight,
        translation_weight=settings.multilingual_translation_weight,
    )
    treatment_retriever = MultilingualRetriever(
        dense,
        rewrite_provider,
        candidate_top_k=settings.multilingual_candidate_top_k,
        final_top_k=settings.multilingual_final_top_k,
        rrf_k=settings.multilingual_rrf_k,
        original_weight=settings.multilingual_original_weight,
        translation_weight=settings.multilingual_translation_weight,
    )

    print("=== control (frozen translation branch) ===", flush=True)
    control_rows = _retriever_rows(dataset["questions"], chunks, control_retriever)
    print("=== treatment (LLM query rewrite branch) ===", flush=True)
    treatment_rows = _retriever_rows(
        dataset["questions"], chunks, treatment_retriever, rewrite_provider.status
    )

    control_breakdowns = _breakdowns(control_rows)
    treatment_breakdowns = _breakdowns(treatment_rows)
    phase_a = _phase_a_table(control_rows, treatment_rows)
    gate = _gate(phase_a, control_breakdowns["overall"], treatment_breakdowns["overall"])

    rewrite_summary = Counter(rewrite_provider.status.values())
    return {
        "evaluation_version": "research-query-rewrite-v1",
        "dataset_hash": dataset["dataset_hash"],
        "llm_available": llm_client is not None,
        "configuration": {
            "control": "chinese original branch + frozen human English translation branch",
            "treatment": "chinese original branch unchanged; translation branch = LLM-expanded English query",
            "llm_model": settings.llm_model if llm_client is not None else None,
            "embedding_model": settings.embedding_model,
            "collection_name": settings.collection_name,
            "candidate_top_k_per_branch": settings.multilingual_candidate_top_k,
            "final_runtime_top_k": settings.multilingual_final_top_k,
            "rrf_k": settings.multilingual_rrf_k,
            "weights": [settings.multilingual_original_weight, settings.multilingual_translation_weight],
            "phase_a_gate": PHASE_A_GATE,
            "hit5_regression_tolerance": HIT5_REGRESS_TOLERANCE,
        },
        "rewrite_summary": {
            "llm_rewrite": rewrite_summary.get("llm_rewrite", 0),
            "cache_hit": rewrite_summary.get("cache_hit", 0),
            "frozen_fallback": rewrite_summary.get("frozen_fallback", 0),
        },
        "control": {
            "breakdowns": control_breakdowns,
            "failure_counts": dict(Counter(r["primary_failure"] for r in control_rows)),
            "question_results": control_rows,
        },
        "treatment": {
            "breakdowns": treatment_breakdowns,
            "failure_counts": dict(Counter(r["primary_failure"] for r in treatment_rows)),
            "question_results": treatment_rows,
        },
        "phase_a": phase_a,
        "gate": gate,
        "runtime_fingerprint": _runtime_fingerprint(),
    }


def main() -> None:
    """运行实验并将完整 JSON 与精简 Markdown 同步落盘。"""

    payload = run()
    OUTPUT_JSON.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    OUTPUT_MD.write_text(render_markdown(payload), encoding="utf-8")
    print(json.dumps(
        {
            "llm_available": payload["llm_available"],
            "rewrite_summary": payload["rewrite_summary"],
            "control_overall": payload["control"]["breakdowns"]["overall"],
            "treatment_overall": payload["treatment"]["breakdowns"]["overall"],
            "gate": payload["gate"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
