"""已冻结的 Phase 12 worksheet 历史脚本。

把逐题信息拼成人可判读的文本，供人工标注（计划 §5A / §14）使用：
  - 问题 + Gold answerability + required claims（含 acceptable evidence）
  - 实际进入 Context 的 chunk 原文（从 tmp/phase12_corpus_dump.json 映射）
  - Runtime 生成的答案 + 解析出的 citation

只读，不写评测产物；输出到 tmp/phase12_annotation_worksheet.txt。
"""
from __future__ import annotations

import json
import textwrap
from pathlib import Path

BASE = Path("evals/evaluation_v3/phase12_rag_quality_v1")
CORPUS = Path("tmp/phase12_corpus_dump.json")
OUT = Path("tmp/phase12_annotation_worksheet.txt")


def wrap(text: str, width: int = 108, indent: str = "      ") -> str:
    return textwrap.fill(text, width, initial_indent=indent, subsequent_indent=indent)


def main() -> None:
    dataset = json.loads((BASE / "generation_eval_dataset.json").read_text(encoding="utf-8"))
    outputs = json.loads((BASE / "runtime_outputs.json").read_text(encoding="utf-8"))
    corpus = json.loads(CORPUS.read_text(encoding="utf-8"))
    by_id = {c["chunk_id"]: c for c in corpus}
    out_by_q = {o["question_id"]: o for o in outputs["outputs"]}

    lines: list[str] = []
    lines.append(f"Phase 12 标注工作底稿  |  questions={len(dataset['questions'])}")
    lines.append("=" * 110)

    for q in dataset["questions"]:
        qid = q["question_id"]
        o = out_by_q[qid]
        lines.append("")
        lines.append(f"### {qid}  [{q['answerability']}]  type={q['question_type']}  "
                     f"q_lang={q['query_language']}  corpus_lang={q['corpus_language']}")
        lines.append(f"QUERY: {q['query']}")
        lines.append(f"REFERENCE: {q.get('reference_answer', '')}")
        lines.append("REQUIRED CLAIMS:")
        for c in q.get("required_claims", []):
            lines.append(f"  [{c['claim_id']}] {c['text']}")
            lines.append(f"        acceptable_evidence: {[e[:12] for e in c.get('acceptable_evidence', [])]}")
        lines.append(f"GENERATED STATUS: {o.get('generated_status')}  ok={o.get('ok')}")
        lines.append(f"ANSWER: {o.get('answer_text')}")
        lines.append(f"CITED LABELS: {o.get('cited_labels')}")
        lines.append("")
        lines.append("--- CONTEXT (实际进入 LLM 的 chunk 原文) ---")
        for cid in o.get("context_chunk_ids", []):
            c = by_id.get(cid)
            if c is None:
                lines.append(f"  [MISSING in corpus dump] {cid}")
                continue
            is_gold = any(cid in rc.get("acceptable_evidence", []) for rc in q.get("required_claims", []))
            tag = " <== GOLD EVIDENCE" if is_gold else ""
            lines.append(f"  chunk {cid[:12]}  {c['file_name']} p{c['page_start']}-{c['page_end']}{tag}")
            lines.append(wrap(c["content"]))
        lines.append("-" * 110)

    OUT.write_text("\n".join(lines), encoding="utf-8")
    print("worksheet ->", OUT, len(lines), "lines")


if __name__ == "__main__":
    main()
