"""Evaluation v3: standard IR metrics plus multi-evidence diagnostics, orchestrated in one place.

Responsibility boundaries (the hard boundary for Phase 11A):
- evaluation arithmetic only; never calls a retriever, Reranker, Embedding, or Chroma;
- standard qrels are derived from the frozen Gold Mapping Artifact and never modify Canonical Gold;
- the run (rankings) is supplied by the caller as a replay of a frozen historical artifact, so
  this module does not re-run the Runtime;
- custom metrics are no longer primary: only evidence_group_recall / all_required_evidence /
  gold_mappability survive as diagnostics, explicitly marked custom=true.
"""

import hashlib
import json
from pathlib import Path
from typing import Any

from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.eval.ir_metrics import STANDARD_DEPTHS, evaluate_ir_run


def _sha256_file(path: Path) -> str:
    """Return the file's SHA-256 hex digest, used for artifact provenance and auditability."""

    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 16), b""):
            h.update(block)
    return h.hexdigest()


def build_qrels(mapping: MappingArtifact) -> dict[str, list[str]]:
    """Derive standard binary IR qrels from the frozen MappingArtifact.

    Rules:
    - only groups with status FULLY_MAPPED / FRAGMENTED_ACROSS_CHUNKS enter the qrels;
    - UNMAPPED / mapping failure never pretends to be relevant (it stays out of the qrels);
    - every verified Gold chunk (including each block of a FRAGMENTED cross-chunk set) becomes a
      rel=1 relevant unit;
    - an unlabelled chunk is never auto-created as relevant.
    """

    qrels: dict[str, list[str]] = {}
    for group in mapping.groups:
        mapped = group.status in {"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"}
        if not mapped:
            continue
        relevant = qrels.setdefault(group.question_id, [])
        for bundle in group.chunk_sets:
            for cid in bundle:
                if cid not in relevant:
                    relevant.append(cid)
    return qrels


def qrels_hash(qrels: dict[str, list[str]]) -> str:
    """Serialize the qrels deterministically, then hash them, so artifacts can be compared across runs."""

    payload = json.dumps(
        {qid: sorted(cids) for qid, cids in sorted(qrels.items())},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _diagnostic_block(
    ranked_ids: list[str], groups_chunk_sets: list[list[list[str]]], depths: tuple[int, ...]
) -> dict[str, float]:
    """Reuse v2.ranking_metrics for the multi-evidence diagnostics, extracting only the two fields whose semantics stay identical.

        If a group cannot be resolved (which should not happen, since the qrels only contain mapped
        groups), fall back to 0.
    """

    from evidenceflow.eval.v2 import ranking_metrics

    # Deduplicate while keeping first-appearance order, so an occasional duplicate in a historical
    # ranking cannot break strict validation.
    seen: set[str] = set()
    unique = [c for c in ranked_ids if not (c in seen or seen.add(c))]
    try:
        full = ranking_metrics(unique, groups_chunk_sets, depths=depths)
    except ValueError:
        return {
            f"evidence_group_recall_at_{k}": 0.0 for k in depths
        } | {f"all_required_evidence_at_{k}": 0.0 for k in depths}
    return {
        f"evidence_group_recall_at_{k}": full[f"evidence_group_recall_at_{k}"]
        for k in depths
    } | {
        f"all_required_evidence_at_{k}": full[f"all_required_evidence_at_{k}"]
        for k in depths
    }


def evaluate_v3(
    qrels: dict[str, list[str]],
    run: dict[str, list[str]],
    mapping: MappingArtifact,
    *,
    metric_depths: tuple[int, ...] = STANDARD_DEPTHS,
    diagnostic_depths: tuple[int, ...] = (5, 10, 20),
) -> dict:
    """Compute the v3 standard metrics plus multi-evidence diagnostics for one frozen ranking replay."""

    ir = evaluate_ir_run(qrels, run, depths=metric_depths)

    # Collect each question's chunk_sets by question_id (used for the multi-evidence diagnostics).
    by_question: dict[str, list[list[list[str]]]] = {}
    for group in mapping.groups:
        by_question.setdefault(group.question_id, []).append(group.chunk_sets)

    per_query = {}
    for qid, row in ir["per_query"].items():
        diag = _diagnostic_block(
            list(run.get(qid, [])), by_question.get(qid, []), diagnostic_depths
        )
        per_query[qid] = {
            "question_id": qid,
            "eligible": row["eligible"],
            "num_relevant": row["num_relevant"],
            "standard_metrics": row["metrics"],
            "diagnostics": diag,
        }

    # Aggregate the diagnostics over eligible queries only.
    eligible = [row for row in per_query.values() if row["eligible"]]
    diag_agg: dict[str, float] = {}
    for k in diagnostic_depths:
        for name in ("evidence_group_recall_at_", "all_required_evidence_at_"):
            key = f"{name}{k}"
            diag_agg[key] = sum(r["diagnostics"][key] for r in eligible) / len(eligible)

    total_groups = len(mapping.groups)
    mapped_groups = sum(
        1 for g in mapping.groups if g.status in {"FULLY_MAPPED", "FRAGMENTED_ACROSS_CHUNKS"}
    )
    gold_mappability_rate = mapped_groups / total_groups if total_groups else None

    return {
        "evaluation_version": "3.0",
        "depths": list(metric_depths),
        "diagnostic_depths": list(diagnostic_depths),
        "aggregate": {
            "question_count": ir["aggregate"]["question_count"],
            "eligible_count": ir["aggregate"]["eligible_count"],
            "precision_at_k": ir["aggregate"]["precision_at_k"],
            "recall_at_k": ir["aggregate"]["recall_at_k"],
            "mrr_at_k": ir["aggregate"]["mrr_at_k"],
            "map_at_k": ir["aggregate"]["map_at_k"],
            "ndcg_at_k": ir["aggregate"]["ndcg_at_k"],
            "diagnostics": diag_agg,
            "gold_mappability_rate": gold_mappability_rate,
        },
        "per_query": per_query,
        "excluded_queries": ir["excluded_queries"],
    }


def save_artifact(payload: dict, path: Path) -> None:
    """Write the versioned JSON. Only new directories such as evals/evaluation_v3/standard_ir_v1/
        are written; historical artifact directories such as evals/evaluation_v2/ are never touched
        (the no-overwrite rule of Step 13).
    """

    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as output:
        output.write(text)
