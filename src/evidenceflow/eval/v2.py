"""Compute layered v2 metrics from frozen rankings; never calls a retriever or a real generation model."""

import json
from pathlib import Path

from evidenceflow.domain.evaluation_v2 import EvaluationDatasetV2, MappingArtifact
from evidenceflow.eval.metrics import custom_bundle_ndcg


def ranking_metrics(ranked_ids: list[str], groups: list[list[list[str]]], depths=(5, 20)) -> dict:
    """Compute evidence-set coverage; a fragmented group is only covered once all its fragments are present."""
    if not groups or any(not group or any(not bundle for bundle in group) for group in groups):
        raise ValueError("only fully resolvable groups can be scored")
    if len(set(ranked_ids)) != len(ranked_ids):
        raise ValueError("duplicate ranked chunk IDs")
    if not depths or len(set(depths)) != len(depths) or any(type(k) is not int or k <= 0 for k in depths):
        raise ValueError("depths must be unique positive integers")
    positions = {cid: rank for rank, cid in enumerate(ranked_ids, 1)}
    # A single bundle is an AND, multiple bundles in the same group are an OR; the coverage rank is
    # the earliest complete set.
    group_ranks = []
    for group in groups:
        ranks = [max(positions[cid] for cid in bundle) for bundle in group
                 if all(cid in positions for cid in bundle)]
        group_ranks.append(min(ranks) if ranks else None)
    known_ids = {cid for group in groups for bundle in group for cid in bundle}
    # Standard RR looks at the first relevant Chunk; the complete-group metrics separately wait for
    # all fragments.
    first = min((rank for cid, rank in positions.items() if cid in known_ids), default=None)
    complete = max(group_ranks) if all(rank is not None for rank in group_ranks) else None
    output = {"first_evidence_rank": first, "complete_evidence_rank": complete}
    for k in depths:
        covered = sum(rank is not None and rank <= k for rank in group_ranks)
        output.update({
            f"any_evidence_hit_at_{k}": float(covered > 0),
            f"evidence_group_recall_at_{k}": covered / len(groups),
            f"all_required_evidence_at_{k}": float(covered == len(groups)),
            f"first_relevant_rr_at_{k}": 1 / first if first is not None and first <= k else 0.0,
            f"custom_complete_evidence_rr_at_{k}": 1 / complete if complete is not None and complete <= k else 0.0,
            f"custom_evidence_group_ndcg_at_{k}": custom_bundle_ndcg(ranked_ids, groups, k),
        })
    return output


def evaluate_rankings(dataset: EvaluationDatasetV2, mapping: MappingArtifact,
                      rankings: dict[str, list[str]], *, depths=(5, 20), configuration=None) -> dict:
    """Per question, separate mapping failures, unanswerable questions, and scorable retrieval results."""
    if (dataset.dataset_id, dataset.corpus_version) != (mapping.dataset_id, mapping.corpus_version):
        raise ValueError("mapping targets a different dataset/corpus")
    expected = {(q.question_id, g.group_id) for q in dataset.questions for g in q.required_evidence_groups}
    mapped = {(g.question_id, g.group_id): g for g in mapping.groups}
    if set(mapped) != expected or set(rankings) != {q.question_id for q in dataset.questions}:
        raise ValueError("mapping/rankings must cover the dataset exactly")
    rows = []
    for q in dataset.questions:
        ids = rankings[q.question_id]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ranked chunk IDs")
        groups = [mapped[q.question_id, group.group_id] for group in q.required_evidence_groups]
        reason = "unanswerable" if q.answerability == "unanswerable" else (
            "gold_mapping_failure" if any(not g.chunk_sets for g in groups) else None)
        metrics = None if reason else ranking_metrics(ids, [g.chunk_sets for g in groups], depths)
        rows.append({"question_id": q.question_id, "family_id": q.family_id,
                     "query_language": q.query_language, "question_type": q.question_type,
                     "evidence_type": q.evidence_type, "ranked_chunk_ids": ids,
                     "returned_count": len(ids), "exclusion": reason, "metrics": metrics})
    def aggregate(selected):
        eligible = [row["metrics"] for row in selected if row["metrics"] is not None]
        result = {"question_count": len(selected), "eligible_count": len(eligible)}
        if eligible:
            for key in eligible[0]:
                if "_at_" in key:
                    aggregate_key = key.replace("first_relevant_rr_at_", "mrr_at_")
                    result[aggregate_key] = sum(row[key] for row in eligible) / len(eligible)
        return result
    total = len(mapping.groups)
    return {
        "evaluation_version": "2.0", "dataset_id": dataset.dataset_id, "split": dataset.split,
        "corpus_version": dataset.corpus_version, "index_version": mapping.index_version,
        "mapping_version": mapping.mapping_version, "configuration": dict(configuration or {}),
        "depths": list(depths), "per_question": rows, "aggregate": aggregate(rows),
        "slices": {field: {value: aggregate([row for row in rows if row[field] == value])
                           for value in sorted({row[field] for row in rows})}
                   for field in ("query_language", "question_type", "evidence_type")},
        "mapping_health": {"group_count": total,
                           "gold_mappability_rate": sum(bool(g.chunk_sets) for g in mapping.groups) / total if total else None,
                           "fragmented_evidence_rate": sum(g.status == "FRAGMENTED_ACROSS_CHUNKS" for g in mapping.groups) / total if total else None},
        "mapping_failures": [row["question_id"] for row in rows if row["exclusion"] == "gold_mapping_failure"],
        "answer_evaluation": None, "performance": None,
        "gate": "NOT_EVALUATED",
    }


def save_artifact(payload: dict, path: Path) -> None:
    """Create the versioned JSON exclusively so that historical experiments are never overwritten."""
    text = json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as output:
        output.write(text)


def evaluate_evidence_coverage(dataset: EvaluationDatasetV2, mapping: MappingArtifact,
                               rankings: dict[str, list[str]], *, depths=(5, 20)) -> dict:
    """Aggregate group-wise (micro) over both denominators; evidence the corpus cannot express counts as zero in system coverage."""
    if (dataset.dataset_id, dataset.corpus_version) != (mapping.dataset_id, mapping.corpus_version):
        raise ValueError("mapping targets a different dataset/corpus")
    if not depths or len(set(depths)) != len(depths) or any(type(k) is not int or k <= 0 for k in depths):
        raise ValueError("depths must be unique positive integers")
    mapped = {(g.question_id, g.group_id): g for g in mapping.groups}
    expected = {(q.question_id, g.group_id) for q in dataset.questions for g in q.required_evidence_groups}
    if set(mapped) != expected or set(rankings) != {q.question_id for q in dataset.questions}:
        raise ValueError("mapping and rankings must cover all canonical questions/groups")
    rows = []
    for q in dataset.questions:
        ids = rankings[q.question_id]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate ranked chunk IDs")
        groups = [mapped[q.question_id, g.group_id] for g in q.required_evidence_groups]
        eligible = [g for g in groups if g.chunk_sets]
        row = {"question_id": q.question_id, "canonical_group_count": len(groups),
               "eligible_group_count": len(eligible), "metrics": {}}
        for k in depths:
            prefix = set(ids[:k])
            hits = sum(any(set(bundle) <= prefix for bundle in g.chunk_sets) for g in eligible)
            row["metrics"][str(k)] = {
                "covered_groups": hits,
                "eligible_evidence_group_recall": hits / len(eligible) if eligible else None,
                "overall_evidence_coverage": hits / len(groups) if groups else None,
            }
        rows.append(row)
    canonical_count = sum(row["canonical_group_count"] for row in rows)
    eligible_count = sum(row["eligible_group_count"] for row in rows)
    aggregate = {"canonical_group_count": canonical_count, "eligible_group_count": eligible_count}
    for k in depths:
        hits = sum(row["metrics"][str(k)]["covered_groups"] for row in rows)
        aggregate[f"eligible_evidence_group_recall_at_{k}"] = hits / eligible_count if eligible_count else None
        aggregate[f"overall_evidence_coverage_at_{k}"] = hits / canonical_count if canonical_count else None
    return {"evaluation_version": "2.1", "aggregation": "micro_required_evidence_groups",
            "dataset_id": dataset.dataset_id, "corpus_version": dataset.corpus_version,
            "index_version": mapping.index_version, "depths": list(depths),
            "per_question": rows, "aggregate": aggregate}
