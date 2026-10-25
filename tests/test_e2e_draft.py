"""Fake-based tests for the derived question-set tooling (draft.py) and the CLI (scripts/e2e_draft.py).

No case touches the network: question drafting uses ``--dry-run`` and probes use an injected fake client;
Chroma / Settings / source-document loading are replaced with monkeypatch in the integration tests.
"""

import json

import pytest

import scripts.e2e_draft as module
from evidenceflow.domain.chunk import DocumentChunk
from evidenceflow.domain.document import Document, DocumentSegment
from evidenceflow.eval.e2e import draft
from evidenceflow.eval.stable_mapping import validate_source

# Three realistic-looking sentences that pass the "self-sufficient candidate sentence" filter
# (60-320 characters, not starting with a pronoun, containing a number or consecutive proper
# nouns).
SENTENCE_PERCENT = (
    "Additionally, we randomly set 1% of the measurements to dmax = 1.6, "
    "and each prediction takes only 0.0012s on a Nvidia GeForce 1080Ti GPU."
)
SENTENCE_CHAMFER = (
    "SnowflakeNet reports a chamfer distance of 7.21 on the PCN benchmark "
    "under the standard evaluation protocol."
)
SENTENCE_GPU = (
    "The Nvidia GeForce GPU is used for all experiments in this paper "
    "and is compared with the baseline method."
)


# ---------- normalization and the sentence pool ----------

def test_normalize_source_text_does_only_nfkc():
    # The "fi" ligature becomes "fi", but case and whitespace stay as they are ((d) performs the
    # case/whitespace normalization).
    assert draft.normalize_source_text("e\uFB03cient") == "efficient"
    assert draft.normalize_source_text("WEIRD  spacing\nkept") == "WEIRD  spacing\nkept"


def test_sentence_candidates_filters_and_merges_lines():
    assert draft.sentence_candidates(SENTENCE_PERCENT) == [SENTENCE_PERCENT]

    # Starting with a pronoun/connective, being too short, having no number and no consecutive
    # proper
    # nouns, or being too long must all be filtered out.
    pronoun = "We propose a method that uses 42 layers and achieves strong results on the benchmark."
    too_short = "We set 1%."
    no_number_no_proper = (
        "the model is trained with a large batch size for many epochs and converges quickly "
        "without any tuning on the development set"
    )
    too_long = "The 42 samples " + "x" * 320
    text = "\n".join([pronoun, too_short, no_number_no_proper, too_long])
    assert draft.sentence_candidates(text) == []

    # The same sentence wrapped across lines must be merged into one line; duplicate sentences are
    # kept only once.
    wrapped = (
        "Real-world 3D data are often\nincomplete, causing loss in geometric and\n"
        "semantic information for 3D sensors."
    )
    merged = draft.sentence_candidates(wrapped)
    assert len(merged) == 1 and "\n" not in merged[0] and "often incomplete" in merged[0]
    assert draft.sentence_candidates(f"{SENTENCE_CHAMFER} {SENTENCE_CHAMFER}") == [SENTENCE_CHAMFER]

    # A sentence without numbers but with two consecutive capitalized proper nouns (Nvidia GeForce)
    # is also selectable.
    assert draft.sentence_candidates(SENTENCE_GPU) == [SENTENCE_GPU]


def test_select_spans_prefers_numbers_and_falls_back_to_entities():
    percent = draft.select_spans("we set 1% of the measurements to dmax")
    assert percent and percent[0].span == "1%" and percent[0].kind == "numeric"

    seconds = draft.select_spans("each prediction takes 0.0012s on a Nvidia GPU")
    assert seconds[0].span == "0.0012s" and seconds[0].kind == "numeric"
    assert seconds[0].span in seconds[0].sentence

    entity_only = draft.select_spans("SnowflakeNet is trained on PCN with chamfer distance")
    assert entity_only and entity_only[0].kind == "entity" and entity_only[0].span == "SnowflakeNet"
    # Every span must be a literal substring of its sentence, otherwise the anchor cannot prove
    # itself.
    for candidate in percent + seconds + entity_only:
        assert candidate.span in candidate.sentence


def test_split_sentences_folds_line_end_hyphenation():
    """Printed PDF hyphenation must be folded while splitting sentences, otherwise the grounding
    check can never pass (sample calibration defect A).
    """

    merged = draft.split_sentences(
        "The PN2 encoder follows the same archi-\ntecture as the SSG network in [38]."
    )
    assert merged == ["The PN2 encoder follows the same architecture as the SSG network in [38]."]
    assert "archi-" not in merged[0]

    # Wrapped lines with indentation must be folded too; a line break without a hyphen is still
    # merged on whitespace.
    assert draft.split_sentences("the im-\n    pressive result holds.") == ["the impressive result holds."]
    assert draft.split_sentences("often\nincomplete data.") == ["often incomplete data."]

    # Decimal numbers must not be split or folded incorrectly (regression protection).
    assert draft.split_sentences("It takes 0.0012s and scores 7.21 points.") == [
        "It takes 0.0012s and scores 7.21 points."
    ]


def test_select_spans_skips_stopwords_and_reference_numbers():
    """Sentence-initial function words, caption words, and bracketed citation numbers are never
    answers (sample calibration defect B).
    """

    # `The` is a sentence-initial function word, not a nameable answer; the code falls back to the
    # next real proper noun.
    stacked = draft.select_spans(
        "The stacked PN encoder used by FC, Folding, PCN-CD and PCN-EMD includes 2 PN layers."
    )
    assert [span.span for span in stacked] and "The" not in [span.span for span in stacked]

    # `Figure` is a caption word and `2` is a figure number, so the answer should be the real
    # component name MLP.
    figure = draft.select_spans(
        "As shown in the lower right of Figure 2, the seed point cloud and the shape code "
        "are sent to MLP respectively and then concatenated."
    )
    assert [span.span for span in figure] == ["MLP"]

    # `[38]` is a bibliography citation number and cannot serve as an answer.
    citation = draft.select_spans("The PN2 encoder follows the same architecture as the SSG network in [38].")
    assert "38" not in [span.span for span in citation]

    # Ordinary proper nouns can still be derived (regression protection).
    pointnet = draft.select_spans("PointNet++ is trained on ShapeNet.")
    assert pointnet[0].span == "PointNet++"

    # When nothing is usable an empty list is returned and the caller drops the case as no_span,
    # rather than inventing a key that can never match.
    assert draft.select_spans("The figure and the table are listed above.") == []


# ---------- answer_keys derivation table ----------

def test_derive_answer_keys_percent_and_unit_forms():
    percent = draft.derive_answer_keys("1%", "we set 1% of the measurements")
    assert len(percent) == 1
    assert {"1%", "1 percent"} <= set(percent[0])
    # For a Chinese query an extra Chinese percent alias is supplied.
    zh = draft.derive_answer_keys("1%", "we set 1% of the measurements", query_language="zh")
    assert "百分之1" in zh[0]

    seconds = draft.derive_answer_keys("0.0012s", "it takes 0.0012s on the GPU")
    assert {"0.0012s", "0.0012", "0.0012 s"} <= set(seconds[0])
    assert "1.5" in draft.derive_answer_keys("1.5mm", "the error is 1.5mm")[0]


def test_derive_answer_keys_drops_long_spans_and_short_aliases():
    # 41 characters => key_matchable=false; an empty list is returned.
    assert draft.derive_answer_keys("a" * 41, "x") == []
    # Every alias is at least 2 characters after normalization (the bare "1" is dropped; a known
    # limitation).
    for keys in (
        draft.derive_answer_keys("1%", "we set 1% of it", query_language="zh"),
        draft.derive_answer_keys("0.0012s", "it takes 0.0012s"),
        draft.derive_answer_keys("view-points", "the view-points are listed"),
    ):
        assert keys and all(len(draft.normalize_text(alias)) >= 2 for alias in keys[0])

    # A hyphenated span is given both in folded form and with a space.
    hyphen = draft.derive_answer_keys("view-points", "the view-points are listed")
    assert {"viewpoints", "view points"} <= set(hyphen[0])

    # (c)①: at least one alias comes from the source sentence itself.
    sentence = "we set 1% of the measurements"
    keys = draft.derive_answer_keys("1%", sentence)
    assert any(alias in draft.normalize_text(sentence) for alias in keys[0])


# ---------- 3-grams and consecutive word runs ----------

def test_three_gram_jaccard_and_longest_common_word_run():
    sentence = "the chamfer distance is 7.21 on pcn"
    assert draft.three_gram_jaccard(sentence, sentence) == 1.0
    assert draft.three_gram_jaccard(sentence, "completely unrelated words here") == 0.0
    # Both sides empty returns 0.0.
    assert draft.three_gram_jaccard("", "") == 0.0

    nine = "one two three four five six seven eight nine"
    assert draft.longest_common_word_run(nine, nine) == 9
    # Only 5 consecutive words are shared: §13.1(c')④ (threshold > 8) is not triggered.
    assert draft.longest_common_word_run(
        "alpha beta gamma delta epsilon zeta", "xx alpha beta gamma delta epsilon yy"
    ) == 5
    assert draft.longest_common_word_run(nine, "xx one two three four five six seven eight nine yy") == 9
    assert draft.longest_common_word_run("", "anything") == 0


# ---------- quotas ----------

def _quota_case(document_id="doc-1", question_type="method", language="en", tags=None):
    return {
        "document_id": document_id,
        "question_type": question_type,
        "query_language": language,
        "tags": tags or [],
    }


def test_plan_quotas_reports_violations_without_raising():
    cases = [_quota_case(tags=["easy"])]
    report = draft.plan_quotas(cases)
    assert report.ok is False
    # A single document / single question type / all three difficulty tiers are all short; the
    # Chinese-English balance is not triggered (|1-0| <= 2).
    assert any("document doc-1" in violation for violation in report.violations)
    assert any("question_type method" in violation for violation in report.violations)
    assert any("difficulty medium" in violation for violation in report.violations)
    assert report.counts["total"] == 1 and report.counts["difficulty"]["easy"] == 1

    # With no difficulty labels at all only one "missing labels" violation is reported; the
    # three-tier proportions are not fabricated.
    no_labels = draft.plan_quotas([_quota_case()])
    assert sum("difficulty labels missing" in violation for violation in no_labels.violations) == 1


# ---------- T5 near_miss ----------

def _t1_item(candidate_id, entity, *, source="derived_t1", accepted=True, qtype="method"):
    return {
        "candidate_id": candidate_id,
        "document_id": "d1",
        "chunk_id": "k1",
        "sentence": f"{entity} is used for point cloud completion in this paper.",
        "span": entity,
        "answer_keys": [[entity]],
        "question_type": qtype,
        "query_language": "en",
        "source": source,
        "decision": "accepted" if accepted else "discarded",
    }


def test_plan_near_miss_requires_entity_present_and_attribute_absent():
    corpus = "PointNet is trained on ShapeNet. The optimizer is Adam. VoxelNet uses voxel grids."
    items = [_t1_item("c1", "PointNet"), _t1_item("c2", "GhostNet")]
    candidates = draft.plan_near_miss_candidates(items, corpus_text=corpus, limit=5, seed=1)
    entities = {c.entity for c in candidates}
    # PointNet is in the corpus and does not co-occur with attributes such as "gpu model" inside
    # the
    # window, so a candidate is expected.
    assert "PointNet" in entities
    # GhostNet is not in the corpus, so it is out_of_corpus and cannot serve as near_miss.
    assert "GhostNet" not in entities
    # The attribute of every candidate really does not co-occur with its entity.
    for candidate in candidates:
        assert not draft._co_occurs(candidate.entity, candidate.attribute, corpus)


def test_plan_near_miss_skips_non_distinctive_entities():
    # Common nouns such as `Feature`/`Natural` are capitalized but carry no discriminating power,
    # so
    # asking about their optimizer is pointless.
    corpus = "Feature maps are extracted. Natural language is processed."
    items = [_t1_item("c1", "Feature"), _t1_item("c2", "Natural")]
    assert draft.plan_near_miss_candidates(items, corpus_text=corpus, limit=5, seed=1) == []


def test_plan_near_miss_ignores_non_t1_and_rejected_items():
    corpus = "PointNet is trained on ShapeNet."
    items = [
        _t1_item("c1", "PointNet", source="derived_t2"),
        _t1_item("c2", "PointNet", accepted=False),
    ]
    assert draft.plan_near_miss_candidates(items, corpus_text=corpus, limit=5, seed=1) == []


def test_plan_near_miss_is_deterministic_and_respects_limit():
    corpus = "PointNet PoinTr SeedFormer VoxelNet PCN are all methods. " * 3
    items = [_t1_item(f"c{i}", name) for i, name in
             enumerate(["PointNet", "PoinTr", "SeedFormer", "VoxelNet", "PCN"])]
    first = draft.plan_near_miss_candidates(items, corpus_text=corpus, limit=3, seed=1)
    again = draft.plan_near_miss_candidates(items, corpus_text=corpus, limit=3, seed=1)
    assert [c.entity for c in first] == [c.entity for c in again]
    assert len(first) == 3
    assert draft.plan_near_miss_candidates(items, corpus_text=corpus, limit=0, seed=1) == []


def test_build_near_miss_case_satisfies_unanswerable_shape():
    from evidenceflow.eval.e2e import dataset as dataset_module

    case = draft.build_near_miss_case(
        case_id="E2E-U001",
        query="What optimizer does PointNet use during training?",
        query_language="en",
        question_type="method",
        entity="PointNet",
        attribute="optimizer",
        approval_basis="gate.json#abc",
    )
    # Unanswerable shape: status, abstention semantics, kind, and no evidence groups (the
    # unanswerable_shape rule in dataset.py).
    assert case.expected_status == "insufficient_context"
    assert case.acceptable_abstention is False
    assert case.unanswerable_kind == "near_miss"
    assert case.required_evidence_groups == []
    assert case.required_claims == []
    assert "near_miss" in case.tags
    assert case.provenance.source == "derived_t5"

    dataset = dataset_module.E2EDataset(dataset_id="d", corpus_version="c", cases=[case])
    report = dataset_module.validate_dataset(
        dataset, lambda *args: "", require_approved=True, allow_machine=True
    )
    assert "unanswerable_shape" not in {issue.rule for issue in report.issues}


def test_near_miss_query_names_the_entity():
    candidate = draft.NearMissCandidate(
        base_candidate_id="c1", entity="PointNet", attribute="optimizer",
        question_type="method", query_language="en",
    )
    assert "PointNet" in draft.near_miss_query(candidate)
    zh = candidate.model_copy(update={"query_language": "zh"})
    assert "PointNet" in draft.near_miss_query(zh)


def test_case_id_ranges_do_not_collide_across_constructors():
    """T3 and T6 share the E2E-M* numbering space, so they must continue from each other without
    collisions (the D2 run produced a duplicate E2E-M122).
    """

    from evidenceflow.eval.e2e import dataset as dataset_module

    document = make_document({1: f"{SENTENCE_PERCENT}\nfooter"})
    anchor = draft.build_evidence_alternative(
        document, sentence=SENTENCE_PERCENT, page_start=1, page_end=1
    )
    base = [
        draft.build_case(
            case_id=f"E2E-M{index:03d}", family_id="f", query=f"Q{index}?",
            query_language="en", question_type="method", sentence=SENTENCE_PERCENT,
            span="1%", answer_keys=[["1%"]], anchor=anchor, approval_basis="g#x",
        )
        for index in range(1, 3)
    ]
    # T6 continues after the base cases (start_index = len(cases) + 1) and produces 2 cases.
    false_premise, _ = module.build_false_premise_cases(
        [{"candidate_id": "c1", "document_id": document.document_id,
          "sentence": SENTENCE_PERCENT, "span": "1%", "answer_keys": [["1%"]],
          "query": "Which percentage?", "query_language": "en",
          "question_type": "quantitative_results", "source": "derived_t2",
          "page_start": 1, "page_end": 1, "decision": "accepted"}],
        count=2, documents={document.document_id: document}, chunks_by_id={},
        approval_basis="g#x", start_index=len(base) + 1,
    )
    all_cases = base + false_premise
    ids = [case.case_id for case in all_cases]
    assert len(ids) == len(set(ids)), f"duplicate case_id: {ids}"
    # It must also pass the validator's ID uniqueness rule.
    dataset = dataset_module.E2EDataset(dataset_id="d", corpus_version="c", cases=all_cases)
    report = dataset_module.validate_dataset(
        dataset, lambda *args: SENTENCE_PERCENT, require_approved=True, allow_machine=True
    )
    assert "duplicate_case_id" not in {issue.rule for issue in report.issues}


def test_case_ids_have_no_gaps_or_collisions_when_a_constructor_drops_items():
    """T3 drops cases internally, leaving gaps in the number range it emits; T6 must continue after
    the **highest number** and must not infer numbers from a count.

    Measured in D2: T3 produced 5 cases but the range was 113-127 (with gaps), and T6 started from
    len(cases)+1, overlapping the range already used by T3 and producing a duplicate E2E-M118.
    """

    from evidenceflow.eval.e2e import dataset as dataset_module

    document = make_document({1: f"{SENTENCE_PERCENT}\nfooter"})
    anchor = draft.build_evidence_alternative(
        document, sentence=SENTENCE_PERCENT, page_start=1, page_end=1
    )
    base = [
        draft.build_case(
            case_id=f"E2E-M{index:03d}", family_id="f", query=f"Q{index}?",
            query_language="en", question_type="method", sentence=SENTENCE_PERCENT,
            span="1%", answer_keys=[["1%"]], anchor=anchor, approval_basis="g#x",
        )
        for index in range(1, 4)  # number range 1..3
    ]
    # Simulate T3's "gaps in the number range": it allocated 5 numbers starting at 4 (4..8) but
    # kept
    # only 2 cases.
    next_number = len(base) + 1 + 5
    false_premise, _ = module.build_false_premise_cases(
        [{"candidate_id": "c1", "document_id": document.document_id,
          "sentence": SENTENCE_PERCENT, "span": "1%", "answer_keys": [["1%"]],
          "query": "Which percentage?", "query_language": "en",
          "question_type": "quantitative_results", "source": "derived_t2",
          "page_start": 1, "page_end": 1, "decision": "accepted"}],
        count=2, documents={document.document_id: document}, chunks_by_id={},
        approval_basis="g#x", start_index=next_number,
    )
    ids = [case.case_id for case in base + false_premise]
    assert len(ids) == len(set(ids)), f"duplicate case_id: {ids}"
    # T6 must land after 9 (4..8 are already taken by T3) and must not reuse 4.
    assert all(int(cid[5:]) >= 9 for cid in [c.case_id for c in false_premise])

    dataset = dataset_module.E2EDataset(
        dataset_id="d", corpus_version="c", cases=base + false_premise
    )
    report = dataset_module.validate_dataset(
        dataset, lambda *args: SENTENCE_PERCENT, require_approved=True, allow_machine=True
    )
    assert "duplicate_case_id" not in {issue.rule for issue in report.issues}


def test_quota_counts_final_dataset_not_just_accepted_items(tmp_path, monkeypatch):
    """Quotas must be based on the final question set: T3/T6 cases do not come from accepted_items,
    so computing them early under-reports (the D2 run missed 5 T3 cases).
    """

    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    dataset_path = tmp_path / "dataset.json"
    assert module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(dataset_path), "--dataset-id", "rag_e2e_v1",
        "--corpus-version", "test",
    ]) == 0

    report = json.loads(out.read_text(encoding="utf-8"))
    # Without --multi-doc the set contains no multi_document cases, so the quota must honestly
    # report 0.
    assert report["quota"]["counts"]["question_types"].get("multi_document", 0) == 0
    # The report must carry a draft question-set summary so the quota and the question set can be
    # cross-checked.
    assert report["dataset_draft"]["cases"] >= 1
    assert report["dataset_draft"]["multi_document"] == 0

    # The approval_basis of every case in the set must be the report path plus the real sha256 (not
    # a
    # placeholder value).
    import hashlib
    dataset = json.loads(dataset_path.read_text(encoding="utf-8"))
    digest = hashlib.sha256(out.read_bytes()).hexdigest()
    for case in dataset["cases"]:
        basis = case["provenance"]["approval_basis"]
        assert basis == f"gate.json#{digest}", basis
        assert "pending" not in basis


# ---------- T3 multi-document pairing ----------

def _t3_item(candidate_id: str, document_id: str, sentence: str, span: str,
             question_type: str = "method", *, accepted: bool = True) -> dict:
    """Build an accepted candidate that can take part in T3 pairing."""

    return {
        "candidate_id": candidate_id,
        "document_id": document_id,
        "sentence": sentence,
        "span": span,
        "answer_keys": [[span]],
        "question_type": question_type,
        "decision": "accepted" if accepted else "discarded",
    }


def test_pair_multi_document_candidates_requires_distinct_documents():
    items = [
        _t3_item("a1", "doc-a", "SnowflakeNet uses a Skip-Transformer decoder here.", "SnowflakeNet"),
        _t3_item("a2", "doc-a", "Another sentence from the same paper about PointNet.", "PointNet"),
        _t3_item("b1", "doc-b", "PoinTr proposes a set-to-set translation strategy.", "PoinTr"),
    ]
    pairs = draft.pair_multi_document_candidates(items, limit=5, seed=1)
    assert pairs, "should pair across documents"
    for pair in pairs:
        assert pair.candidate_a["document_id"] != pair.candidate_b["document_id"]


def test_pair_multi_document_candidates_requires_same_type_and_key_matchable():
    items = [
        _t3_item("a1", "doc-a", "SnowflakeNet uses a Skip-Transformer decoder.", "SnowflakeNet", "method"),
        _t3_item("b1", "doc-b", "PoinTr proposes set-to-set translation.", "PoinTr", "ablation"),
        _t3_item("c1", "doc-c", "SeedFormer proposes patch seeds for generation.", "SeedFormer", "method"),
    ]
    pairs = draft.pair_multi_document_candidates(items, limit=5, seed=1)
    # Only a1/c1 are both "method" and from different documents, so only they can pair; b1 has a
    # different question type and does not participate.
    assert len(pairs) == 1
    assert {pairs[0].candidate_a["document_id"], pairs[0].candidate_b["document_id"]} == {"doc-a", "doc-c"}
    assert pairs[0].question_type == "method"

    # A candidate with empty answer_keys cannot be paired (there is no way to verify "only half of
    # it
    # hits").
    no_keys = [
        _t3_item("a1", "doc-a", "SnowflakeNet uses a Skip-Transformer decoder.", "SnowflakeNet"),
        _t3_item("b1", "doc-b", "PoinTr proposes set-to-set translation.", "PoinTr"),
    ]
    no_keys[0]["answer_keys"] = []
    assert draft.pair_multi_document_candidates(no_keys, limit=5, seed=1) == []


def test_pair_multi_document_candidates_is_deterministic_and_respects_limit():
    items = [
        _t3_item(f"a{index}", "doc-a", f"Sentence A number {index} about PointNet.", f"PointNet{index}")
        for index in range(6)
    ] + [
        _t3_item(f"b{index}", "doc-b", f"Sentence B number {index} about PoinTr.", f"PoinTr{index}")
        for index in range(6)
    ]
    first = draft.pair_multi_document_candidates(items, limit=4, seed=1)
    again = draft.pair_multi_document_candidates(items, limit=4, seed=1)
    assert [p.candidate_a["candidate_id"] for p in first] == [p.candidate_a["candidate_id"] for p in again]
    assert len(first) == 4
    # The same sentence / span must not be reused in the output.
    spans = [p.candidate_a["span"] for p in first] + [p.candidate_b["span"] for p in first]
    assert len(spans) == len(set(spans))
    # limit<=0 returns an empty list directly.
    assert draft.pair_multi_document_candidates(items, limit=0, seed=1) == []


class _FakeSingleBlockClient:
    """T3 single-block probe: answers only the fact of the block it was asked about, used to verify
    the "only half hits" verdict.

    ``answers`` maps "a marker word in the context" to an answer; when given block A it answers A's
    fact, and when given block B it answers B's fact.
    """

    def __init__(self, answers, *, answer_both=False):
        self.answers = answers
        self.answer_both = answer_both
        self.last_usage = None
        self.calls = 0

    def generate_json(self, messages, *, schema_name, json_schema):
        self.calls += 1
        user = messages[1].content
        if self.answer_both:
            # Both blocks' facts can be answered: the question does not really need the second
            # block.
            return json.dumps({"verdict": "answered", "answer": " ".join(self.answers.values()),
                               "reason": "both facts present"})
        for marker, answer in self.answers.items():
            if marker in user:
                return json.dumps({"verdict": "answered", "answer": answer, "reason": "marker hit"})
        return json.dumps({"verdict": "insufficient_context", "answer": "", "reason": "no answer"})


def _multi_doc_pair_items():
    """Two candidates from different documents, of the same question type, both key_matchable."""

    return [
        {"candidate_id": "a1", "document_id": "d1", "chunk_id": "k1",
         "sentence": "SnowflakeNet uses a Skip-Transformer decoder for generation.",
         "span": "SnowflakeNet", "answer_keys": [["SnowflakeNet"]],
         "question_type": "method", "query": "Which decoder does SnowflakeNet use?",
         "query_language": "en", "page_start": 1, "page_end": 1, "decision": "accepted"},
        {"candidate_id": "b1", "document_id": "d2", "chunk_id": "k2",
         "sentence": "PoinTr proposes a set-to-set translation strategy for completion.",
         "span": "PoinTr", "answer_keys": [["PoinTr"]],
         "question_type": "method", "query": "What strategy does PoinTr propose?",
         "query_language": "en", "page_start": 1, "page_end": 1, "decision": "accepted"},
    ]


def test_build_multi_document_cases_accepts_half_hit():
    from evidenceflow.domain.evaluation_v2 import EvidenceLocator

    documents = {
        "d1": make_document({1: "SnowflakeNet uses a Skip-Transformer decoder for generation."},
                            document_id="d1", content_hash="h1"),
        "d2": make_document({1: "PoinTr proposes a set-to-set translation strategy for completion."},
                            document_id="d2", content_hash="h2"),
    }
    chunks = {"k1": {"text": "SnowflakeNet uses a Skip-Transformer decoder for generation."},
              "k2": {"text": "PoinTr proposes a set-to-set translation strategy for completion."}}
    client = _FakeSingleBlockClient({"Skip-Transformer": "SnowflakeNet", "set-to-set": "PoinTr"})
    results = []
    cases, failures = module.build_multi_document_cases(
        [draft.MultiDocPair(question_type="method",
                            candidate_a=_multi_doc_pair_items()[0],
                            candidate_b=_multi_doc_pair_items()[1])],
        client=client, documents=documents, chunks_by_id=chunks,
        approval_basis="gate.json#abc", probe_results=results,
    )
    assert failures == [] and len(cases) == 1
    case = cases[0]
    assert case.question_type == "multi_document"
    assert [g.group_id for g in case.required_evidence_groups] == ["G1", "G2"]
    assert [c.claim_id for c in case.required_claims] == ["C1", "C2"]
    assert "multi_doc" in case.tags
    # Each paper occupies one evidence group (a multi_doc validation requirement).
    assert {a.document_id for g in case.required_evidence_groups for a in g.alternatives} == {"d1", "d2"}
    assert {r["mode"] for r in results} == {"single_block_a", "single_block_b"}


def test_build_multi_document_cases_rejects_when_other_key_also_hits():
    documents = {
        "d1": make_document({1: "SnowflakeNet uses a Skip-Transformer decoder for generation."},
                            document_id="d1", content_hash="h1"),
        "d2": make_document({1: "PoinTr proposes a set-to-set translation strategy for completion."},
                            document_id="d2", content_hash="h2"),
    }
    chunks = {"k1": {"text": "SnowflakeNet uses a Skip-Transformer decoder for generation."},
              "k2": {"text": "PoinTr proposes a set-to-set translation strategy for completion."}}
    client = _FakeSingleBlockClient({"Skip-Transformer": "SnowflakeNet", "set-to-set": "PoinTr"},
                                    answer_both=True)
    cases, failures = module.build_multi_document_cases(
        [draft.MultiDocPair(question_type="method",
                            candidate_a=_multi_doc_pair_items()[0],
                            candidate_b=_multi_doc_pair_items()[1])],
        client=client, documents=documents, chunks_by_id=chunks, approval_basis="gate.json#abc",
    )
    assert cases == []
    assert failures and "t3_single_block_inconclusive" in failures[0]["reason"]


def test_multi_doc_requires_dataset_out(tmp_path, monkeypatch, capsys):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    # T3 was constructed but no output path was given: this must be refused, not silently
    # discarded.
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--multi-doc", "2",
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err


# ---------- T6 false-premise cases ----------

def test_replacement_number_follows_spec_rules():
    # Decimals: the last digit is +1 and the number of digits is unchanged (the integer part grows
    # on
    # a carry but the decimal places stay fixed).
    assert draft.replacement_number("0.19", context_text="") == "0.20"
    assert draft.replacement_number("1.99", context_text="") == "2.00"
    assert draft.replacement_number("2.5", context_text="") == "2.6"
    # Integers >= 10: +1.
    assert draft.replacement_number("128", context_text="") == "129"
    assert draft.replacement_number("256", context_text="") == "257"
    # Units and percent signs are preserved.
    assert draft.replacement_number("0.0012s", context_text="") == "0.0013s"
    assert draft.replacement_number("1.5mm", context_text="") == "1.6mm"
    assert draft.replacement_number("50%", context_text="") == "51%"
    # Integers < 10 are not selected (including `1%`: the numeric part is < 10).
    for value in ("7", "1%", "2"):
        with pytest.raises(ValueError):
            draft.replacement_number(value, context_text="")
    with pytest.raises(ValueError):
        draft.replacement_number("abc", context_text="")


def test_replacement_number_skips_values_already_in_context():
    # 129 is already in the context, so applying the rule again yields 130.
    assert draft.replacement_number("128", context_text="the layer has 129 units") == "130"
    # Still colliding after 3 attempts => ValueError, rather than returning a value already present
    # in
    # the source.
    with pytest.raises(ValueError):
        draft.replacement_number("128", context_text="129 130 131", max_attempts=3)


def test_build_false_premise_case_satisfies_d8():
    from evidenceflow.eval.e2e import dataset as dataset_module

    document = make_document({1: f"{SENTENCE_PERCENT}\nPage footer."})
    anchor = draft.build_evidence_alternative(
        document, sentence=SENTENCE_PERCENT, page_start=1, page_end=1
    )
    case = draft.build_false_premise_case(
        case_id="E2E-M900",
        family_id="derived_t6",
        query="Is it true that the model takes 0.0013s on a Nvidia GeForce 1080Ti GPU?",
        query_language="en",
        question_type="quantitative_results",
        sentence=SENTENCE_PERCENT,
        span="0.0012s",
        answer_keys=[["0.0012s", "0.0012"]],
        anchor=anchor,
        approval_basis="gate.json#abc",
        false_value="0.0013s",
    )
    # D8 shape: scored as an answerable case, abstention allowed, tagged, and unanswerable_kind
    # empty.
    assert case.expected_status == "answered"
    assert case.acceptable_abstention is True
    assert "false_premise" in case.tags
    assert case.unanswerable_kind is None
    # The required claim states the corrected fact, and the forbidden claim carries the
    # contradicting
    # anchor.
    assert len(case.required_claims) == 1 and case.required_claims[0].answer_keys
    assert len(case.forbidden_claims) == 1
    assert case.forbidden_claims[0].evidence_anchors
    assert "0.0013s" in case.forbidden_claims[0].text

    # Passes the D8 rule of validate_dataset (using the real document as the page-text provider).
    dataset = dataset_module.E2EDataset(
        dataset_id="d", corpus_version="c", cases=[case]
    )
    provider = lambda document_id, start, end: draft.page_text(document, start, end)  # noqa: E731
    report = dataset_module.validate_dataset(
        dataset, provider, require_approved=True, allow_machine=True, enforce_grounding=True
    )
    assert "false_premise" not in {issue.rule for issue in report.issues}


# ---------- question-type classification ----------

def test_classify_question_type_reaches_implementation():
    """`implementation` must be reachable: it is checked before `method`, otherwise it could never get
    a case (the quota would always fail).
    """

    assert draft.classify_question_type(
        "Our model is implemented in PyTorch with the official library.", "entity"
    ) == "implementation"
    # A pure method description is still classified as method (regression protection).
    assert draft.classify_question_type(
        "The architecture consists of an encoder and a decoder.", "entity"
    ) == "method"
    # A numeric span is always quantitative_results (existing semantics unchanged).
    assert draft.classify_question_type("we set 1% of the measurements", "numeric") == "quantitative_results"
    # `experiment_setup` is still checked before `implementation`: sentences with experiment
    # configuration such as GPU/dataset belong to experiment setup.
    assert draft.classify_question_type(
        "The model is implemented in PyTorch and trained on a single GPU.", "entity"
    ) == "experiment_setup"
    # Every one of the 8 question types must have a corresponding keyword table entry
    # (`multi_document` is constructed by T3 rather than keywords).
    keyword_types = {name for name, _ in draft._QUESTION_TYPE_KEYWORDS}
    assert keyword_types == set(draft.QUESTION_TYPES) - {"multi_document"}


# ---------- stratified sampling ----------
def _candidate(document_id: str, index: int, sentence: str, language: str = "en") -> dict:
    """Build one candidate-sentence row in pool form."""

    return {
        "candidate_id": f"{document_id}#{index}",
        "chunk_id": f"chunk-{document_id}-{index}",
        "document_id": document_id,
        "sentence": sentence,
        "page_start": 1,
        "page_end": 1,
        "query_language": language,
    }


# Two question-generating types: numeric sentences (quantitative_results) and proper-name sentences
# (method).
_NUMERIC_SENTENCE = "The model reduces the chamfer distance to 7.21 on the benchmark."
_ENTITY_SENTENCE = "SnowflakeNet uses a Skip-Transformer decoder for point generation."


def test_stratified_sample_spreads_across_documents():
    # 3 papers x 2 question types x 10 rows: prefix truncation would only take the first paper,
    # while
    # stratified sampling should reach all three.
    candidates = [
        _candidate(document, index, _NUMERIC_SENTENCE if index % 2 == 0 else _ENTITY_SENTENCE)
        for document in ("doc-a", "doc-b", "doc-c")
        for index in range(10)
    ]
    sample = draft.stratified_sample(candidates, limit=12, seed=1)
    assert len(sample) == 12
    assert len({row["document_id"] for row in sample}) == 3
    assert len({row["candidate_id"] for row in sample}) == 12


def test_stratified_sample_is_deterministic_and_seed_sensitive():
    candidates = [_candidate(f"doc-{index % 4}", index, _NUMERIC_SENTENCE) for index in range(40)]
    first = [row["candidate_id"] for row in draft.stratified_sample(candidates, limit=15, seed=1)]
    again = [row["candidate_id"] for row in draft.stratified_sample(candidates, limit=15, seed=1)]
    other = [row["candidate_id"] for row in draft.stratified_sample(candidates, limit=15, seed=2)]
    assert first == again          # the same seed gives element-wise identical results (recomputable)
    assert first != other          # a different seed gives an independent second sample


def test_stratified_sample_limits_and_no_span_bucket():
    candidates = [_candidate("doc-a", index, _NUMERIC_SENTENCE) for index in range(5)]
    # When limit exceeds the total, everything is returned without duplicates.
    assert len(draft.stratified_sample(candidates, limit=99, seed=1)) == 5
    assert draft.stratified_sample(candidates, limit=0, seed=1) == []
    with pytest.raises(ValueError):
        draft.stratified_sample(candidates, limit=-1, seed=1)

    # Sentences with no usable span go into the no_span stratum and rotate there on their own, so
    # they
    # do not crowd out the strata that can produce questions.
    mixed = candidates + [_candidate("doc-b", 99, "The figure and the table are listed above.")]
    sample = draft.stratified_sample(mixed, limit=6, seed=1)
    no_span = [row for row in sample if row["candidate_id"] == "doc-b#99"]
    assert len(no_span) <= 1


# ---------- anchor locating ----------

def make_document(pages: dict[int, str], *, document_id="doc-1", content_hash="hash-1") -> Document:
    """Build a small document with page-numbered Segments."""

    return Document(
        document_id=document_id,
        source_path="paper.pdf",
        file_name="paper.pdf",
        file_type="pdf",
        content_hash=content_hash,
        segments=[DocumentSegment(content=text, page_number=page) for page, text in sorted(pages.items())],
    )


def test_build_evidence_alternative_matches_validate_source():
    sentence = SENTENCE_CHAMFER
    document = make_document({1: f"1. Introduction\n{sentence}\nWe then describe the method."})
    anchor = draft.build_evidence_alternative(document, sentence=sentence, page_start=1, page_end=1)
    assert anchor.document_id == document.document_id
    assert anchor.content_version == document.content_hash
    assert anchor.locator.page_start == anchor.locator.page_end == 1
    assert anchor.locator.occurrence == 1
    assert anchor.locator.block_index is None and anchor.locator.span_start is None
    # The offsets must agree with what stable_mapping.validate_source returns (the same
    # normalization
    # algorithm).
    validation = validate_source(anchor, document)
    assert validation["source_valid"] is True
    assert (anchor.locator.source_span_start, anchor.locator.source_span_end) == (
        validation["source_span_start"],
        validation["source_span_end"],
    )
    assert anchor.locator.left_context and anchor.locator.right_context

    with pytest.raises(ValueError, match="not found"):
        draft.build_evidence_alternative(
            document, sentence="This sentence is not present in the page.", page_start=1, page_end=1
        )
    # A nonexistent page number is equally unlocatable.
    with pytest.raises(ValueError, match="not found"):
        draft.build_evidence_alternative(document, sentence=sentence, page_start=2, page_end=2)


def test_build_evidence_alternative_reuses_hyphen_folding():
    document = make_document({1: "We discuss the view-\npoints of the paper in detail."})
    anchor = draft.build_evidence_alternative(
        document, sentence="the viewpoints of the paper", page_start=1, page_end=1
    )
    validation = validate_source(anchor, document)
    assert validation["source_valid"] is True
    assert anchor.locator.source_span_start == validation["source_span_start"]


# ---------- CLI: three-row fixture ----------

def _write_chunk_fixture(path):
    rows = [
        {"chunk_id": "k1", "document_id": "d1", "text": SENTENCE_PERCENT, "page_start": 1, "page_end": 1},
        {"chunk_id": "k2", "document_id": "d2", "text": SENTENCE_CHAMFER, "page_start": 2, "page_end": 2},
        {"chunk_id": "k3", "document_id": "d3", "text": SENTENCE_GPU, "page_start": 3, "page_end": 3},
    ]
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def test_pool_from_json_reports_read_only_statistics(tmp_path):
    fixture = _write_chunk_fixture(tmp_path / "chunks.json")
    report_path = tmp_path / "pool.json"
    assert module.main(["pool", "--from-json", str(fixture), "--report", str(report_path)]) == 0
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "b3-pool-v1"
    stats = payload["stats"]
    assert stats["chunks"] == 3 and stats["sentences"] == 3
    assert stats["candidate_sentences"] == 3
    # Only the first sentence carries both a number and a unit (1% / 0.0012s).
    assert stats["numeric_unit_sentences"] == 1
    assert set(stats["per_document"]) == {"d1", "d2", "d3"}
    assert all(entry["chunks"] == 1 for entry in stats["per_document"].values())
    assert len(payload["candidates"]) == 3
    assert {"candidate_id", "chunk_id", "document_id", "sentence", "page_start", "page_end"} <= set(
        payload["candidates"][0]
    )

    # --limit only truncates the number of chunks.
    limited = tmp_path / "pool_limited.json"
    assert module.main(["pool", "--from-json", str(fixture), "--report", str(limited), "--limit", "1"]) == 0
    assert json.loads(limited.read_text(encoding="utf-8"))["stats"]["chunks"] == 1


def test_draft_dry_run_writes_expected_jsonl_fields(tmp_path):
    fixture = _write_chunk_fixture(tmp_path / "chunks.json")
    items_path = tmp_path / "items.jsonl"
    assert module.main(["draft", "--from-json", str(fixture), "--out", str(items_path), "--dry-run"]) == 0
    lines = [json.loads(line) for line in items_path.read_text(encoding="utf-8").splitlines() if line]
    assert len(lines) == 3 and all(row["decision"] == "accepted" for row in lines)
    required = {
        "candidate_id", "document_id", "chunk_id", "sentence", "span", "answer_keys", "query",
        "query_language", "question_type", "source", "generation", "decision",
    }
    for row in lines:
        assert required <= set(row)
        assert row["query"] == "<DRY-RUN>" and row["dry_run"] is True
        assert row["generation"]["prompt_version"] == "draft-v1"
        assert row["generation"]["usage"] is None
        assert row["source"] in {"derived_t1", "derived_t2"}
    # Chinese and English alternate by candidate index, so the split is naturally close to half and
    # half.
    assert {row["query_language"] for row in lines} == {"en", "zh"}


# ---------- CLI: output protection and the gates pre-check refusal ----------

def test_cli_refuses_outputs_under_evals(tmp_path, capsys):
    target_dir = module.ROOT / "evals" / "_b3a2_test_never_created"
    missing = tmp_path / "missing.json"
    commands = [
        ["pool", "--from-json", str(missing), "--report", str(target_dir / "pool.json")],
        ["draft", "--from-json", str(missing), "--out", str(target_dir / "items.jsonl")],
        ["gates", "--items", str(missing), "--out", str(target_dir / "gate.json")],
        ["report", "--gate", str(missing), "--out", str(target_dir / "gate.md")],
    ]
    for argv in commands:
        assert module.main(argv) == 2, argv
    captured = capsys.readouterr()
    assert captured.err.count("REFUSED:") == len(commands)
    # Output protection takes effect before any input is read: neither the target directory nor any
    # file should be created.
    assert not target_dir.exists()


def test_gates_refuses_same_vendor_before_reading_items(tmp_path, monkeypatch, capsys):
    # The judge model starts with "deepseek", so it is the same vendor; the refusal must happen
    # before
    # reading items or connecting to Chroma.
    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "https://api.deepseek.com")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "deepseek-chat")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "test-key-not-real")
    out = tmp_path / "gate.json"
    assert module.main(["gates", "--items", str(tmp_path / "missing.jsonl"), "--out", str(out)]) == 2
    captured = capsys.readouterr()
    assert "REFUSED:" in captured.err and "vendor" in captured.err
    assert not out.exists()


def test_report_renders_markdown(tmp_path):
    gate = {
        "gates_version": "b3-gates-v1",
        "collection": "col",
        "items": {"total": 1, "accepted": 1, "discarded": 0},
        "gates": {
            name: {"passed": 1, "failed": 0, "failures": []} for name in ("G3", "G4", "G5", "G6")
        },
        "quota": {
            "ok": False,
            "violations": ["document doc-1: 1 < 6"],
            "counts": {"total": 1, "documents": {"doc-1": 1}, "question_types": {"method": 1},
                       "languages": {"en": 1}, "difficulty": {"easy": 1}},
        },
        "discarded": [{"candidate_id": "k9#0", "stage": "draft", "reason": "no_span"}],
        "probe_config": {"judge_host": "open.bigmodel.cn", "judge_model": "glm-4-flash",
                         "generator_host": "api.deepseek.com", "generator_model": "deepseek-flash",
                         "same_vendor": False, "prompt_version": "probe-v1"},
        "rerun_consistency": {"available": False, "checked": 0, "consistent": 0,
                              "value": None, "note": "single-pass items"},
    }
    gate_path = tmp_path / "gate.json"
    gate_path.write_text(json.dumps(gate), encoding="utf-8")
    out = tmp_path / "gate.md"
    assert module.main(["report", "--gate", str(gate_path), "--out", str(out)]) == 0
    text = out.read_text(encoding="utf-8")
    assert "## 闸门通过率" in text and "| G3 | 1 | 0 | 100.0% |" in text
    assert "document doc-1: 1 < 6" in text and "k9#0" in text and "probe-v1" in text


# ---------- full gates path (fake client, no network) ----------

class _FakeProbeClient:
    """Answers with the tested value when the context contains it, and abstains otherwise; never
    goes online.
    """

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.last_usage = None
        self.calls = 0

    def generate_json(self, messages, *, schema_name, json_schema):
        self.calls += 1
        user = messages[1].content
        if "1%" in user:
            return json.dumps({"verdict": "answered", "answer": "1%", "reason": "context states 1%"})
        return json.dumps({"verdict": "insufficient_context", "answer": "", "reason": "no answer"})


def _prepare_gates_env(tmp_path, monkeypatch):
    """Set up the fake environment for gates: fake probe, fake Chroma, single-page document; returns
    (items_path, dense_path, query).
    """

    monkeypatch.setenv("EVAL_JUDGE_BASE_URL", "https://open.bigmodel.cn")
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "glm-4-flash")
    monkeypatch.setenv("EVAL_JUDGE_API_KEY", "test-key-not-real")

    page = f"{SENTENCE_PERCENT} The rest of the page is unrelated."
    document = make_document({1: page}, document_id="d1", content_hash="hash-1")
    gold_chunk = DocumentChunk(chunk_id="k1", document_id="d1", content_hash="hash-1",
                               file_name="paper.pdf", content=page, token_count=50, chunk_index=0,
                               page_start=1, page_end=1)
    other_chunk = DocumentChunk(chunk_id="k2", document_id="d1", content_hash="hash-1",
                                file_name="paper.pdf", content="A completely unrelated baseline paragraph.",
                                token_count=10, chunk_index=1, page_start=1, page_end=1)

    from evidenceflow.config import Settings

    tmp_chroma = tmp_path / "chroma"
    tmp_chroma.mkdir()
    monkeypatch.setattr(module, "Settings", lambda: Settings(
        _env_file=None, vector_db_path=str(tmp_chroma), source_docs_path=str(tmp_path / "docs"),
        llm_base_url="https://api.deepseek.com", llm_model="deepseek-flash",
    ))
    monkeypatch.setattr(module, "resolve_active_collection", lambda settings: "col")
    monkeypatch.setattr(module, "load_source_documents", lambda root: {document.document_id: document})
    monkeypatch.setattr(module, "OpenAICompatibleLLMClient", _FakeProbeClient)

    class FakeStore:
        def __init__(self, path, name, *, create_if_missing=True):
            assert create_if_missing is False

        def iter_all_chunks(self):
            yield gold_chunk
            yield other_chunk

    monkeypatch.setattr(module, "ChromaVectorStore", FakeStore)

    query = "What percentage of the measurements is randomly set to dmax?"
    item = {
        "candidate_id": "k1#0", "document_id": "d1", "chunk_id": "k1",
        "sentence": SENTENCE_PERCENT, "page_start": 1, "page_end": 1, "span": "1%",
        "answer_keys": [["1%", "1 percent"]], "query": query, "query_language": "en",
        "question_type": "quantitative_results", "source": "derived_t2",
        "generation": {"model": "deepseek-flash", "base_url_host": "api.deepseek.com",
                       "prompt_version": "draft-v1", "usage": None},
        "decision": "accepted", "decision_reason": None, "difficulty": None, "dry_run": True,
    }
    items_path = tmp_path / "items.jsonl"
    items_path.write_text(json.dumps(item) + "\n", encoding="utf-8")
    dense_path = tmp_path / "dense.json"
    dense_path.write_text(json.dumps({query: ["k2", "k1"]}), encoding="utf-8")
    return items_path, dense_path, query


def test_gates_pipeline_with_fake_probe(tmp_path, monkeypatch):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    code = module.main(["gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path)])
    assert code == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert {name: gate["failed"] for name, gate in report["gates"].items()} == {
        "G3": 0, "G4": 0, "G5": 0, "G6": 0
    }
    assert report["items"] == {"total": 1, "accepted": 1, "discarded": 0}
    assert report["probe_config"]["same_vendor"] is False
    assert report["quota"]["ok"] is False and "difficulty" in report["quota"]["counts"]
    modes = {entry["mode"] for entry in report["probe_results"]}
    assert modes == {"gold", "distractor"}
    # Without --rerun the consistency rate must not be fabricated: available=false and value=null.
    assert report["rerun_consistency"]["available"] is False
    assert report["rerun_consistency"]["value"] is None


# ---------- persisting the question-set draft ----------

def test_gates_dataset_out_writes_valid_dataset(tmp_path, monkeypatch):
    from evidenceflow.eval.e2e import dataset as dataset_module

    items_path, dense_path, query = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    dataset_path = tmp_path / "dataset.json"
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(dataset_path),
        "--dataset-id", "rag_e2e_v1",
        "--corpus-version", "test-corpus-version",
    ])
    assert code == 0 and dataset_path.is_file()

    # The fingerprint must self-verify, otherwise the identity binding between the question set and
    # the
    # mapping is untrustworthy.
    dataset = dataset_module.load_dataset(dataset_path, verify_fingerprint=True)
    assert dataset.dataset_id == "rag_e2e_v1"
    assert dataset.corpus_version == "test-corpus-version"
    assert len(dataset.cases) == 1
    case = dataset.cases[0]
    assert case.case_id == "E2E-M001"
    assert case.query == query
    assert case.expected_status == "answered"
    # approval_basis points at the report path plus the sha256 of that report's bytes (the report
    # is
    # written to disk first, then hashed).
    assert case.provenance.approval_basis.startswith("gate.json#")
    # Persisted with fixed LF: the dataset.sha256 standard depends on this.
    assert b"\r\n" not in dataset_path.read_bytes()


def test_gates_dataset_out_requires_dataset_id_and_corpus_version(tmp_path, monkeypatch, capsys):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    dataset_path = tmp_path / "dataset.json"
    # --dataset-out alone is not enough: the corpus_version standard is not settled, so the script
    # must not guess a default.
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(dataset_path),
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not dataset_path.exists()


def test_gates_dataset_out_refuses_evals_path(tmp_path, monkeypatch, capsys):
    items_path, dense_path, _ = _prepare_gates_env(tmp_path, monkeypatch)
    out = tmp_path / "gate.json"
    target = module.EVALS_ROOT / "_b3_never_created" / "dataset.json"
    code = module.main([
        "gates", "--items", str(items_path), "--out", str(out), "--dense", str(dense_path),
        "--dataset-out", str(target), "--dataset-id", "x", "--corpus-version", "y",
    ])
    assert code == 2
    assert "REFUSED" in capsys.readouterr().err
    assert not target.exists()


# ---------- rerun consistency ----------

def _rerun_item(candidate_id: str, span: str, keys: list[str], *, accepted: bool = True) -> dict:
    """Build a candidate in draft form for rerun-consistency comparison."""

    return {
        "candidate_id": candidate_id,
        "document_id": "d1",
        "sentence": SENTENCE_PERCENT,
        "span": span,
        "answer_keys": [keys],
        "decision": "accepted" if accepted else "discarded",
    }


def test_rerun_consistency_reports_full_agreement():
    first = [_rerun_item("k1#0", "1%", ["1%", "1 percent"]), _rerun_item("k1#1", "1.6", ["1.6"])]
    report, conflicts = module.compute_rerun_consistency(first, first)
    assert report["available"] is True
    assert report["checked"] == 2 and report["consistent"] == 2
    assert report["value"] == 1.0
    assert conflicts == []


def test_rerun_consistency_counts_conflicts_and_accepts_key_overlap():
    first = [
        _rerun_item("k1#0", "1%", ["1%", "1 percent"]),
        _rerun_item("k1#1", "1.6", ["1.6"]),
    ]
    second = [
        # Same span => consistent, even when the alias sets are not exactly identical.
        _rerun_item("k1#0", "1%", ["1 percent"]),
        # Different span with no alias overlap => conflict.
        _rerun_item("k1#1", "9.9", ["9.9"]),
    ]
    report, conflicts = module.compute_rerun_consistency(first, second)
    assert report["checked"] == 2 and report["consistent"] == 1
    assert report["value"] == 0.5
    assert conflicts == [
        {"candidate_id": "k1#1", "stage": "rerun", "reason": "rerun_conflict"}
    ]


def test_rerun_consistency_ignores_single_sided_acceptance():
    first = [_rerun_item("k1#0", "1%", ["1%"]), _rerun_item("k1#1", "1.6", ["1.6"])]
    # The second pass dropped k1#1: it has nothing comparable, so it counts neither as checked nor
    # as
    # a conflict.
    second = [_rerun_item("k1#0", "1%", ["1%"]), _rerun_item("k1#1", "1.6", ["1.6"], accepted=False)]
    report, conflicts = module.compute_rerun_consistency(first, second)
    assert report["checked"] == 1 and report["consistent"] == 1
    assert conflicts == []


def test_rerun_consistency_empty_denominator_is_null():
    report, conflicts = module.compute_rerun_consistency(
        [_rerun_item("k1#0", "1%", ["1%"], accepted=False)], []
    )
    assert report["available"] is True
    assert report["checked"] == 0
    # An empty denominator returns null rather than a fabricated 0 (§13.1(e) standard).
    assert report["value"] is None
    assert conflicts == []
